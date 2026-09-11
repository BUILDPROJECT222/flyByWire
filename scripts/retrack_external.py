#!/usr/bin/env python3
"""Offline external-camera retracker with reacquisition (numpy-fast)."""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
N = 16


def decode(path: Path, width: int = 320) -> np.ndarray:
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height', '-of', 'json', str(path)],
        capture_output=True, text=True, check=True,
    )
    meta = json.loads(probe.stdout)['streams'][0]
    w0, h0 = int(meta['width']), int(meta['height'])
    h = int(round(width * h0 / w0))
    proc = subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', str(path), '-vf', f'scale={width}:{h}',
         '-pix_fmt', 'gray', '-f', 'rawvideo', 'pipe:1'],
        capture_output=True, check=True, timeout=180,
    )
    return np.frombuffer(proc.stdout, np.uint8).reshape(-1, h, width).copy()


def sample(frame: np.ndarray, box):
    x, y, w, h = [float(v) for v in box]
    H, W = frame.shape
    if w < 10 or h < 10:
        return None
    x0, y0 = int(round(x)), int(round(y))
    x1, y1 = int(round(x + w)), int(round(y + h))
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H or x1 <= x0 or y1 <= y0:
        return None
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    # bilinear-ish via PIL resize (fast enough, C)
    small = np.asarray(Image.fromarray(patch).resize((N, N), Image.BILINEAR), dtype=np.float64).ravel()
    small -= small.mean()
    energy = float(np.dot(small, small))
    if energy / len(small) < 36:
        return None
    small /= np.sqrt(energy)
    return small


def search(frame, template, prev, origin, radius=24, scales=(0.9, 1.0, 1.1), step=2):
    candidates = []
    cx0 = prev[0] + prev[2] / 2
    cy0 = prev[1] + prev[3] / 2
    for scale in scales:
        w = prev[2] * scale
        h = prev[3] * scale
        if w < origin[2] * 0.45 or w > origin[2] * 2.5 or h < 10:
            continue
        for dy in range(-radius, radius + 1, step):
            for dx in range(-radius, radius + 1, step):
                box = (cx0 + dx - w / 2, cy0 + dy - h / 2, w, h)
                vals = sample(frame, box)
                if vals is None:
                    continue
                candidates.append((float(np.dot(vals, template)), box))
    candidates.sort(key=lambda c: -c[0])
    return candidates


def rival_of(best, cands):
    if not best:
        return None
    bx = best[1][0] + best[1][2] / 2
    by = best[1][1] + best[1][3] / 2
    thresh = max(8.0, min(best[1][2], best[1][3]) * 0.75)
    for score, box in cands[1:]:
        cx = box[0] + box[2] / 2
        cy = box[1] + box[3] / 2
        if ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5 > thresh:
            return (score, box)
    return None


def retrack_trial(trial: Path, out_dir: Path):
    tracking_path = next(trial.glob('external-*.tracking.json'))
    mp4 = next(p for p in trial.glob('external-*.mp4') if '.source.' not in p.name)
    cid = tracking_path.name.split('.')[0].replace('external-', '')
    live = json.loads(tracking_path.read_text())
    samples = live['samples']
    first = next(s for s in samples if s.get('box'))
    box0 = first['box']
    origin = tuple(float(v) for v in box0)
    prev = origin

    frames = decode(mp4, 320)
    # process every frame but print progress; videos are short (~few hundred frames)
    template = None
    start_idx = 0
    for i in range(min(40, len(frames))):
        template = sample(frames[i], origin)
        if template is not None:
            start_idx = i
            break
    if template is None:
        raise RuntimeError(f'no template for {trial.name}')

    results = []
    lost = False
    recoveries = 0
    tracked = 0
    t_report = time.time()
    for i in range(start_idx, len(frames)):
        if time.time() - t_report > 5:
            print(f'  {trial.name} frame {i}/{len(frames)} tracked={tracked} recoveries={recoveries}', flush=True)
            t_report = time.time()
        frame = frames[i]
        if lost:
            base = prev if prev else origin
            cands = search(frame, template, base, origin, radius=40, scales=(0.85, 1.0, 1.2), step=4)
            if not cands:
                cands = search(frame, template, origin, origin, radius=72, scales=(0.9, 1.0, 1.25), step=6)
            best = cands[0] if cands else None
            rival = rival_of(best, cands) if best else None
            if best and best[0] >= 0.70 and (rival is None or best[0] - rival[0] >= 0.07):
                lost = False
                recoveries += 1
                prev = best[1]
                if best[0] >= 0.84:
                    refreshed = sample(frame, prev)
                    if refreshed is not None:
                        template = 0.7 * template + 0.3 * refreshed
                        template /= np.linalg.norm(template)
                status, score, box = 'tracking', best[0], prev
            else:
                status, score, box = 'lost', (best[0] if best else 0.0), None
        else:
            cands = search(frame, template, prev, origin, radius=24, scales=(0.9, 1.0, 1.1), step=3)
            best = cands[0] if cands else None
            rival = rival_of(best, cands) if best else None
            if (not best) or best[0] < 0.58 or (rival and best[0] - rival[0] < 0.05):
                lost = True
                status, score, box = 'lost', (best[0] if best else 0.0), None
            else:
                prev = best[1]
                status, score, box = 'tracking', best[0], prev
                tracked += 1
                if score >= 0.9:
                    refreshed = sample(frame, prev)
                    if refreshed is not None:
                        template = 0.85 * template + 0.15 * refreshed
                        template /= np.linalg.norm(template)

        dx = dy = None
        if box is not None:
            dx = box[0] + box[2] / 2 - origin[0] - origin[2] / 2
            dy = box[1] + box[3] / 2 - origin[1] - origin[3] / 2
        results.append(dict(
            frame=i, status=status, score=float(score),
            box=[float(v) for v in box] if box else None, dx=dx, dy=dy,
        ))

    picks = np.linspace(start_idx, len(frames) - 1, 12).astype(int)
    tiles = []
    for idx in picks:
        rgb = np.stack([frames[idx]] * 3, axis=-1)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        r = results[idx - start_idx]
        if r['box']:
            x, y, w, h = r['box']
            color = (0, 255, 80) if r['status'] == 'tracking' else (255, 80, 80)
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        draw.text((4, 4), f"{idx} {r['status'][:4]} {r['score']:.2f}", fill=(255, 220, 0))
        tiles.append(img)
    sheet = Image.new('RGB', (tiles[0].width * 4, tiles[0].height * 3), (10, 12, 12))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 4) * tile.width, (i // 4) * tile.height))
    sheet_path = out_dir / f'{trial.name}-retrack-sheet.jpg'
    sheet.save(sheet_path, quality=90)

    out_json = trial / f'external-{cid}.retrack.json'
    summary = dict(
        trial=trial.name,
        capture_id=cid,
        frames=len(frames),
        start_idx=start_idx,
        tracked_frames=tracked,
        recoveries=recoveries,
        tracking_fraction=tracked / max(1, len(results)),
        live_lost_samples=sum(1 for s in samples if s.get('status') == 'lost'),
        live_samples=len(samples),
        sheet=str(sheet_path.relative_to(ROOT)),
        method='offline template + reacquisition + mild template refresh',
    )
    out_json.write_text(json.dumps(dict(summary=summary, origin=list(origin), samples=results), indent=2))
    return summary


def main():
    out = ROOT / 'experiments' / ('retrack-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    out.mkdir(parents=True)
    summaries = []
    seen = set()
    for tpath in sorted(ROOT.glob('recordings/trial-*/external-*.tracking.json')):
        trial = tpath.parent
        if trial.name in seen:
            continue
        seen.add(trial.name)
        print('retracking', trial.name, flush=True)
        t0 = time.time()
        try:
            s = retrack_trial(trial, out)
            s['elapsed_s'] = time.time() - t0
            summaries.append(s)
            print(json.dumps(s), flush=True)
        except Exception as e:
            summaries.append(dict(trial=trial.name, error=str(e)))
            print('ERROR', trial.name, e, flush=True)
    (out / 'summary.json').write_text(json.dumps(summaries, indent=2))
    print('DONE', out / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
