"""Train quadratic T4/T5 readout from synthetic clips + flight data.

MaleCNS graded graph stays frozen. Labels come from:
  1) synthetic scene_stimuli clips (clean ±1)
  2) onboard flight.mp4 windows via image_motion (weak)
  3) external retrack dx windows when *.retrack.json exists

Writes experiments/recording-train-<stamp>/ and refreshes
experiments/active-quadratic-T4_T5.npz for the workspace.
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .core import DATA, ROOT
from .graded_engine import GradedEngine
from .full_vision import RetinaEncoder
from .scene_stimuli import clip, image_motion
from .validate_full import sha

FRAMES = 12


def decode_mp4(path: Path, fps: int = 10) -> np.ndarray:
    process = subprocess.run(
        [
            'ffmpeg', '-v', 'error', '-i', str(path),
            '-vf', f'fps={fps},scale=320:240',
            '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1',
        ],
        capture_output=True, check=True, timeout=180,
    )
    return np.frombuffer(process.stdout, np.uint8).reshape(-1, 240, 320, 3)


def to_retina(frame: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(frame).resize((64, 48)), np.float32)


def quadratic_fit(features: np.ndarray, labels: np.ndarray, penalty: float = 1e-3):
    mean = features.mean(0)
    scale = np.maximum(features.std(0), 1e-4)
    x = (features - mean) / scale
    kernel = lambda a, b: (1 + a.astype(np.float64) @ b.astype(np.float64).T / x.shape[1]) ** 2
    dual = np.linalg.solve(kernel(x, x) + np.eye(len(x)) * penalty, labels.astype(np.float64))
    return dict(train=x, mean=mean, scale=scale, dual=dual)


def quadratic_score(model, features: np.ndarray) -> np.ndarray:
    z = (features - model['mean']) / model['scale']
    train = model['train']
    kernel = (1 + z.astype(np.float64) @ train.astype(np.float64).T / train.shape[1]) ** 2
    return kernel @ model['dual']


def windows_from_video(rgb: np.ndarray, label_fn):
    small = np.array([to_retina(f) for f in rgb])
    out = []
    for start in range(0, max(0, len(small) - FRAMES + 1), FRAMES):
        window = small[start:start + FRAMES]
        label = label_fn(window, start)
        if label == 0:
            continue
        out.append((window, float(label), start))
    return out


def retrack_labeler(samples, min_abs_delta=1.0):
    """Label from change in retrack dx over the window (not absolute dx from origin)."""
    by_frame = {int(s['frame']): s for s in samples if 'frame' in s}

    def label_fn(window, start):
        series = []
        for scale in (2, 1):
            series = []
            for i in range(FRAMES):
                hit = None
                for f in (scale * (start + i), scale * (start + i) + 1):
                    s = by_frame.get(f)
                    if s and s.get('dx') is not None and (s.get('status') == 'tracking' or s.get('box')):
                        hit = float(s['dx'])
                        break
                if hit is not None:
                    series.append(hit)
            if len(series) >= max(3, FRAMES // 4):
                break
        if len(series) < max(3, FRAMES // 4):
            return 0
        delta = series[-1] - series[0]
        if abs(delta) < min_abs_delta:
            # fall back to median step
            steps = np.diff(series)
            if len(steps) == 0:
                return 0
            delta = float(np.median(steps)) * max(1, len(steps))
            if abs(delta) < min_abs_delta:
                return 0
        return 1 if delta > 0 else -1

    return label_fn



def main():
    started = time.time()
    out = ROOT / 'experiments' / (
        'recording-train-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    )
    out.mkdir(parents=True)

    nodes = json.loads((DATA / 'full-brain.json').read_text())['nodes']
    types = np.array([x['type'] for x in nodes])
    indices = np.concatenate([
        np.flatnonzero(np.char.startswith(types, 'T4')),
        np.flatnonzero(np.char.startswith(types, 'T5')),
    ])
    encoder = RetinaEncoder()
    bias = np.full(len(nodes), 0.5, np.float32)
    bias[types == 'R1-R6'] = 0
    with np.load(DATA / 'full-graded.npz') as a:
        graph = {k: a[k] for k in a.files}
    engine = GradedEngine(graph, bias)

    report = dict(
        scope='Fixed full graded MaleCNS + quadratic T4/T5; onboard image_motion + external retrack dx',
        started_utc=datetime.now(timezone.utc).isoformat(),
        train_synthetic_seeds=list(range(500, 512)),
        test_synthetic_seeds=list(range(900, 908)),
        recording_paths=[],
        retrack_paths=[],
        notes=[
            'Connectome weights are not trained.',
            'Onboard labels: scene_stimuli.image_motion.',
            'External labels: sign(median retrack dx) when tracking.',
            'Active checkpoint copied to experiments/active-quadratic-T4_T5.npz.',
        ],
        hashes={
            'full-graded.npz': sha(DATA / 'full-graded.npz'),
            'full-retina.json': sha(DATA / 'full-retina.json'),
        },
    )

    try:
        baseline = engine.batch(np.zeros(len(nodes), np.float32), 200)

        def features_from_frames(frames_64: np.ndarray) -> np.ndarray:
            engine.reset(baseline)
            history = []
            for image in frames_64:
                v = engine.batch(encoder.encode(image) / 120, 4)
                history.append(v[indices])
            return np.concatenate([part.mean(0) for part in np.array_split(np.asarray(history), 3)])

        syn_x, syn_y = [], []
        for seed in report['train_synthetic_seeds']:
            for kind in ('texture', 'on_edge', 'off_edge'):
                for label in (-1, 1):
                    frames = clip(seed, kind, label)
                    syn_x.append(features_from_frames(frames))
                    syn_y.append(float(label))
                    print(f'synthetic train seed={seed} {kind} label={label}', flush=True)
        syn_x = np.asarray(syn_x)
        syn_y = np.asarray(syn_y)

        rec_x, rec_y, rec_meta = [], [], []

        paths = []
        demo = ROOT / 'recordings/demo-default.mp4'
        if demo.exists():
            paths.append(demo)
        takeoff = ROOT / 'recordings/20260909T192201Z/takeoff-land.mp4'
        if takeoff.exists():
            paths.append(takeoff)
        paths += sorted(ROOT.glob('recordings/trial-*/flight.mp4'))
        seen = set()
        for path in paths:
            key = path.resolve()
            if key in seen or not path.exists():
                continue
            seen.add(key)
            report['recording_paths'].append(str(path.relative_to(ROOT)))
            rgb = decode_mp4(path, fps=10)

            def motion_label(window, start):
                return image_motion(window)

            for window, label, start in windows_from_video(rgb, motion_label):
                rec_x.append(features_from_frames(window))
                rec_y.append(label)
                rec_meta.append(dict(
                    path=str(path.relative_to(ROOT)), start=start, label=label, source='image_motion',
                ))
                print(f'onboard {path.parent.name or path.name} start={start} label={label}', flush=True)

        for rpath in sorted(ROOT.glob('recordings/trial-*/external-*.retrack.json')):
            trial = rpath.parent
            mp4 = next((p for p in trial.glob('external-*.mp4') if '.source.' not in p.name), None)
            if mp4 is None:
                continue
            data = json.loads(rpath.read_text())
            samples = data.get('samples') or []
            if not samples:
                continue
            report['retrack_paths'].append(str(rpath.relative_to(ROOT)))
            rgb = decode_mp4(mp4, fps=10)
            label_fn = retrack_labeler(samples)
            for window, label, start in windows_from_video(rgb, label_fn):
                rec_x.append(features_from_frames(window))
                rec_y.append(label)
                rec_meta.append(dict(
                    path=str(mp4.relative_to(ROOT)), start=start, label=label, source='retrack_dx',
                ))
                print(f'retrack {trial.name} start={start} label={label}', flush=True)

        if not rec_x:
            raise RuntimeError('No labeled recording/retrack windows found')

        rec_x = np.asarray(rec_x)
        rec_y = np.asarray(rec_y)
        hold = np.zeros(len(rec_y), dtype=bool)
        hold[3::4] = True
        rec_train_x, rec_train_y = rec_x[~hold], rec_y[~hold]
        rec_test_x, rec_test_y = rec_x[hold], rec_y[hold]

        train_x = np.concatenate([syn_x, rec_train_x])
        train_y = np.concatenate([syn_y, rec_train_y])
        model = quadratic_fit(train_x, train_y)

        def accuracy(features, labels):
            scores = quadratic_score(model, features)
            pred = np.where(scores >= 0, 1, -1)
            return float(np.mean(pred == labels)), scores

        test_x, test_y = [], []
        for seed in report['test_synthetic_seeds']:
            for kind in ('texture', 'on_edge', 'off_edge'):
                for label in (-1, 1):
                    frames = clip(seed + 1000, kind, label)
                    test_x.append(features_from_frames(frames))
                    test_y.append(float(label))
        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y)

        syn_train_acc, _ = accuracy(syn_x, syn_y)
        syn_test_acc, _ = accuracy(test_x, test_y)
        rec_train_acc, _ = accuracy(rec_train_x, rec_train_y)
        rec_test_acc, rec_test_scores = accuracy(rec_test_x, rec_test_y)

        syn_only = quadratic_fit(syn_x, syn_y)

        def acc_with(m, features, labels):
            scores = quadratic_score(m, features)
            return float(np.mean(np.where(scores >= 0, 1, -1) == labels))

        report['results'] = dict(
            synthetic_train_n=int(len(syn_y)),
            recording_train_n=int(len(rec_train_y)),
            recording_test_n=int(len(rec_test_y)),
            synthetic_test_n=int(len(test_y)),
            retrack_windows=sum(1 for m in rec_meta if m['source'] == 'retrack_dx'),
            image_motion_windows=sum(1 for m in rec_meta if m['source'] == 'image_motion'),
            mixed_model=dict(
                synthetic_train_accuracy=syn_train_acc,
                synthetic_test_accuracy=syn_test_acc,
                recording_train_accuracy=rec_train_acc,
                recording_test_accuracy=rec_test_acc,
            ),
            synthetic_only_model=dict(
                synthetic_test_accuracy=acc_with(syn_only, test_x, test_y),
                recording_test_accuracy=acc_with(syn_only, rec_test_x, rec_test_y),
            ),
            elapsed_s=time.time() - started,
        )
        report['recording_windows'] = rec_meta
        np.savez_compressed(
            out / 'quadratic-T4_T5-mixed.npz',
            train=model['train'], mean=model['mean'], scale=model['scale'],
            dual=model['dual'], indices=indices,
        )
        np.savez_compressed(
            out / 'quadratic-T4_T5-synthetic-only.npz',
            train=syn_only['train'], mean=syn_only['mean'], scale=syn_only['scale'],
            dual=syn_only['dual'], indices=indices,
        )
        np.savez_compressed(
            out / 'splits.npz',
            syn_train_x=syn_x, syn_train_y=syn_y,
            syn_test_x=test_x, syn_test_y=test_y,
            rec_train_x=rec_train_x, rec_train_y=rec_train_y,
            rec_test_x=rec_test_x, rec_test_y=rec_test_y,
            rec_test_scores=rec_test_scores,
        )
        active = ROOT / 'experiments' / 'active-quadratic-T4_T5.npz'
        active.write_bytes((out / 'quadratic-T4_T5-mixed.npz').read_bytes())
        report['active_checkpoint'] = str(active.relative_to(ROOT))
    finally:
        engine.close()
        (out / 'report.json').write_text(json.dumps(report, indent=2))
        print('Report:', out / 'report.json', flush=True)
        print(json.dumps(report.get('results', report), indent=2), flush=True)


if __name__ == '__main__':
    main()
