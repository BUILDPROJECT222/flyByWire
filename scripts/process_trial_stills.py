#!/usr/bin/env python3
"""Extract contact sheets from every trial flight.mp4 and fix stale status paths."""
import json, subprocess
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments' / 'preflight-stills'
OUT.mkdir(parents=True, exist_ok=True)

def sheet_for(mp4: Path, dest: Path, n=8):
    proc = subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', str(mp4), '-vf', 'fps=2,scale=320:240',
         '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1'],
        capture_output=True, check=True, timeout=120,
    )
    frames = np.frombuffer(proc.stdout, np.uint8)
    if frames.size == 0:
        return False
    frames = frames.reshape(-1, 240, 320, 3)
    picks = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    tiles = [Image.fromarray(frames[i]) for i in picks]
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new('RGB', (320 * cols, 240 * rows), (0, 0, 0))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 320, (i // cols) * 240))
    sheet.save(dest, quality=88)
    return True

summaries = []
for trial in sorted((ROOT / 'recordings').glob('trial-*')):
    mp4 = trial / 'flight.mp4'
    if not mp4.exists():
        continue
    dest = OUT / f'{trial.name}-onboard.jpg'
    ok = sheet_for(mp4, dest)
    # fix status.json droneControl paths
    for st in trial.glob('external-*.status.json'):
        data = json.loads(st.read_text())
        if isinstance(data.get('mp4'), str) and 'droneControl' in data['mp4']:
            data['mp4'] = data['mp4'].replace('/Users/sawaiz/droneControl/', '/Users/sawaiz/flyByWire/')
            st.write_text(json.dumps(data))
            print('fixed path', st)
    summaries.append(dict(trial=trial.name, onboard_sheet=str(dest.relative_to(ROOT)) if ok else None, ok=ok))
    print(trial.name, 'ok' if ok else 'fail')

(OUT / 'summary.json').write_text(json.dumps(summaries, indent=2))
print('wrote', OUT)
