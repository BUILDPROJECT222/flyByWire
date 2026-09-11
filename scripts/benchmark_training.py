#!/usr/bin/env python3
"""Time compact CES readout training (CPU). Optional graded step probe if data present."""
import json, platform, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def host_info():
    info = dict(
        utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        python=platform.python_version(),
    )
    try:
        import os
        info['cpu_count'] = os.cpu_count()
    except Exception:
        pass
    return info

def bench_compact(generations=6, seed=7):
    import sys
    sys.path.insert(0, str(ROOT))
    from lab.train import train
    t0 = time.perf_counter()
    _, result = train(generations=generations, seed=seed)
    elapsed = time.perf_counter() - t0
    return dict(
        task='compact_ces_readout',
        generations=generations,
        wall_s=elapsed,
        best_reward=result['evaluation']['reward'] if 'evaluation' in result else result.get('history',[{}])[-1].get('reward'),
        history=result.get('history'),
        evaluation=result.get('evaluation'),
        run_id=result.get('run_id'),
    )

def bench_graded_steps(steps=40):
    import sys, numpy as np, json
    sys.path.insert(0, str(ROOT))
    data = ROOT/'data'/'malecns'
    if not (data/'full-graded.npz').exists():
        return dict(task='graded_steps', skipped=True, reason='full-graded.npz missing')
    from lab.graded_engine import GradedEngine
    from lab.full_vision import RetinaEncoder
    nodes=json.loads((data/'full-brain.json').read_text())['nodes']
    types=np.array([n['type'] for n in nodes])
    bias=np.full(len(nodes), .5, np.float32); bias[types=='R1-R6']=0
    with np.load(data/'full-graded.npz') as a: graph={k:a[k] for k in a.files}
    enc=RetinaEncoder(); eng=GradedEngine(graph,bias)
    try:
        drive=enc.encode(np.zeros((48,64),np.float32))/120
        # warmup
        eng.batch(drive, 4)
        t0=time.perf_counter()
        for _ in range(steps):
            eng.batch(drive, 4)
        elapsed=time.perf_counter()-t0
    finally:
        eng.close()
    return dict(task='graded_batch4_steps', steps=steps, wall_s=elapsed, ms_per_step=1000*elapsed/steps, neurons=len(nodes))

def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--tag', default='host'); p.add_argument('--generations', type=int, default=6)
    p.add_argument('--graded-steps', type=int, default=40); args=p.parse_args()
    out=ROOT/'experiments'/f"bench-train-{args.tag}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    out.mkdir(parents=True)
    report=dict(host=host_info(), compact=None, graded=None)
    print('host', report['host'], flush=True)
    report['compact']=bench_compact(args.generations)
    print('compact', report['compact']['wall_s'], flush=True)
    report['graded']=bench_graded_steps(args.graded_steps)
    print('graded', report['graded'], flush=True)
    (out/'report.json').write_text(json.dumps(report, indent=2))
    print('wrote', out/'report.json')

if __name__=='__main__':
    main()
