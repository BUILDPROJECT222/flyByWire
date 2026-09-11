"""Reproducible full-graph causal checks, with every 10 ms count window saved.
No network or drone access. Pause other GPU workloads before timing.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import wgpu
from .core import DATA, ROOT
from .full_engine import FullEngine


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', type=int, default=30)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if not 10 <= args.batches <= 10000 or not 0 <= args.seed < 2**32:
        parser.error('Use 10–10000 batches and a uint32 seed.')
    out = ROOT/'experiments'/('full-validation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True)
    meta = json.loads((DATA/'full-brain.json').read_text())
    nodes = meta['nodes']
    selected = np.array([i for i, n in enumerate(nodes) if n['type'] == 'LC4'])
    descending = np.array([i for i, n in enumerate(nodes) if n['group'] == 'descending_neuron'])
    engine = FullEngine(np.load(DATA/'full-spiking.npz'))
    report = dict(scope='Full-graph causal smoke checks; not a flight or learning qualification',
                  adapter=dict(engine.adapter.info), wgpu=wgpu.__version__,
                  seed=args.seed, batches=args.batches, manifest=meta['manifest'],
                  files={str(p.relative_to(ROOT)):sha(p) for p in
                         [DATA/'full-spiking.npz', ROOT/'lab/full_engine.py',
                          ROOT/'lab/shaders/full-spiking.wgsl', Path(__file__).resolve()]},
                  cases={}, gates={})
    try:
        # Warm pipeline/readback before timing; each case starts from a reset.
        engine.batch(np.zeros(engine.n, np.float32))
        for name, hz, recurrence in [('quiet',0,True), ('isolated',120,False),
                                     ('intact',120,True), ('repeat',120,True)]:
            engine.reset()
            rates = np.zeros(engine.n, np.float32)
            rates[selected] = hz
            times, rows, totals, outside, outputs = [], [], [], [], []
            digest = hashlib.sha256()
            start = time.perf_counter()
            for batch in range(args.batches):
                before = time.perf_counter()
                counts = engine.batch(rates, recurrent=recurrence, seed=args.seed)
                times.append((time.perf_counter()-before)*1000)
                if not np.all(np.isfinite(counts)) or np.any(counts < 0):
                    raise RuntimeError('Invalid spike counts')
                digest.update(counts.tobytes())
                ids = np.flatnonzero(counts)
                rows.append(np.column_stack([np.full(len(ids),batch),ids,counts[ids]]).astype(np.uint32))
                total = int(counts.sum())
                totals.append(total)
                outside.append(total-int(counts[selected].sum()))
                outputs.append(int(counts[descending].sum()))
            elapsed = time.perf_counter()-start
            np.savez_compressed(out/(name+'.npz'), spikes=np.concatenate(rows),
                                wall_ms=np.array(times), selected=selected)
            report['cases'][name] = dict(input_hz=hz, recurrence=recurrence,
                neural_ms=args.batches*10, wall_seconds=elapsed,
                batch_p50_ms=float(np.percentile(times,50)), batch_p95_ms=float(np.percentile(times,95)),
                batch_max_ms=max(times), neural_to_wall_ratio=args.batches*.01/elapsed,
                count_digest=digest.hexdigest(), total_spikes=sum(totals),
                outside_stimulus_spikes=sum(outside), descending_spikes=sum(outputs))
            print(name, json.dumps(report['cases'][name]), flush=True)
        c = report['cases']
        report['gates'] = {
            'quiet_without_input':c['quiet']['total_spikes']==0,
            'stimulus_produces_spikes':c['isolated']['total_spikes']>0,
            'recurrence_off_blocks_spread':c['isolated']['outside_stimulus_spikes']==0,
            'recurrence_reaches_other_cells':c['intact']['outside_stimulus_spikes']>0,
            'reaches_descending_population':c['intact']['descending_spikes']>0,
            'same_device_repeatable':c['intact']['count_digest']==c['repeat']['count_digest'],
        }
        report['passed'] = all(report['gates'].values())
    except Exception as exc:
        report.update(passed=False,error=str(exc))
        raise
    finally:
        engine.close()
        (out/'report.json').write_text(json.dumps(report,indent=2))
        print('Report:',out/'report.json',flush=True)
    if not report['passed']:
        raise SystemExit('One or more full-graph gates failed')


if __name__ == '__main__':
    main()
