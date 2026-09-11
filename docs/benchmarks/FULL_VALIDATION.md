# Full spiking causal validation — 10 September 2026

Runner: `python -m lab.validate_full --batches 50 --seed 7`.
Evidence: `experiments/full-validation-20260910T130400904083Z/`.

All six causal gates passed: quiet without input, stimulus produces spikes, disabling recurrence blocks spread, intact recurrence reaches other cells, descending-population activation, and same-device deterministic repeat.

| Condition | Total spikes / 500 ms neural | Spikes outside stimulus | Descending spikes | Batch p95 / 10 ms neural |
|---|---:|---:|---:|---:|
| No input | 0 | 0 | 0 | 12.09 ms |
| LC4, recurrence off | 5,913 | 0 | 0 | 9.31 ms |
| LC4, recurrence on | 64,096 | 57,843 | 3,686 | 17.22 ms |
| Identical repeat | 64,096 | 57,843 | 3,686 | 17.03 ms |

Intact run: 500 ms neural time in 817 ms wall time including count processing, about 0.612× real time. External input was 120 Hz to 126 LC4 neurons. The interactive worker was paused for the run and restored afterward. Graph, code and shader hashes, adapter and wgpu version are stored in the JSON report. The two intact count digests match exactly.

Compressed sparse recordings retain every 10 ms count window. Reconstructing all four recordings reproduced their original count digests and total spikes. This checks the saved artifacts, not just the in-memory results.

These are short, single-stimulus, single-seed smoke checks. They do not establish visual discrimination, learned behavior, biological fidelity, sustained performance, camera-to-action latency or flight readiness. Reaching descending neurons does not establish an appropriate or decodable movement command.
