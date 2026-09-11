# Scene generalization, continuous replay and latency — 10 September 2026

**Outcome: the frozen grating decoder fails the broader scene gate.** Recording playback and the full-network inference pipeline work, but their direction output cannot yet be used for navigation.

## Frozen scene tests

No parameters or readout weights were fitted or selected using these tests. Twenty new seeds (2000–2019), paired directions, were used for each moving condition. The same full graded graph and quadratic T4/T5 readout were retained.

| Input | Neural decoder | Independent image-translation reference |
|---|---:|---:|
| Translating blurred random textures | 20/40 (50%) | 40/40 |
| Moving light edge | 20/40 (50%) | 40/40 |
| Moving dark edge | 17/40 (42.5%) | 40/40 |

Texture and light-edge outputs were always rightward. The dark-edge model output was rightward on 37/40 clips. This is a strong distribution-dependent bias, not useful direction decoding.

Twenty stationary textures and twenty looming disks each produced 20/20 forced horizontal outputs. The existing binary decoder cannot express stationary, looming or unknown. These tests do not measure looming-direction accuracy; they expose unsupported categories. The image-translation reference is diagnostic, not a capacity-matched learned model.

The two stimulus-generator tests verify known direction with an independent shift matcher, stationary consistency, dimensions and luminance bounds. A test generator artifact or missing image motion does not explain the neural failure.

## Continuous synthetic motion

A 72-frame texture sequence at a nominal 10 fps contains stationary → rightward → leftward → stationary segments, without resetting the network at transitions. There were 26 windows fully contained in a moving segment with a complete 12-frame history. Accuracy was 50%; the leftward transition never produced a correct output during its segment, and stopping could not be reported because the decoder lacks a neutral class.

These overlapping windows are dependent. Their fraction correct is a descriptive sequence result, not a statistical confidence estimate. This sequence was run without real-time pacing; timestamps represent its input schedule.

## Full recording replay

All 123 frames obtained by sampling the already clockwise-rotated takeoff/landing MP4 at 10 fps were processed continuously through the same RGB-to-retina pathway. No second rotation was applied. There were 112 direction outputs after the initial history filled, and no reset from the current scene-change heuristic. About half the outputs were rightward.

The recording has no trustworthy pose/motion labels, so no accuracy is claimed. Image-translation estimates were logged as a reference, not treated as ground truth: camera rotation, object motion, blur and perspective can confound them.

Offline processing after decode had median 14.5 ms, p95 15.8 ms and maximum 16.3 ms per frame. This excludes video decode, pacing, display and the workspace's activity/highlight publication work. It is not the running UI pipeline latency.

## Running replay pipeline

The first approximately 30-second timing pilot sampled the latest state through HTTP and therefore missed some completed inferences. It was superseded by a second run with a bounded worker timing ring recording **every completed inference**. The final run contains 270 unique records, matching the processed-frame counter.

| Metric | Median | p95 | p99 | Observed maximum |
|---|---:|---:|---:|---:|
| Per-input computation/publication preparation | 39.3 ms | 50.2 ms | 65.3 ms | 248.9 ms |
| Decoded-frame receipt → inference completion | 56.9 ms | 74.8 ms | 85.5 ms | 255.6 ms |
| Input age when HTTP state is sampled | 104.1 ms | 168.0 ms | 249.5 ms | 309.9 ms |
| Actual 12-sample history span | 1,113.9 ms | 1,297.2 ms | 1,335.4 ms | 1,404.7 ms |

Approximately 30 seconds: 285 received camera-frame increments, 270 processed frames, 14 skipped frames counted within active processing spans. Source-boundary/accounting timing means these counters need not sum exactly. The workspace processes the latest frame; it does not build an unbounded queue.

History spans are measured between receipt times of the 12 processed input samples. They are **separate from** per-frame latency, not an extra fixed time to add blindly. Dropped frames stretch the window. At a nominal 10 fps the span would be 1.1 seconds.

The result was temporarily unavailable in 63/333 HTTP samples, including history refill around recording loops/resets. The preliminary and final runs both had latency outliers. These short runs do not qualify sustained thermal behavior, long-duration stability or a control deadline.

Camera exposure, RTSP/radio transport, browser display/paint and actuator response were not measured. The physical live camera was not tested. All replay/source mode settings were restored afterward.

## Artifacts

- Scene suite: `experiments/scene-validation-20260910T140646337963Z/report.json`
- Final replay timing: `experiments/replay-latency-20260910T141301277592Z/report.json`
- Superseded sampled timing pilot: `experiments/replay-latency-20260910T140947731561Z/report.json`
- Dashboard summary: `benchmarks/camera-validation.json`

Scene artifacts retain exact generated clips, seeds, code snapshots, graph/encoder/checkpoint hashes, predictions and timing. Continuous and recorded sequences include per-frame traces. Final timing includes all inference records plus HTTP samples. Diagnostic fields now include frame skips, current input age and measured history span.

## Next gate

Train and validate a broader readout on diverse textures, edges, speed/contrast changes and stationary scenes, with an explicit stationary/unknown output. Treat looming as a separate detection problem. Keep held-out scene families and independent motion references, and retain the complete connectome. Measure robustness to timing variation and resets rather than only reset clips.

The current grating-only decoder remains clearly labeled experimental; its failed texture/edge tests are now shown in the main view and Diagnostics. No physical flight or motor command was issued.
