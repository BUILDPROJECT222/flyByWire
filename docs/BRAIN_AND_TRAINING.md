# What flyByWire does with the fly brain

## Layman version

Scientists mapped almost every neuron and connection in a male fruit fly’s central nervous system (**MaleCNS**). flyByWire loads that wiring diagram onto your Mac’s GPU and treats it like a fixed “reservoir”:

1. Camera frames (live drone, recording, or fake test patterns) are turned into light for the fly’s photoreceptor cells.
2. Activity ripples through the **whole** annotated network (~166,700 neurons, ~25.6M connections) using a simple approximate dynamical model — **not** a claim that this is how a real fly thinks.
3. A tiny **artificial readout** (a small learned calculator sitting on top) looks mainly at motion-detector cells (T4/T5) and guesses “image sliding left vs right.”
4. That guess is **observation only**. It never moves the motors. Takeoff/land/E-STOP are a separate supervised controller.

So training here means: **keep the fly wiring frozen; only teach the small readout**. Using your flight recordings adds real camera footage as extra homework, labeled by a dumb pixel-motion checker (not by the stick commands), because natural video doesn’t come with ground-truth “left/right” tags.

## Technical version

| Layer | Role | Trained? |
|---|---|---|
| MaleCNS topology + contact counts | Measured connectome | No |
| Graded / spiking dynamics | Declared engineering approximation (Metal/WebGPU) | No (params can be searched separately) |
| Retina encoder | Maps 64×48 luminance into R1–R6 | Fixed mapping |
| Quadratic T4/T5 readout | Direction score from 12-frame activity | **Yes** |
| Compact CES readout (`lab/train.py`) | Planar arena exploration on a 1,252-neuron subgraph | **Yes** (separate toy task) |
| Flight UDP process | TC packets, leases, watchdogs | Hand-designed; not ML |

### This continuation run

Script: `lab/train_from_recordings.py`  
Output: `experiments/recording-train-20260911T015021606924Z`

- Synthetic supervised clips (textures / ON / OFF edges) for clean labels.
- All `recordings/trial-*/flight.mp4` plus the original takeoff/land MP4, split into 12-frame windows, **weakly labeled** by `scene_stimuli.image_motion`.
- Fit a mixed quadratic T4/T5 readout; compare to a synthetic-only readout on held-out windows.

**Mixed model**
- Synthetic train accuracy: 1.000
- Synthetic test accuracy: 0.854
- Recording train accuracy: 1.000
- Recording held-out accuracy: 0.375
- Windows: synthetic train 72, recording train 24, recording test 8, synthetic test 48
- Wall time: 32.1s

**Synthetic-only baseline on the same tests**
- Synthetic test: 0.875
- Recording held-out: 0.500

Interpretation: synthetic gratings remain easy; natural flight video is much harder. Mixing recordings into training changes recording held-out score relative to synthetic-only — still **not** a validated navigation signal (see earlier scene-validation failures on textures/stationary bias).

## Training hardware benchmark

Same compact CES job (`lab.train`, 6 generations, seed 7) and (on Mac) graded full-graph steps.

| Machine | Hardware | Compact CES 6 gens | Full graded (166,700 n) |
|---|---|---|---|
| MacBook Air | Apple M3, 8 CPU, 10 GPU cores, 8 GB | **7.29s** · eval reward 28.93 | **16.27 ms/step** (40× batch-4) |
| Grok Bot computer | Linux x86_64, 8× Xeon vCPU, 16 GB | **12.50s** · eval reward 28.93 | Skipped (no `full-graded.npz` / Metal) |

Compact CES is CPU-bound on the small visual subgraph; the M3 finished ~1.72× faster than the box Xeon VM here. Full-graph training/feature extraction for recordings belongs on the Mac’s Metal path (~16 ms per integration burst).

Reports: `experiments/bench-train-macbook-m3-20260911T015010/report.json`, `experiments/bench-train-box-xeon-20260911T015041/report.json`, `experiments/recording-train-20260911T015021606924Z/report.json`.


## Brain yaw assist (2026-09-11)

Trial profile `brain-yaw-assist-v2` lets the workspace motion score bias **yaw only**, with a deadzone and hard clamp (±6 protocol units). The UDP control process still owns every motor packet and never loads MaleCNS; assist rides shared memory from the observer thread.

Default offline webpage video is `recordings/demo-default.mp4` (copy of the strong `trial-20260911T004359…` onboard flight). Active readout weights: `experiments/active-quadratic-T4_T5.npz` (refreshed by `python3 -m lab.train_from_recordings`).

Retrain mixes synthetic clips, onboard `image_motion` windows, and external `*.retrack.json` dx signs when available.

### Retrain with flight + retrack (2026-09-11 evening)

Script: `lab/train_from_recordings.py`  
Active weights: `experiments/active-quadratic-T4_T5.npz`  
Run: `experiments/recording-train-20260911T020819752244Z`

- Mixed model recording held-out accuracy: **0.70** (was ~0.38 earlier today)
- Synthetic test: **0.85**
- Windows: 38 onboard `image_motion` + 4 external retrack-dx (strict); synthetic 72 train / 48 test
- Offline demo video: `data/demo-default.mp4` (onboard from trial `20260911T004359…`)
- Control: trial profile `brain-yaw-assist-v2` applies clamped yaw bias (±6) from motion score via shared memory; UDP process still never loads MaleCNS
