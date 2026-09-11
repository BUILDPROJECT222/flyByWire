# Full-brain visual pilot — 10 September 2026

**Outcome: sensory-to-action qualification failed.** The offline image pipeline works, but this homogeneous spiking model and readout did not decode motion above chance. No drone commands were sent.

## Implemented

- `lab/full_vision.py`: fixed luminance encoder into 3,335 R1–R6 neurons; 42 of 3,377 receptors lack a qualifying measured mapping and receive no external image drive.
- Positions inferred from the strongest measured same-side L1/L2/L3 partner with published optic-lobe hex coordinates. Both eyes sample the same image; the left image is mirrored. This is an engineered camera mapping, not calibrated fly optics.
- All 166,700 network entries remain simulated. Only the external input population is restricted.
- `lab/vision_experiment.py`: full-graph motion pilot, trained direction readout, fixed-readout controls and existing drone MP4 replay.
- `lab/vision_pathway_probe.py`: diagnostic background-drive sweep.
- `lab/vision_baseline.py`: image-only translation control.
- Four encoder tests pass: black/white scaling, spatial sampling, RGB luminance and invalid-frame rejection. Prepared mapping indices are unique and image coordinates bounded.

## Experiment

Moving gratings at 12 frames/clip. Each frame drives two 10 ms neural batches, for 240 ms neural time/clip. Training: eight starting-phase seeds, two directions each (16 clips). Test: four separate phase seeds, two directions each (8 clips). Directions share a neural-noise seed within each pair, preventing class-specific RNG cues.

Readout: fixed-penalty ridge classifier using three temporal bins of descending-neuron spike counts. Training-only centering/scaling. No image features or labels feed directly into this neural readout. The frame-shuffle and transmission-off controls reuse the fitted classifier.

| Configuration | Intact | Frame order shuffled | Transmission disabled |
|---|---:|---:|---:|
| No lamina background drive | 4/8 (50%) | 4/8 | 4/8 |
| Exploratory 80 Hz lamina drive | 4/8 (50%) | 4/8 | 4/8 |

Chance is 50%. The second run followed diagnosis of the first and reuses test phases: it is exploratory development, not independent confirmation. No statistically supported motion selectivity or connectome advantage is established.

An engineered pixel-translation baseline identifies 8/8 directions, confirming the images contain a usable direction signal. It is a diagnostic, not a capacity-matched learned baseline. See `vision-pixel-baseline.json`.

## Diagnosis

With no background drive, bright input produced 95,243 retinal spikes over 300 ms and zero lamina or descending spikes. All modeled R1–R6 neurons have inhibitory histamine output. In this rest-initialized LIF network, external retinal drive therefore cannot start downstream firing by itself.

| Background into L1/L2/L3 | Descending spikes, black image | Descending spikes, white image |
|---|---:|---:|
| 0 Hz | 0 | 0 |
| 5 Hz | 0 | 0 |
| 20 Hz | 0 | 0 |
| 80 Hz | 1,275 | 1,116 |

These are one seed and 300 ms/condition. Background activity can enable propagation, but the direction experiment still fails. Constant Poisson voltage kicks are not an adequate validated account of graded visual physiology. Do not change inhibitory signs just to force activity or interpret the 80 Hz condition as a fitted biological parameter.

The inhibitory sign has biological motivation: photoreceptor histamine can hyperpolarize downstream L1/L2 cells ([primary study](https://elifesciences.org/articles/10972)). That does not validate this model's homogeneous spiking dynamics, external-drive amplitude, anatomical image mapping, or treatment of visual ON/OFF pathways. [ON/OFF pathway experiments](https://www.nature.com/articles/nature09545) motivate testing cell-type-specific visual dynamics next.

## Real recording

The first 12 frames sampled at 2 Hz from the already clockwise-rotated `recordings/20260909T192201Z/takeoff-land.mp4` were fed through the same encoder. No second rotation was applied. The zero-background replay generated 43,040 total spikes, with zero descending spikes.

This represents approximately six seconds of source-video sampling compressed into 240 ms neural time. It is an offline plumbing test, not real-time camera inference or a labeled motion benchmark. The exploratory background variant also processed the recording.

## Evidence and reproduction

- Initial pilot: `experiments/full-vision-20260910T131456086304Z/report.json`
- Pathway sweep: `experiments/visual-pathway-20260910T131619863801Z/report.json`
- Background variant: `experiments/full-vision-20260910T131701141382Z/report.json`

Trial NPZs retain input frames, every sparse firing-count window, descending counts and batch timing. Reports retain graph/encoder/engine hashes, seeds, predictions and controls; readout weights and preprocessing are saved separately. An early launch before mapping preparation completed failed with a missing metadata file and produced no model result; it is not included in these completed runs.

```sh
.venv/bin/python -m lab.full_vision
.venv/bin/python -m unittest lab.test_full_vision -v
# Pause the interactive full-brain worker for each independent GPU run:
.venv/bin/python -m lab.vision_experiment
.venv/bin/python -m lab.vision_pathway_probe
.venv/bin/python -m lab.vision_experiment --lamina-background-hz 80
.venv/bin/python -m lab.vision_baseline
```

## Next decision

Keep the full graph. Improve and validate the visual cell model before movement training: graded visual responses, appropriate operating points and ON/OFF responses, then direction tests in T4/T5 and downstream populations. Separate failure of the encoder, early visual dynamics and downstream decoding using intermediate readouts and fresh held-out stimuli. The existing full-brain UI still runs the stimulus demonstration; these new offline tools have not yet been wired into its controls. Autonomous flight remains gated on a successful sensory/action pipeline.

Follow-up: the complete-graph [graded prototype and frozen confirmation](GRADED_VISION.md) restore intermediate responses but do not yet qualify motion decoding.
