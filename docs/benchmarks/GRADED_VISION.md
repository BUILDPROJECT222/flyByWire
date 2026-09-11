# Full-graph graded visual diagnosis — 10 September 2026

**Result:** continuous dynamics restore a visual response through the previously silent motion pathway. Frozen-readout confirmation does not yet establish reliable motion decoding or a usable descending controller.

## Localizing the spiking failure

The saved spike-recording audit (`visual-stage-audit.json`, reproduced by `python -m lab.audit_visual_stages`) found zero T4/T5 spikes in both the original no-background pilot and the 80 Hz lamina-background variant. Original zero-background lamina activity was also zero. This localizes a failure upstream of descending decoding; increasing artificial background enough to produce some descending spikes did not restore the motion pathway.

The stage audit uses the original decoder's three mean temporal bins, training-only scaling and ridge penalty. Its counts are raw spike totals, not mean-bin values. In all cases the complete graph ran; these are readouts of stages, not isolated simulations.

## Separate graded prototype

`lab/graded_engine.py` and `lab/shaders/full-graded.wgsl` run the same **166,700 nodes and 25,582,938 edge entries** using dimensionless continuous state:

```
dv_i/dt = (-v_i + b_i + 0.8 * Σ_j W_ij ReLU(v_j) + image_drive_i) / 50 ms
Euler step = 5 ms
b = 0 for R1–R6; 0.5 otherwise
W = signed measured contacts / total measured incoming contacts
```

Photoreceptors receive fixed luminance in [0,1] through the existing mapped encoder. Unmapped photoreceptors have zero external input. Every node retains recurrent input; receptors are not clamped. Unknown transmitter signs remain zero, with their edge entries retained. Bias, gain, common time constant, normalization, transfer function and camera mapping are engineering assumptions. Synaptic counts and topology are measured; weights are not pretrained.

The absolute incoming row sums are at most 1 within float32 tolerance. Gain 0.8 and this update make the idealized map contractive under the infinity norm, rather than relying on an arbitrary hard voltage cap. The tests check finite behavior and CPU/GPU agreement; this is not a statement of biological stability.

A shared dark baseline is obtained after 1 second of simulated time. Every clip restores that state, avoiding prior-clip contamination. Each image is held for 20 ms. Saved trial voltages sample each 20 ms frame boundary, not all internal 5 ms steps.

The design is motivated by [FlyVis's continuous passive dynamics and cell-type parameterization](https://turagalab.github.io/flyvis/reference/network/), but it is **not a reproduction of FlyVis**, does not use its pretrained weights, and uses a different connectome. The [published model](https://doi.org/10.1038/s41586-024-07939-3) fits parameters to a visual task; our prototype is untrained. No continuous value is labeled as a biological spike. The spiking UI remains unchanged.

## Response and timing

A paired dark/bright/dark stimulus versus an all-dark control produced changes in retina, lamina, T4/T5 and descending state. Mean lamina change during illumination was −0.089 dimensionless units; peak absolute T4/T5 change was 0.0351. Descending changes were much weaker, with peak 0.00358. A change in voltage alone does not establish correct ON/OFF or direction tuning.

All saved states were finite; maximum absolute voltage across the pilot was 1.528. Median of per-trial p95 frame compute times was 20.29 ms for 20 ms neural time. This is an offline GPU measurement with the interactive simulation paused, not live camera-to-action latency or a sustained flight qualification.

Three tests pass: signed CPU/GPU trajectory parity and deterministic reset; finite bounded evolution and invalid input rejection; exact warm-state restoration.

## Pilot and frozen confirmation

Pilot used 16 training clips (eight phase seeds 200–207) and eight test clips (four seeds 300–303). Independent ridge readouts at each stage use three mean temporal voltage bins, training-only scaling with floor 1e-4 and fixed penalty 1.

Pilot T4/T5 and lamina decoding each scored 6/8; descending scored 4/8. This small result was **not** accepted as success. Readouts were frozen and evaluated on 20 new starting phases (400–419), two directions each, with identical-condition shuffled-frame controls. No model or readout refitting occurred.

| Frozen readout | Intact accuracy / 40 clips | Shuffled accuracy | 95% phase-bootstrap interval for intact accuracy |
|---|---:|---:|---:|
| Retina | 42.5% | 47.5% | 22.5–62.5% |
| Lamina | 65.0% | 42.5% | 45.0–85.0% |
| T4/T5 | 57.5% | 37.5% | 37.5–77.5% |
| Descending | 35.0% | 37.5% | 15.0–55.0% |

Bootstrap uses 10,000 resamples of the **20 phases**, keeping each opposite-direction pair together. These are exploratory percentile intervals, not multiplicity-corrected hypothesis tests. All intact-accuracy intervals include chance (50%). The positive lamina/T4-T5 differences from shuffled inputs warrant further temporal-response testing but do not establish reliable decoding.

## Evidence

- Pilot: `experiments/graded-vision-20260910T132410836064Z/report.json`
- Frozen confirmation: `experiments/graded-confirmation-20260910T132542581524Z/report.json`
- Stage audit: `benchmarks/visual-stage-audit.json`

Pilot NPZs contain the exact input frames, continuous states and timing; readouts include preprocessing and neuron indices. Confirmation records predictions and scores by phase, direction and condition, with model/checkpoint hashes. It reuses the frozen pilot checkpoint path explicitly.

## Next experiment

Keep the complete graph. Introduce and fit a small set of cell-type-specific time constants and operating points using training stimuli, then test fresh flash, ON/OFF edge and direction stimuli before fitting navigation outputs. Evaluate graded or hybrid dynamics without silently replacing the existing spike model. A pretrained FlyVis response is a useful external reference but must not be represented as MaleCNS ground truth or transferred blindly.

The sensory gate remains open. No motion packets, physical tests, UI model switch or trained aircraft policy were added during this work.

Follow-up: [parameter fitting and quadratic-readout confirmation](VISUAL_FIT.md) identify a linear-decoder limitation and demonstrate fresh simple-grating decoding, without claiming natural-scene or flight readiness.
