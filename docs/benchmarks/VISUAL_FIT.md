# Visual dynamics fit and nonlinear readout — 10 September 2026

**Outcome:** the bounded dynamics search did not materially improve linear decoding. A quadratic T4/T5 readout on the **default full graded network** recovered synthetic motion direction, and this result survived a fresh phase/speed/frequency confirmation. Descending-neuron decoding remains unsuccessful. This is sensory progress, not a flight policy or proof of biological motion tuning.

## Controlled parameter search

All 166,700 neurons and 25,582,938 retained edge entries stayed in use. Eight scalar parameters were varied: bias and time constant for L1, L2, T4-family and T5-family cells. Other cells retained the default parameters. Six seeded random candidates plus the default were compared; this is a small bounded search, not exhaustive optimization.

Readouts were fitted on six training phases (500–505), selected using four separate phases (600–603), and evaluated only afterward on 20 new phases (700–719). Both directions were used at each phase. Selection was fixed to T4/T5 accuracy, with earliest candidate winning ties. The selection checkpoint was written before test evaluation.

The winning candidate had only 25% selection accuracy, versus 12.5% for the default; neither selection result was satisfactory. On 40 test clips, linear T4/T5 decoding was 55% for the selected model versus 50% for the default. Descending decoding was 35% versus 37.5%. Therefore the tuned parameters are not promoted.

Source snapshots, parameters, preprocessing, weights, model hashes, stage-mean flash responses and all feature matrices are retained in:
`experiments/visual-fit-20260910T133833056081Z/`.

The configurable GPU engine passes four tests, including CPU parity with heterogeneous time constants and biases, default parity/reset, warm-state restoration and invalid-input rejection.

## Testing the decoder bottleneck

A fixed quadratic kernel readout was then fitted to the same training features:

```
K(x, y) = (1 + standardized(x) · standardized(y) / feature_count)^2
ridge penalty = 0.001
```

Only T4/T5 continuous activity enters this readout, summarized in three temporal bins across 12 frames. There is no direct pixel shortcut. The full recurrent graph is still computed; this is a readout of its motion-detector population, not a crop of the simulated network.

The quadratic decoder scored 40/40 on the existing intact test clips for both default and selected dynamics, while shuffled and input-off controls stayed near chance. This was a post-hoc diagnosis, so it was not treated as confirmation. The simpler default dynamics and its decoder were frozen for a new experiment.

This supports a narrower conclusion than “all prior failures were neural”: the linear decoder could not recover a signal that a nonlinear decoder can use. More complex readouts must be included before attributing failures exclusively to the neural dynamics.

## Fresh frozen confirmation

No refitting or parameter selection occurred. Twenty new phases (1000–1019), both directions per phase, were tested in four conditions:

| Condition | Correct / 40 | Accuracy |
|---|---:|---:|
| Original frequency and speed, new phases | 40 | 100% |
| Same frames, shuffled temporal order | 21 | 52.5% |
| Lower spatial frequency, slower movement, lower contrast | 40 | 100% |
| Higher spatial frequency and faster movement | 40 | 100% |

The original grating has frequency 3 cycles/image and phase increment 0.35/frame. Slow/coarse uses 2, 0.25 and contrast 60; fast/fine uses 4, 0.5 and contrast 100. All clips have 12 frames. Each frame advances 20 ms of neural time.

These are 20 independent starting phases with paired directions, not 40 independent scenes; the same phases also appear across conditions. All-perfect phase bootstrap intervals collapse to [1,1], which must not be interpreted as certainty about future accuracy. The evidence supports these simple sinusoidal motion conditions, not arbitrary camera imagery.

The pre-existing direct-image translation diagnostic also solves simple gratings. A connectome advantage has not been established, and no improvement over conventional optical flow is claimed.

Fresh evidence:
`experiments/quadratic-confirmation-20260910T134139278856Z/report.json`.
Features, per-trial predictions/scores, source/model hashes and frozen readout are retained.

## Reusable readout

`lab/motion_readout.py` loads the frozen decoder, buffers 12 frame-boundary full-network states and returns left/right plus an **uncalibrated score**. It returns no result before its history is full and can be reset between sequences. It does not issue flight commands.

Checkpoint:
`experiments/visual-fit-20260910T133833056081Z/quadratic-0-T4_T5.npz`.

Two readout tests pass, and the reusable implementation reproduces all 160 saved confirmation scores. Scores are not probabilities. Overlapping rolling windows and natural-scene inputs have not been validated; the current evidence uses independent reset clips.

## Next validation

Test moving edges, textured scenes, stationary-camera controls, looming and actual rotated drone footage. Keep both the full-network nonlinear readout and a conventional image-motion baseline. Add sequence resets, scene-change detection and frame timing; measure total capture-to-proposal latency.

The full graph's current successful readout is from T4/T5, not descending or biological motor neurons. A navigation/action policy still needs training and calibration. The spiking UI remains unchanged; the successful model is the separate graded prototype, whose continuous values must not be labeled as spikes.

Reproduce:
```sh
# Pause the interactive GPU simulation during independent full-network runs.
.venv/bin/python -m lab.fit_visual_dynamics
.venv/bin/python -m lab.audit_visual_readout
.venv/bin/python -m lab.confirm_quadratic_vision
.venv/bin/python -m unittest lab.test_graded_engine lab.test_motion_readout -v
```

The audit/confirmation scripts explicitly reference the saved source run above. New fit runs do not silently replace the frozen checkpoint.
