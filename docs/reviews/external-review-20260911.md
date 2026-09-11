# External tracking and hover review

Reviewed September 10, 2026 (EDT): four trials started 20:41:53–20:44:47. Directory timestamps are UTC September 11. Reviewed external and onboard frame sequences alongside tracking samples, command logs, and recording timestamps. No hardware commands or application behavior changed during this review.

## Findings

| Trial start (EDT) | Test | Tracking | Visible flight outcome |
|---|---|---|---|
| 20:41:53 | Stationary | 219/219 samples tracking; essentially zero displacement | Drone stays on rug. Useful stationary control, not evidence of airborne tracking. |
| 20:42:28 | Baseline 1 | Lost 1.498 s after takeoff command | Climbs, travels toward upper-right of external view and leaves it. Onboard footage returns to a surface around +6 s and later changes position. Exact touchdown and any intervention are not resolved by the external view. |
| 20:43:59 | Baseline 2 | Lost 1.456 s after takeoff command | Climbs and approaches the camera, becoming much larger by +4–5 s; person retrieves/handles it by about +8 s. Not a clean unassisted baseline. |
| 20:44:47 | Baseline 3 | Lost 1.459 s after takeoff command | Climbs and drifts left in the image, descends around +5 s. Onboard view is back on rug around +6 s and subsequently steady. Best visible landing of this batch, but position is not held. |

Times refer to the commanded takeoff, not measured liftoff. External video alignment uses recording-start clock mapping and is approximate; onboard captions use frame receipt timestamps, not sensor exposure times. Directions are image directions, not calibrated drone axes. No metric height, velocity, or distance can be inferred reliably from this uncalibrated camera.

## Why tracking fails

The fixed-template tracker works while the drone remains on the rug, then fails at the ground-to-air transition in all three flights. The selected patch contains background texture; the airborne silhouette, rotor blur, scale, and background change. This is consistent with the failure timing and visible footage, although the original selection template was not saved for exact reproduction.

The first rejected correlation scores were 0.672, 0.614, and 0.736. The first and third exceed the 0.65 threshold: the code's competing-match margin rejection explains those losses, rather than merely a low score. The middle score is below threshold. Sample gaps at loss were only 117, 117, and 100 ms; these are not the 500 ms sampling-gap failure. Once lost, the current implementation intentionally stays lost until manually reselected. It therefore never tracks any sustained airborne portion in these trials.

Later trials used accepted boxes only about 15×11 and 13×10 pixels at the 320×240 tracking resolution, leaving little target detail. Baseline 2 also starts with dx=-44, dy=20 relative to an earlier selection, and remains there until loss. That is not measured in-trial drift: the reference predates the recording. The saved data cannot establish whether that earlier offset was actual movement or a false match. A per-trial origin reset and stored selection frame would remove this ambiguity.

Correlation is a similarity score, not a calibrated probability that the drone was identified correctly. Lowering the threshold alone would risk following the rug or another object.

## Flight-control and capture checks

All three baseline logs contain only roll/pitch/throttle/yaw=(128,128,128,128). Flags are neutral, takeoff, and landing only; no steering pulse or emergency-stop packet was logged. The stationary trial sent no nine-byte motor-control packets. Brain control was disabled. The external tracker provides observation only and neither corrects drift nor triggers landing when it loses the target.

Landing began 4.017, 4.049, and 4.037 seconds after takeoff was requested. These are four-second takeoff-to-landing-command tests, not four seconds of settled hover. They do not establish a stable hover plateau before descent. Recorded command intervals were roughly 53–54 ms median, with maxima 58–63 ms; there is no large sender stall in this batch. UDP send logs do not prove every packet reached the aircraft or tell us its internal controller state.

All four external MP4 conversions and onboard recordings completed. External recordings contain 455, 412, 493, and 499 frames at about 20 fps average. They are variable-frame-rate; the last MP4's reported nominal 2000 fps is a timestamp/container artifact, not actual camera speed. Observed maximum external frame spacing was about 81–100 ms. Onboard recordings are 10 fps. This evidence points to matching failure rather than a broken recording pipeline.

Physical drift is evident despite neutral steering. The app presently has no position-feedback correction. These recordings do not isolate whether the underlying drift comes from aircraft trim, onboard stabilization, airflow, launch conditions, or another physical cause. They do not support attributing it to battery degradation; the same-battery performance assumption is retained.

## Recommended next work

1. Use these clips as an offline tracking benchmark before collecting more airborne tests. Manually annotate target boxes through liftoff, motion, landing, and loss of view. Measure localization error, visible-target coverage, false locks, and recovery time separately. Include stationary and person-entering-view negatives.
2. Replace the single fixed ground patch with foreground-motion candidates from the fixed camera, appearance verification, and motion prediction. Allow verified reacquisition after loss; do not blindly follow the nearest moving object. Store rejection reasons, candidate margin, and selection snapshots. Test at higher processing resolution so the drone is not reduced to a ten-pixel target.
3. Reset displacement origin for each trial. Display “lost / position unknown” immediately and distinguish a saved landing command from a visually or manually confirmed landing. Do not credit the middle flight as an unassisted baseline.
4. Validate with motors-off hand movement through the expected flight volume first. Keep the whole volume in the external camera view; use consistent lighting and a target with clear contrast. Verify reacquisition and handling of a person entering the image without risking another flight.
5. Once airborne tracking is demonstrated in replay and manual movement, calibrate image directions to drone axes and establish small bounded correction responses. Actual position-holding requires a separate feedback controller; the current neutral baseline cannot correct drift. A single uncalibrated webcam only supplies image-plane location, with depth/scale ambiguity.

## Evidence

- `summary.json`: measured tracking and timing statistics.
- `external-1.jpg` through `external-4.jpg`: webcam contact sheets with recorded tracking overlays.
- `onboard-2.jpg` through `onboard-4.jpg`: paired onboard contact sheets.
- `review.py`: external contact-sheet and summary extraction using actual video presentation timestamps.
- Original footage and logs remain in the four `recordings/trial-20260911T004...` directories.
