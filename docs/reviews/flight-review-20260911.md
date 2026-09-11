# Review of the September 10 supervised drone clips

Reviewed all 12 MP4s by fully decoding them, checking their frame/event/observer logs, and inspecting command-aligned contact sheets. Five ambiguous or representative clips also received half-second visual sampling through the first eight seconds after takeoff. No drone commands were sent during this review.

## Outcome

The capture and command pipeline worked consistently. Physical outcomes varied: ten clips show visible rise or substantial camera movement consistent with takeoff, one commanded hover has no clear lift, and one preparation expired without issuing motor commands. Completing the software sequence must not be counted as a successful physical landing.

The user confirms all runs used the same battery and one flight fell from the table after landing. Clip 2 is the likely incident based on its tabletop-to-under-table view; the user did not explicitly identify its filename. Battery state was not measured. Battery depletion, a previous impact, launch surface, and trial order therefore remain possible confounds rather than diagnosed causes.

## Clip-by-clip notes

Times below are September 10 in New York (EDT). Clip links preserve the original recordings.

| # | Local time | Profile | MP4 length | Visual / log outcome |
| --- | --- | --- | --- | --- |
| 1 | 19:56:16 | Hover 2 s | 21.5 s | Table launch; rises and returns to a tabletop view. End framing differs from the starting framing. No calibrated drift distance. |
| 2 | 19:57:05 | Hover 2 s | 17.7 s | Table launch; rises, then a sharp drop/rotation around +4 s and a stable under-table view. Likely the reported fall after landing. Exclude from clean-landing calibration. |
| 3 | 19:58:02 | Hover 4 s | 25.8 s | Clearest longer baseline: rises from floor view, shifts horizontally/changes view, then returns to carpet around +6 s. Useful takeoff/descent example; not steady position hold. |
| 4 | 19:58:57 | Yaw − | 36.7 s | Visible rise and return to carpet; yaw pulse sent during the takeoff transient. Cannot isolate yaw response. |
| 5 | 19:59:41 | Yaw + | 24.3 s | Visible rise and return near the starting floor view; pulse also overlaps takeoff. |
| 6 | 20:00:15 | Roll − | 24.3 s | Visible rise and return to a changed carpet view. Translation, tilt and yaw cannot be separated from this camera alone. |
| 7 | 20:00:59 | Roll + | 16.9 s | Rises, shifts view, and reaches floor level near a phone at about +4–5 s. Later camera movement and visible handling should be excluded from flight-response labels. |
| 8 | 20:01:28 | Pitch − | 32.0 s | Visible rise and return to carpet. No isolated, calibrated pitch response. |
| 9 | 20:02:11 | Pitch + | 32.4 s | Readiness expired after 30 s; no takeoff/control-axis packets. Apparently stationary carpet scene. Keep as a negative-motion example. |
| 10 | 20:02:59 | Pitch + | 18.2 s | Visible rise and return to carpet. Completes the commanded positive-pitch attempt, but not a clean steering calibration. |
| 11 | 20:03:31 | Hover 2 s | 19.1 s | Takeoff and landing packets sent, but no obvious lift in the received footage; floor-level framing persists throughout the sampled sequence. Mark “lift not observed,” not successful hover. Cause unknown. |
| 12 | 20:03:58 | Hover 2 s | 16.0 s | Visible rise, then a changed carpet view after descent around +4 s. Later lift was possible, so clip 11 alone does not establish a depleted battery. |

[Clips 1–4 contact sheet](contact-1.jpg) · [Clips 5–8](contact-2.jpg) · [Clips 9–12](contact-3.jpg)

[Table incident sequence](detail-2.jpg) · [Four-second baseline](detail-3.jpg) · [Roll-positive / handling](detail-7.jpg) · [No-clear-lift hover](detail-11.jpg) · [Final hover](detail-12.jpg)

## Capture and control measurements

- **12/12 MP4s decode**, with 2,849 total frames and 284.9 s of nominal playback: **4 min 44.9 s**. All are 320 × 240 and visually upright after the requested clockwise rotation.
- Every MP4 frame count matches both its `frames.jsonl` count and the encoder's final count. All encoder exit codes are zero. No logged recording-queue drops, transmission errors, or fault-triggered landings. There is one exact duplicate decoded frame in the last clip; that is not evidence of a queue drop.
- Eleven takeoff sequences were transmitted. Ten scheduled landing starts were 2.007–2.043 s after takeoff; the four-second test was 4.032 s. All eleven sent the bounded landing sequence. The expired preparation sent no control packets.
- Control-packet interval: median **53.9 ms**, p95 **55.3 ms**, maximum **63.1 ms**. Actual cadence is approximately 18.5–18.6 Hz, slightly below the intended 20 Hz; there were no large scheduling stalls in this batch.
- Each of the six axis trials transmitted three non-centered packets, first at **+1.222 to +1.234 s** and last at **+1.325 to +1.342 s**. All matched the chosen axis, sign and amplitude; throttle stayed centered. This verifies software output, not the physical axis response.
- Decoded frame-receipt interval: median **107.5 ms**, p95 **125.0 ms**, maximum **346.1 ms**. Nominal MP4 time is 100 ms per frame; use the timestamp logs rather than the playback clock for timing analysis.
- Brain computation: median **34.0 ms**, p95 **49.0 ms**, maximum **331.3 ms**. Camera receipt to inference completion: median **50.9 ms**, p95 **72.5 ms**, maximum **351.8 ms**. These measurements exclude camera exposure and upstream video buffering.
- **No E-stop packets were issued in this batch.** Emergency cutoff remains physically unvalidated. Status replies and successful UDP sends do not prove motor state or a safe landing.

## What the clips reveal

### The steering pulses are too early to calibrate cleanly

In the detailed floor-launch sequences, blur/movement begins around +1.5 s, and clear raised views appear around +2 s. The pulses are already over by about +1.4 s, and the two-second scripts then request landing. Command-to-visible-response timing combines actual motor/takeoff response with camera/RTSP buffering; these videos do not separate those delays.

The next control experiment needs a measured takeoff/settling phase before applying an axis pulse, followed by a short observation phase and landing. Do not simply interpret the current two-second sequences as two seconds of stabilized hover. The existing four-second neutral trial is the best baseline in this batch for diagnosing that timing.

### The motion decoder still fails a basic negative control

The expired preparation (clip 9) produced **324 leftward proposals and zero rightward proposals** despite its apparently stationary carpet view. Several other clips hold one direction through preflight, takeoff and settled views. The model is producing image-dependent biases rather than a trustworthy calibrated navigation signal. No motion accuracy percentage is justified because the flights lack independent motion labels.

These natural textures, blur, lighting changes, stationary segments and takeoff/descent transitions are valuable evaluation material. Clip 9 is especially useful as a held-out stationary test. Motor profile names must not be used as ground-truth image-motion labels, and adjacent frames from one clip must not be split between training and testing as if independent samples.

## Next validation batch

1. Use a charged battery and a **floor-level landing area**, with room for the observed drift. The reported table fall makes another tabletop trial inappropriate. Inspect the drone after that impact before further tests.
2. Repeat neutral baselines from a marked position and record battery/run order. Use a fixed external video of the drone to distinguish lift, drift, landing and post-landing motion, and to estimate the onboard video delay.
3. Separate takeoff/settling from the pulse window; then repeat matched positive/negative single-axis trials with enough room and supervision. Measure outcomes before increasing duration or amplitude.
4. Label these clips first: stationary, takeoff, motion, descent, landed, handling, incident and uncertain. Add a stationary/unknown option to the perception task and test against simple image-motion baselines before returning motor authority to a learned model.

Original recordings and operational code were not changed. Numerical outputs are in [summary.json](summary.json) and [review.json](review.json); the reproducible extraction script is [review.py](review.py). Pixel-change values in the machine-readable review are only image-difference diagnostics, not optical flow, physical velocity or an accuracy score.
