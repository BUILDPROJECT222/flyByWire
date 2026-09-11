# Supervised controller validation — 2026-09-10

Twelve tests passed in `lab.test_flight_runner` on this Mac. Ten core/recording tests passed together (25.181 s); two additional emergency/ownership tests passed (0.809 s). All UDP tests targeted fake drones on loopback. No real hardware session was launched.

- Known TC packets, takeoff timing, repeated landing, bounded single-axis pulses, and E-stop priority passed.
- Recording preflight, complete two-second sequence plus nine-second landing, video-loss landing and browser-lease-loss landing passed with real FFmpeg encoding.
- Each test MP4 decoded to 320 × 240 and its frame count matched the frame timestamp log. Normal control transmission interval p95 was below 100 ms.
- Replay could not launch; missing owner/confirmation could not start. Emergency-only operation required no camera or prepared session and emitted no takeoff packets. A competing controller lock prevented all UDP transmissions.
- TypeScript and production build passed. HTTP checks returned 200 with E-STOP markup on Workspace, Diagnostics and Research tools. The API returned expected 409/422/403 responses for unprepared starts, invalid profiles/actions and an external Origin.
- The running controller remained idle. The camera was replaying existing footage and the graded brain was observing it. Browser interaction/visual QA and real hover/E-stop tests were not performed.

The tests establish software sequencing, not physical delivery, stability, collision clearance or the drone's link-loss behavior. Operating procedure and data formats: [FLIGHT_TESTS.md](../FLIGHT_TESTS.md).

## Revised trials and compact UI

Nineteen tests passed together in 42.715 s after adding v2 trials. New checks cover stationary capture without motor packets, no-confirmation landing, the settling window, each axis pulse, a hard deadline, emergency/fault priority, baseline eligibility from physical outcomes, and an end-to-end confirmed pulse with MP4 recording against a loopback fake drone.

TypeScript and the production build passed. All three local UI routes returned 200 with E-stop and without battery-input markup. API checks rejected an unowned stable confirmation, steering before baseline reports, and retired v1 profiles. The controller was idle; no hardware flight was launched during this update.

The desktop layout now uses available viewport height for the camera/brain panels with compact controls. Outcome details open on demand; Diagnostics retains review detail. The user requested a same-battery assumption and no battery parameters, which are now reflected in the UI and logs. Browser interaction/visual QA was not performed; layout changes used the supplied screenshot and CSS viewport constraints.
