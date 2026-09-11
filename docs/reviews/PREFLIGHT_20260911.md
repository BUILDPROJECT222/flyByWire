# Preflight status — 2026-09-11

## Media processed
- Onboard contact sheets for all 16 `trial-*` flights: `experiments/preflight-stills/`
- External webcam: 4 trials; status paths fixed `droneControl/` → `flyByWire/`
- Offline retrack + reacquisition: `experiments/retrack-20260911T020008/`

## Retrack results (vs live lost)

- `trial-20260911T004153980568Z`: offline track **100%** (678/678 frames), recoveries=0, live_lost_samples=0
- `trial-20260911T004228743476Z`: offline track **42%** (259/614 frames), recoveries=0, live_lost_samples=112
- `trial-20260911T004359297362Z`: offline track **94%** (696/737 frames), recoveries=5, live_lost_samples=111
- `trial-20260911T004447260405Z`: offline track **52%** (258/500 frames), recoveries=1, live_lost_samples=113

## Tracking notes
- Live tracker stays lost once lost (no reacquisition)
- Liftoff still hard; offline recovery helps post-hoc analysis
- Stationary external held 100% offline and live

## Still before more flight tests
- [ ] Physical E-stop unvalidated — bench kill before airborne work
- [ ] Floor-only / indoor envelope only
- [ ] Reselect external box after liftoff if lock drops
- [ ] Confirm Wi‑Fi RC link + battery before arm
- [ ] Brain observation-only; motors on supervised flight process

## Ready
- Layout, branding, training docs, stills, and retrack artifacts are in place
- Safe for more supervised RC trials after E-stop check

## Brain assist (added)
- Offline demo video: `recordings/demo-default.mp4` (004359 onboard)
- Active readout: `experiments/active-quadratic-T4_T5.npz`
- New trial: **Brain yaw assist · 8 s** (`brain-yaw-assist-v2`) — needs 2 clean baselines; clamped yaw only
- Still validate physical E-stop before using assist airborne
