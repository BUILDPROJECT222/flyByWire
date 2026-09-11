"""Map uncalibrated motion readout scores to small yaw stick offsets.

Human / scripted packets still own the UDP process. Assist is a clamped bias only.
"""
from __future__ import annotations

def score_to_yaw(score: float | None, *, deadzone: float = 0.08, gain: float = 5.0, limit: int = 6) -> int:
    """Return integer yaw axis offset in [-limit, limit] (protocol units around 128)."""
    if score is None:
        return 0
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 0
    if not (s == s):  # NaN
        return 0
    if abs(s) < deadzone:
        return 0
    raw = s * gain
    if raw > limit:
        return limit
    if raw < -limit:
        return -limit
    return int(round(raw))
