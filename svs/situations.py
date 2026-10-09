"""Driving situation from PAST ego motion only (what the car knows at prediction time).

  slow   : speed now < 10 km/h
  left   : heading turned left by more than 5 deg over the past 1.5 s
  right  : heading turned right by more than 5 deg over the past 1.5 s
  normal : everything else

The future maneuver (from the ground-truth future) is returned only for analysis.
"""

import hashlib

import numpy as np

from svs import common as C

SLOW_KMH = 10.0
THETA_DEG = 5.0
SITUATIONS = ["normal", "left", "right", "slow"]


def past_features(pack: dict, k: int):
    """(heading change over the past 1.5 s in deg, + = left; speed now in km/h; future maneuver)."""
    hx, hr, fx, fr = C.ego_local(pack, k)
    h, hr = hx[0, 0].numpy(), hr[0, 0].numpy()
    past_yaw = -np.degrees(np.arctan2(hr[0, 1, 0], hr[0, 0, 0]))
    v_now = np.linalg.norm(h[-1, :2] - h[-2, :2]) / 0.1 * 3.6
    fut_yaw = np.degrees(np.arctan2(fr[-1, 1, 0], fr[-1, 0, 0]))
    step = np.linalg.norm(np.diff(np.vstack([[0, 0], fx[:, :2]]), axis=0), axis=1) / 0.1
    if step.max() < 0.5:
        fut = "stop"
    elif fut_yaw > 30:
        fut = "left_turn"
    elif fut_yaw < -30:
        fut = "right_turn"
    elif abs(fx[-1, 1]) > 2.5:
        fut = "lane_change_curve"
    else:
        fut = "straight"
    return past_yaw, v_now, fut


def situation(past_yaw: float, v_now: float, theta: float = THETA_DEG) -> str:
    if v_now < SLOW_KMH:
        return "slow"
    if past_yaw > theta:
        return "left"
    if past_yaw < -theta:
        return "right"
    return "normal"


def split_of(clip_id: str) -> str:
    """Selection ("dev") / held-out ("test") split by a hash of the clip id."""
    return "dev" if int(hashlib.md5(clip_id.encode()).hexdigest(), 16) % 2 == 0 else "test"
