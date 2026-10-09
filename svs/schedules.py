"""Camera schedules: one camera letter per frame slot, oldest first (t-0.3, t-0.2, t-0.1, t).

  F front_wide (id 1)   L cross_left (id 0)   R cross_right (id 2)
  "FRLL" = front at t-0.3, right at t-0.2, left at t-0.1 and t (4 images)
  "full" = all three cameras at all four slots (12 images)

Only for the inference-time figure (Fig. 1), larger inputs use extra cameras:
  T front_tele (id 6), X rear_left (id 3), Y rear_right (id 5)
  "full4" = F, T, L, R x 4 slots (16 images), "full6" = all six x 4 slots (24 images)
"""

import hashlib
import itertools

import numpy as np

NUM_FRAMES = 4
CAM_OF = {"L": 0, "F": 1, "R": 2, "X": 3, "Y": 5, "T": 6}
FULL3, FULL4, FULL6 = (0, 1, 2), (0, 1, 2, 6), (0, 1, 2, 3, 5, 6)
ALL_81 = ["".join(p) for p in itertools.product("FLR", repeat=NUM_FRAMES)]


def views(name: str) -> dict[int, list[int]]:
    """Schedule name -> {camera_id: [frame slots]} (cameras in ascending id order)."""
    if name in ("full", "full4", "full6"):
        cams = {"full": FULL3, "full4": FULL4, "full6": FULL6}[name]
        return {c: list(range(NUM_FRAMES)) for c in cams}
    v: dict[int, list[int]] = {}
    for slot, ch in enumerate(name):
        v.setdefault(CAM_OF[ch], []).append(slot)
    return {c: v[c] for c in sorted(v)}


def random_schedule(clip_id: str, k: int) -> str:
    """One random camera (F/L/R) per slot, reproducible for each (clip, moment)."""
    seed = int(hashlib.md5(f"{clip_id}-{k}".encode()).hexdigest(), 16) % (2 ** 32)
    return "".join(np.random.default_rng(seed).choice(list("FLR"), NUM_FRAMES))
