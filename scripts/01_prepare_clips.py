"""PhysicalAI-AV chunks -> one 10 Hz pack per clip.

Each clip becomes one pickle holding a 0.1 s time grid and, on that grid,
  - JPEG frames resized to 320x576 (H x W) for the 3 cameras used (cross_left 0, front_wide 1,
    cross_right 2); 320x576 = 184,320 px lies in the model processor's pixel range and keeps 16:9
  - the actual frame timestamps picked for each grid point
  - ego pose (xyz, quat xyzw) in the dataset's world frame
Samples are cut at evaluation time: a prediction at grid index k uses frames k-3..k,
ego history k-15..k and ego future k+1..k+64.

  python scripts/01_prepare_clips.py --chunks 186 187 188 --split val
  python scripts/01_prepare_clips.py --split val --add-extra-cams   # only for Fig. 1 (16/24 images)
"""

import argparse
import glob
import io
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from svs import config  # noqa: E402

CAMS = {  # canonical alpamayo2 camera id -> PhysicalAI-AV feature name
    0: "camera_cross_left_120fov",
    1: "camera_front_wide_120fov",
    2: "camera_cross_right_120fov",
}
DT_US = 100_000
H, W = 320, 576
NUM_HIST, NUM_FUT, NUM_FRAMES = 16, 64, 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=os.path.join(config.WORK, "data", "pai"))
    p.add_argument("--out", default=os.path.join(config.WORK, "data", "prep"))
    p.add_argument("--chunks", type=int, nargs="+", default=None, help="required unless --add-extra-cams")
    p.add_argument("--split", required=True, help="expected split of these chunks (train/val)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--max-clips", type=int, default=0, help="0 = all (smoke tests use a few)")
    p.add_argument("--add-extra-cams", action="store_true",
                   help="add rear-left/rear-right/front-tele to existing packs (same grid)")
    return p.parse_args()


_AVDI = None


def avdi(src):
    global _AVDI
    if _AVDI is None:
        import physical_ai_av
        os.environ.setdefault("HF_HUB_OFFLINE", "0")
        _AVDI = physical_ai_av.PhysicalAIAVDatasetInterface(
            local_dir=src, confirm_download_threshold_gb=float("inf"))
        # download_pai.py writes files under --output-dir, but open_file() only looks in the
        # HF cache dir; read the chunk files straight from src instead.
        import contextlib

        @contextlib.contextmanager
        def open_local(filename, mode="rb", maybe_stream=False):
            path = os.path.join(src, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} missing (download the chunk first)")
            with open(path, mode) as f:
                yield f

        _AVDI.open_file = open_local
    return _AVDI


def cam_feature(av, name):
    return getattr(av.features.CAMERA, name.upper())


def pack_clip(clip_id, src, out_path, quality):
    av = avdi(src)
    ego = av.get_clip_feature(clip_id, av.features.LABELS.EGOMOTION)
    ego_ts = np.asarray(ego.timestamps if hasattr(ego, "timestamps") else ego.x, dtype=np.int64)

    readers = {cid: av.get_clip_feature(clip_id, cam_feature(av, name)) for cid, name in CAMS.items()}
    lo = max([int(ego_ts.min())] + [int(r.timestamps.min()) for r in readers.values()])
    hi = min([int(ego_ts.max())] + [int(r.timestamps.max()) for r in readers.values()])
    grid = np.arange(lo + DT_US, hi - DT_US, DT_US, dtype=np.int64)
    if len(grid) < NUM_HIST + NUM_FUT + 1:
        return clip_id, "too_short", 0

    st = ego(grid)
    xyz = np.asarray(st.pose.translation, dtype=np.float64)
    quat = np.asarray(st.pose.rotation.as_quat(), dtype=np.float64)

    jpegs, frame_ts = {}, {}
    for cid, r in readers.items():
        frames, ts = r.decode_images_from_timestamps(grid)
        buf = []
        for f in frames:
            im = Image.fromarray(f).resize((W, H), Image.BICUBIC)
            b = io.BytesIO()
            im.save(b, format="JPEG", quality=quality)
            buf.append(b.getvalue())
        jpegs[cid] = buf
        frame_ts[cid] = np.asarray(ts, dtype=np.int64)
        r.close()

    pack = {"clip_id": clip_id, "grid_us": grid, "ego_xyz": xyz, "ego_quat": quat,
            "cams": list(CAMS), "jpegs": jpegs, "frame_ts": frame_ts, "hw": (H, W)}
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(pack, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out_path)
    return clip_id, "ok", len(grid)


EXTRA_CAMS = {  # the rest of the 6-camera ring the model was trained on (used only for Fig. 1)
    3: "camera_rear_left_70fov",
    5: "camera_rear_right_70fov",
    6: "camera_front_tele_30fov",
}


def add_cams(path, src, quality):
    """Decode the extra cameras on the pack's existing grid and write them into the same pack."""
    with open(path, "rb") as f:
        pack = pickle.load(f)
    todo = [c for c in EXTRA_CAMS if c not in pack["jpegs"]]
    if not todo:
        return pack["clip_id"], "already", 0
    av = avdi(src)
    grid = pack["grid_us"]
    for cid in todo:
        r = av.get_clip_feature(pack["clip_id"], cam_feature(av, EXTRA_CAMS[cid]))
        if r.timestamps.min() > grid[0] or r.timestamps.max() < grid[-1]:
            r.close()
            return pack["clip_id"], f"cam{cid}_short", 0
        frames, ts = r.decode_images_from_timestamps(grid)
        buf = []
        for fr in frames:
            b = io.BytesIO()
            Image.fromarray(fr).resize((W, H), Image.BICUBIC).save(b, format="JPEG", quality=quality)
            buf.append(b.getvalue())
        pack["jpegs"][cid] = buf
        pack["frame_ts"][cid] = np.asarray(ts, dtype=np.int64)
        r.close()
    pack["cams"] = sorted(pack["jpegs"])
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(pack, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return pack["clip_id"], "ok", len(grid)


def main_add(a):
    out_dir = os.path.join(a.out, a.split)
    paths = sorted(glob.glob(os.path.join(out_dir, "*.pkl")))
    if a.max_clips:
        paths = paths[: a.max_clips]
    print(f"adding cameras {list(EXTRA_CAMS)} to {len(paths)} packs in {out_dir}", flush=True)
    t0, done, stats = time.time(), 0, {}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(add_cams, p, a.src, a.jpeg_quality) for p in paths]
        for fu in as_completed(futs):
            try:
                cid, status, _ = fu.result()
            except Exception as e:
                cid, status = "?", f"error:{type(e).__name__}:{e}"[:120]
            stats[status.split(":")[0]] = stats.get(status.split(":")[0], 0) + 1
            done += 1
            if status not in ("ok", "already"):
                print(f"  {cid}: {status}", flush=True)
            if done % 20 == 0 or done == len(paths):
                print(f"  {done}/{len(paths)}  {done/(time.time()-t0):.2f} clip/s  {stats}", flush=True)


def main():
    a = parse_args()
    if a.add_extra_cams:
        return main_add(a)
    if not a.chunks:
        sys.exit("--chunks is required")
    import pandas as pd
    idx = pd.read_parquet(os.path.join(a.src, "clip_index.parquet"))
    sel = idx[idx.chunk.isin(a.chunks)]
    bad = sel[sel.split != a.split]
    if len(bad):
        sys.exit(f"{len(bad)} clips in these chunks are not split={a.split}; refusing to mix splits")
    out_dir = os.path.join(a.out, a.split)
    os.makedirs(out_dir, exist_ok=True)
    todo = [c for c in sel.index if not os.path.exists(os.path.join(out_dir, f"{c}.pkl"))]
    if a.max_clips:
        todo = todo[: a.max_clips]
    print(f"chunks {a.chunks}: {len(sel)} clips, {len(todo)} to pack -> {out_dir}", flush=True)

    t0, done, stats = time.time(), 0, {}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(pack_clip, c, a.src, os.path.join(out_dir, f"{c}.pkl"), a.jpeg_quality)
                for c in todo]
        for fu in as_completed(futs):
            try:
                cid, status, n = fu.result()
            except Exception as e:  # keep going; one broken clip must not stop the chunk
                cid, status, n = "?", f"error:{type(e).__name__}:{e}"[:120], 0
            stats[status.split(":")[0]] = stats.get(status.split(":")[0], 0) + 1
            done += 1
            if status != "ok":
                print(f"  {cid}: {status}", flush=True)
            if done % 20 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"  {done}/{len(todo)}  {done/el:.2f} clip/s  {stats}", flush=True)


if __name__ == "__main__":
    main()
