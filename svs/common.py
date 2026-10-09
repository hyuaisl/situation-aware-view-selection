"""Model loading, sample cutting and prompt building for per-slot camera inputs.

Camera ids follow alpamayo2_super.common.constants: 0 cross_left (L), 1 front_wide (F),
2 cross_right (R). Frame slot 0..3 = t-0.3 s .. t (0.1 s apart).
"""

import io
import pickle
import sys

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from svs import config

sys.path.insert(0, f"{config.MODEL_CODE}/tools")
sys.path.insert(0, f"{config.MODEL_CODE}/alpamayo2/src")

NUM_HIST, NUM_FUT, NUM_FRAMES = 16, 64, 4  # ego history 1.6 s, future 6.4 s, 4 image slots


def load_pack(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def valid_t0_indices(pack: dict) -> range:
    n = len(pack["grid_us"])
    return range(max(NUM_HIST - 1, NUM_FRAMES - 1), n - NUM_FUT)


def ego_local(pack: dict, k: int):
    """History (16) / future (64) poses in the ego frame at grid index k."""
    xyz, quat = pack["ego_xyz"], pack["ego_quat"]
    h = slice(k - NUM_HIST + 1, k + 1)
    f = slice(k + 1, k + 1 + NUM_FUT)
    r0_inv = Rotation.from_quat(quat[k]).inv()
    to_local = lambda s: (r0_inv.apply(xyz[s] - xyz[k]),
                          (r0_inv * Rotation.from_quat(quat[s])).as_matrix())
    hx, hr = to_local(h)
    fx, fr = to_local(f)
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))[None, None]
    return t(hx), t(hr), fx.astype(np.float32), fr.astype(np.float32)


def decode(jpg: bytes) -> torch.Tensor:
    return torch.from_numpy(np.array(Image.open(io.BytesIO(jpg)).convert("RGB"))).permute(2, 0, 1)


def make_sample(pack: dict, k: int, views: dict[int, list[int]]) -> dict:
    """Images for the requested {camera: [slots]} plus ego history and ground-truth future."""
    cells = []
    for cam in sorted(views):
        frames = []
        for slot in views[cam]:
            gi = k - (NUM_FRAMES - 1) + slot
            frames.append((slot, decode(pack["jpegs"][cam][gi])))
        cells.append((cam, frames))
    hx, hr, fx, fr = ego_local(pack, k)
    return {"views": cells, "ego_history_xyz": hx, "ego_history_rot": hr,
            "gt_xyz": fx, "gt_rot": fr, "clip_id": pack["clip_id"], "k": k,
            "t0_us": int(pack["grid_us"][k])}


# ----------------------------------------------------------------- prompt building
def install_slot_image_builder():
    """Replace construct_image so each image is labelled with its real frame slot.

    The library builder numbers frames with enumerate() inside each camera, which is right
    when every camera has all 4 frames, and wrong when a camera is seen only at some slots
    (a camera seen only at slot 3 would be called "frame 0").
    """
    from alpamayo2_super.chat_template import conversation as conv
    from alpamayo2_super.common import constants

    def construct_image(data, include_camera_ids, camera_ids, include_frame_nums):
        msgs = []
        for cam_id, frames in data["views"]:
            if include_camera_ids:
                name = constants.CAMERA_INDICES_TO_DISPLAY_NAMES[cam_id]
                msgs.append({"type": "text", "text": f"{name}: "})
            for slot, img in frames:
                if include_frame_nums:
                    msgs.append({"type": "text", "text": f"frame {slot} "})
                msgs.append({"type": "image", "image": img})
        return msgs

    conv._stock_construct_image = getattr(conv, "_stock_construct_image", conv.construct_image)
    conv.construct_image = construct_image
    return conv


class InputBuilder:
    """Tokenize one sample the way helper.prepare_model_inputs does, with a cached processor."""

    def __init__(self, cfg, tokenizer):
        from alpamayo2_super import helper
        self.cfg = cfg
        self.proc = helper.get_processor(tokenizer, cfg)
        self.conv = install_slot_image_builder()

    def _conversation(self, data, cams):
        cfg = self.cfg
        msgs = self.conv.build_conversation(
            data=data,
            num_tokens_per_history_traj=cfg.tokens_per_history_traj,
            num_tokens_per_future_traj=cfg.tokens_per_future_traj,
            components_order=["image", "traj_history", "prompt"],
            components_prompt=["cot", "traj_future"],
            generation_mode=True,
            include_camera_ids=cfg.include_camera_ids,
            camera_ids=torch.tensor(cams),
            include_frame_nums=cfg.frame_label == "frame_num",
        )
        if msgs[-1]["role"] == "assistant" and not msgs[-1]["content"]:
            msgs = msgs[:-1]
        return self.proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, add_vision_id=False)

    def __call__(self, sample):
        text = self._conversation(sample, [c for c, _ in sample["views"]])
        images = torch.stack([img for _, frames in sample["views"] for _, img in frames])
        images = images.float() / 255.0
        td = dict(self.proc(text=text, images=images, videos=None, padding=False,
                            return_tensors="pt", do_rescale=False))
        return {"tokenized_data": td, "ego_history_xyz": sample["ego_history_xyz"],
                "ego_history_rot": sample["ego_history_rot"]}, text

    def stock_text(self, sample):
        """Prompt from the untouched library builder (all cameras x all slots only).

        Used once as a check: for the full input it must equal the patched builder's prompt."""
        cams = [c for c, _ in sample["views"]]
        grid = torch.stack([torch.stack([img for _, img in fr]) for _, fr in sample["views"]])
        patched = self.conv.construct_image
        self.conv.construct_image = self.conv._stock_construct_image
        try:
            return self._conversation({"image_frames": grid}, cams)
        finally:
            self.conv.construct_image = patched


def build_model(dev: str):
    """Load the 2B stage-2 checkpoint (model code: SVS_MODEL_CODE/tools/eval_ade.py)."""
    import argparse
    import eval_ade as ea
    a = argparse.Namespace(
        stage2=config.CKPT, ckpt=config.CKPT,
        expert_hidden=448, expert_inter=1792, expert_heads=8, backbone=config.BACKBONE,
        data=config.COC_DATA, teacher=f"{config.COC_DATA}/_teacher_assets",
        cameras="0,1,2", frames="0,1,2,3", gpu=0, coc=True, kd_weight=0.0, grad_ckpt=False,
        min_pixels=None, max_pixels=None, index=None, max_shards=None, sample_policy="all",
        max_min_ade=None, val_clip_frac=0.02, traj_future=True, samples=6, temperature=0.6,
        max_new_tokens=200, n=1)
    model, cfg = ea.build(a, dev)
    return model, cfg, a


def predict(model, cfg, inp, args, dev):
    """Reasoning text + 6 trajectory candidates for one tokenized input already on `dev`."""
    import eval_ade as ea
    traj = {"hxyz": inp["ego_history_xyz"], "hrot": inp["ego_history_rot"]}
    return ea.expert_traj(model, cfg, inp, traj, args, dev)


def min_ade(preds: list, gt: np.ndarray) -> tuple[float, float]:
    """minADE over the candidates on xy, and FDE of the same best candidate."""
    ades = [float(np.linalg.norm(p[:, :2] - gt[:, :2], axis=-1).mean()) for p in preds]
    b = int(np.argmin(ades))
    return ades[b], float(np.linalg.norm(preds[b][-1, :2] - gt[-1, :2]))


def to_device(x, dev):
    if isinstance(x, torch.Tensor):
        return x.to(dev)
    if isinstance(x, dict):
        return {k: to_device(v, dev) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, dev) for v in x)
    return x
