import argparse
import datetime
import gc
import logging
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger("raft_extractor")

LOG = setup_logging()

# ── Path helpers ──────────────────────────────────────────────────────────────

def rgb_pt_path(rgb_dir, pid, vid, sf, ef):
    return os.path.join(rgb_dir, pid, vid, f"{pid}_{vid}_{sf}_{ef}.pt")

def flow_pt_path(flow_dir, pid, vid, sf, ef):
    return os.path.join(flow_dir, pid, vid, f"{pid}_{vid}_{sf}_{ef}.pt")

def src_video_path(video_base, pid, vid):
    return os.path.join(video_base, pid, "videos", f"{vid}.MP4")

# ── RAFT GPU Logic ────────────────────────────────────────────────────────────

class RaftManager:
    def __init__(self, device="cuda"):
        self.device = device
        # Load RAFT Large
        self.weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=self.weights).to(device).eval()
        self.transforms = self.weights.transforms()
        LOG.info("RAFT Model loaded on %s", device)

    @torch.no_grad()
    def compute_flow(self, rgb_tensor: torch.Tensor):
        """
        Input: (T, 3, H, W) uint8 tensor
        Output: u_tensor, v_tensor (T, 3, H, W) float32
        """
        # 1. Standardize resolution to avoid shape mismatch and satisfy RAFT (multiple of 8)
        # We use 384 as a safe standard for EPIC-KITCHENS
        target_h, target_w = 384, 384
        if rgb_tensor.shape[-2:] != (target_h, target_w):
            rgb_tensor = F.interpolate(rgb_tensor.float(), size=(target_h, target_w), mode='bilinear').byte()

        T, C, H, W = rgb_tensor.shape
        
        # 2. Prepare pairs (0-1, 1-2, 2-3, ...)
        img1 = rgb_tensor[:-1].to(self.device)
        img2 = rgb_tensor[1:].to(self.device)
        
        # 3. Apply RAFT normalization [-1, 1]
        img1, img2 = self.transforms(img1, img2)

        # 4. Inference (batch process all pairs in the clip)
        # raft_large returns a list of flow updates; we take the final one
        out = self.model(img1, img2)[-1]  # (T-1, 2, H, W)

        # 5. Zero-pad frame 0 to maintain T length
        full_flow = torch.zeros((T, 2, H, W), device=self.device)
        full_flow[1:] = out

        # 6. Normalize to [-1, 1] based on max displacement in this clip
        # V-JEPA expectations often benefit from this local scaling
        abs_max = torch.max(torch.abs(full_flow))
        if abs_max > 1e-6:
            full_flow /= abs_max

        # 7. Split and repeat to 3 channels (T, 3, H, W) for ViT
        u_tensor = full_flow[:, 0:1, :, :].repeat(1, 3, 1, 1).cpu()
        v_tensor = full_flow[:, 1:2, :, :].repeat(1, 3, 1, 1).cpu()

        return u_tensor, v_tensor

# ── RGB re-extraction logic (Kept from original) ─────────────────────────────

def compute_observation_indices(action_start_frame, src_fps, frames_per_clip, target_fps, anticipation_sec):
    anticipation_gap = int(round(anticipation_sec * src_fps))
    obs_end = action_start_frame - anticipation_gap - 1
    src_stride = src_fps / target_fps
    return [max(0, int(round(obs_end - (frames_per_clip - 1 - i) * src_stride))) for i in range(frames_per_clip)]

def extract_rgb_from_video(video_file, sf, ef, frames_per_clip, target_fps, anticipation_sec):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open: {video_file}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    indices = compute_observation_indices(sf, src_fps, frames_per_clip, target_fps, anticipation_sec)
    
    frames_list = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, bgr = cap.read()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ret else np.zeros((384, 512, 3), dtype=np.uint8)
        frames_list.append(rgb)
    cap.release()
    
    arr = np.stack(frames_list, axis=0) # (T, H, W, 3)
    return torch.from_numpy(arr.transpose(0, 3, 1, 2)).to(torch.uint8)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_dir", default="/hadatasets/EPIC-KITCHENS_rgb_crops")
    parser.add_argument("--flow_u_dir", default="/hadatasets/EPIC-KITCHENS_rgb_crops_flow_u")
    parser.add_argument("--flow_v_dir", default="/hadatasets/EPIC-KITCHENS_rgb_crops_flow_v")
    parser.add_argument("--video_path", default="/hadatasets/EPIC-KITCHENS_384")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--frames_per_clip", type=int, default=32)
    parser.add_argument("--anticipation_sec", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    raft = RaftManager(device=device)

    # Walk disk to find crops
    crops = []
    for pid in sorted(os.listdir(args.rgb_dir)):
        pid_path = os.path.join(args.rgb_dir, pid)
        if not os.path.isdir(pid_path): continue
        for vid in sorted(os.listdir(pid_path)):
            vid_path = os.path.join(pid_path, vid)
            for fname in os.listdir(vid_path):
                if fname.endswith(".pt"):
                    parts = fname[:-3].split("_")
                    crops.append({'pid': pid, 'vid': vid, 'sf': int(parts[-2]), 'ef': int(parts[-1])})

    LOG.info("Processing %d crops using RAFT...", len(crops))
    
    counters = {"ok": 0, "skipped": 0, "error": 0}

    for c in tqdm(crops):
        out_u = flow_pt_path(args.flow_u_dir, c['pid'], c['vid'], c['sf'], c['ef'])
        out_v = flow_pt_path(args.flow_v_dir, c['pid'], c['vid'], c['sf'], c['ef'])

        if os.path.isfile(out_u) and os.path.isfile(out_v):
            counters["skipped"] += 1
            continue

        rgb_path = rgb_pt_path(args.rgb_dir, c['pid'], c['vid'], c['sf'], c['ef'])
        
        try:
            # 1. Load or extract RGB
            if os.path.exists(rgb_path):
                rgb_tensor = torch.load(rgb_path, weights_only=True)
            else:
                v_path = src_video_path(args.video_path, c['pid'], c['vid'])
                rgb_tensor = extract_rgb_from_video(v_path, c['sf'], c['ef'], args.frames_per_clip, args.fps, args.anticipation_sec)
                os.makedirs(os.path.dirname(rgb_path), exist_ok=True)
                torch.save(rgb_tensor, rgb_path)

            # 2. Compute RAFT Flow
            u_tensor, v_tensor = raft.compute_flow(rgb_tensor)

            # 3. Save
            os.makedirs(os.path.dirname(out_u), exist_ok=True)
            os.makedirs(os.path.dirname(out_v), exist_ok=True)
            torch.save(u_tensor, out_u)
            torch.save(v_tensor, out_v)
            counters["ok"] += 1

        except Exception as e:
            LOG.error("Failed %s: %s", rgb_path, e)
            counters["error"] += 1

        # Clear cache periodically
        if counters["ok"] % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    LOG.info("Done. OK: %d, Skipped: %d, Errors: %d", counters["ok"], counters["skipped"], counters["error"])

if __name__ == "__main__":
    main()