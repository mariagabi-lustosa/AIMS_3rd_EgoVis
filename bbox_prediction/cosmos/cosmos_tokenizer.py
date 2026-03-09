# Installing Cosmos Tokenizer from GitHub
# 
# git clone https://github.com/NVIDIA/Cosmos-Tokenizer.git
# 
# apt-get install -y ffmpeg git-lfs
# 
# cd Cosmos-Tokenizer
# git lfs pull
# pip install -e .
# 

import os, json
from pathlib import Path
import cv2 as cv
import numpy as np
import torch

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

from huggingface_hub import login, snapshot_download

login()
model_names = [
        "Cosmos-Tokenizer-CI8x8",
        "Cosmos-Tokenizer-CI16x16",
        "Cosmos-Tokenizer-CV4x8x8",
        "Cosmos-Tokenizer-CV8x8x8",
        "Cosmos-Tokenizer-CV8x16x16",
        "Cosmos-Tokenizer-DI8x8",
        "Cosmos-Tokenizer-DI16x16",
        "Cosmos-Tokenizer-DV4x8x8",
        "Cosmos-Tokenizer-DV8x8x8",
        "Cosmos-Tokenizer-DV8x16x16",
]
for model_name in model_names:
    hf_repo = "nvidia/" + model_name
    local_dir = "pretrained_ckpts/" + model_name
    os.makedirs(local_dir, exist_ok=True)
    print(f"downloading {model_name}...")
    snapshot_download(repo_id=hf_repo, local_dir=local_dir)
    

MODEL_DIR = Path("/work/your.user/sta/Cosmos-Tokenizer/pretrained_ckpts")
ENC = MODEL_DIR / "encoder.jit"
DEC = MODEL_DIR / "decoder.jit"

from cosmos_tokenizer.video_lib import CausalVideoTokenizer

tokenizer = CausalVideoTokenizer(
    checkpoint_enc=str(ENC),
    checkpoint_dec=str(DEC),
    device=device,
    dtype="bfloat16",
)

def round_up(x, m=16):
    return int(np.ceil(x / m) * m)

def sample_frames(video_filepath, center_frame, num_frames, step):
    cap = cv.VideoCapture(video_filepath)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_filepath}")

    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_filepath}")

    start = max(0, center_frame - step * (num_frames - 1))

    frames = []
    last_good = None
    for i in range(num_frames):
        idx = start + i * step
        if idx >= total:
            break
        cap.set(cv.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
            last_good = frame
        else:
            if last_good is not None:
                frames.append(last_good)
            else:
                cap.release()
                raise ValueError(f"Failed to read frames from: {video_filepath}")

    cap.release()

    while len(frames) < num_frames:
        frames.insert(0, frames[0])

    return frames[:num_frames]

def build_batched_input_video(frames_bgr):
    h0, w0 = frames_bgr[0].shape[:2]
    H = round_up(h0, 16)
    W = round_up(w0, 16)

    resized = [cv.resize(f, (W, H), interpolation=cv.INTER_AREA) for f in frames_bgr]
    video_np = np.expand_dims(np.stack(resized, axis=0), axis=0)
    video_t = torch.from_numpy(video_np).permute(0, 4, 1, 2, 3).to(device)

    return video_t, (H, W)

CACHE_DIR = Path("/your/path/cosmos_cache_sta_train")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def cosmos_encode_pooled(video_t):
    codes, latents = tokenizer.encode(video_t)
    cosmos_feats = latents.mean(dim=(3,4))
    cosmos_feats = cosmos_feats[0].float().cpu()
    return cosmos_feats

STA_JSON = Path("/hadatasets/ego4d_data/v1/annotations/fho_sta_train.json")
FULL_SCALE_DIR = Path("/hadatasets/ego4d_data/v1/video_540ss")

data = json.load(open(STA_JSON, "r"))
annotations = data["annotations"]

available = {p.stem for p in FULL_SCALE_DIR.glob("*.mp4")}

ann_filt = [a for a in annotations if (a.get("video_uid") in available or a.get("video_id") in available)]
print("filtered annotations:", len(ann_filt))

for a in ann_filt[100:]:
    uid = a["uid"]
    out = CACHE_DIR / f"{uid}.pt"
    if out.exists():
        continue

    video_uid = a.get("video_uid") or a.get("video_id")
    video_path = FULL_SCALE_DIR / f"{video_uid}.mp4"
    center = int(a["frame"])

    frames = sample_frames(str(video_path), center_frame=center, num_frames=16, step=2)
    video_t, _ = build_batched_input_video(frames)
    feats = cosmos_encode_pooled(video_t)

    torch.save(feats, out)
    print("saved", uid, feats.shape)