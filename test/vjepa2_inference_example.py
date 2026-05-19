"""
V-JEPA 2.1 Inference Example
==============================
Loads a pre-sampled RGB .pt file (T, C, H, W) uint8,
runs it through the vjepa2 processor + encoder,
and prints output statistics.

Usage:
    python vjepa2_inference_example.py --pt_path /path/to/clip.pt
"""

import os
import sys
import argparse
import torch
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

CACHE_DIR    = "/home/lucas.ueda/github/AIMS_3rd_EgoVis/cache_models"
LOCAL_REPO   = f"{CACHE_DIR}/facebookresearch_vjepa2_main"
CKPT_DIR     = f"{CACHE_DIR}/checkpoints"

os.environ["TORCH_HOME"] = CACHE_DIR
torch.hub.set_dir(CACHE_DIR)

# ── Load model ────────────────────────────────────────────────────────────────

def load_model():
    print("=" * 60)
    print("[1/3] Loading processor and model from local cache …")
    print(f"      repo : {LOCAL_REPO}")
    print(f"      ckpt : {CKPT_DIR}")

    processor = torch.hub.load(
        LOCAL_REPO, 'vjepa2_preprocessor', source='local'
    )
    print("      ✓ processor loaded")

    ## It returns an encoder and the predictor (used in self trainign)
    (model, predictor) = torch.hub.load(
        LOCAL_REPO, 'vjepa2_1_vit_base_384', source='local'
    )

    # print(model)
    model.eval()
    print("      ✓ model loaded")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"      ✓ parameters : {n_params:.1f} M")
    print()
    return processor, model


# ── Load .pt clip ─────────────────────────────────────────────────────────────

def load_pt_clip(pt_path: str) -> torch.Tensor:
    print("[2/3] Loading RGB clip …")
    print(f"      path : {pt_path}")

    tensor = torch.load(pt_path, map_location="cpu")   # (T, C, H, W) uint8

    # print(tensor.shape)

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)   # single-frame edge case

    T, C, H, W = tensor.shape
    print(f"      shape  : {tuple(tensor.shape)}  [T={T}, C={C}, H={H}, W={W}]")
    print(f"      dtype  : {tensor.dtype}")
    print(f"      range  : [{tensor.min().item()}, {tensor.max().item()}]")
    print(f"      size   : {os.path.getsize(pt_path) / 1024:.1f} KB")
    print()
    return tensor   # (T, C, H, W) uint8


# ── Pre-process ───────────────────────────────────────────────────────────────

def preprocess(processor, tensor: torch.Tensor) -> torch.Tensor:
    """
    The vjepa2 processor expects (T, C, H, W) uint8 → returns (1, C, T, H', W').
    """
    print("[3/3] Pre-processing …")

    # processor may accept a single clip tensor directly
    clip = processor(tensor)            # returns dict or tensor depending on version

    print(len(clip))
    print(clip[0].shape)
    clip = clip[0]

    # normalise varying return types
    if isinstance(clip, dict):
        # some versions return {"video": tensor}
        clip = clip.get("video", next(iter(clip.values())))

    if clip.ndim == 4:                  # (C, T, H, W) → add batch dim
        clip = clip.unsqueeze(0)

    print(f"      input  shape : {tuple(tensor.shape)}")
    print(f"      output shape : {tuple(clip.shape)}  [B, C, T, H, W]")
    print(f"      dtype        : {clip.dtype}")
    print(f"      value range  : [{clip.min().item():.3f}, {clip.max().item():.3f}]")
    print()
    return clip   # (1, C, T, H, W) float32


# ── Forward pass ──────────────────────────────────────────────────────────────

def run_forward(model, clip: torch.Tensor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[4/3] Running forward pass on {device.upper()} …")

    model  = model.to(device)
    clip   = clip.to(device)

    with torch.no_grad():
        output = model(clip)

    # ── interpret output ──────────────────────────────────────────────────────
    # vjepa2 encoders typically return (B, num_patches, embed_dim)
    if isinstance(output, (tuple, list)):
        features = output[0]
    elif hasattr(output, "last_hidden_state"):
        features = output.last_hidden_state
    else:
        features = output

    B, N, D = features.shape
    feat_np  = features.cpu().float().numpy()

    print()
    print("=" * 60)
    print("  OUTPUT STATISTICS")
    print("=" * 60)
    print(f"  Tensor shape   : {tuple(features.shape)}")
    print(f"                   B={B} (batch)  N={N} (patches)  D={D} (embed dim)")
    print()
    print(f"  {'':12s}  {'min':>8}  {'max':>8}  {'mean':>8}  {'std':>8}")
    print(f"  {'global':12s}  {feat_np.min():>8.4f}  {feat_np.max():>8.4f}"
          f"  {feat_np.mean():>8.4f}  {feat_np.std():>8.4f}")

    # per-patch L2 norm distribution
    norms = np.linalg.norm(feat_np[0], axis=-1)   # (N,)
    print()
    print(f"  Per-patch L2 norms (over {N} patches):")
    print(f"    min  = {norms.min():.4f}")
    print(f"    max  = {norms.max():.4f}")
    print(f"    mean = {norms.mean():.4f}")
    print(f"    std  = {norms.std():.4f}")

    # first 5 patches — first 8 dims
    print()
    print("  First 5 patch vectors (first 8 dims each):")
    for i in range(min(5, N)):
        vals = "  ".join(f"{x:+.3f}" for x in feat_np[0, i, :8])
        print(f"    patch[{i:03d}] : {vals} …")

    print("=" * 60)
    print("  Done.")
    print("=" * 60)
    return features


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pt_path", required=True,
                   help="Path to a (T, C, H, W) uint8 RGB .pt file.")
    return p.parse_args()


def main():
    args      = parse_args()
    processor, model = load_model()
    tensor    = load_pt_clip(args.pt_path)
    clip      = preprocess(processor, tensor)
    run_forward(model, clip)


if __name__ == "__main__":
    main()