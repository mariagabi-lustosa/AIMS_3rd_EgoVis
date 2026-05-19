"""
V-JEPA 2.1 Fine-tuning — EPIC-KITCHENS Verb + Noun Classification
==================================================================
RGB + Optical Flow (U + V) Fusion Edition

Architecture:
  1. Three parallel V-JEPA encoders for RGB, Flow-U, Flow-V
  2. Cross-modal attention alignment: (U → RGB) and (V → RGB)
  3. Statistical Attentive Pooling (SAP) to fuse the two aligned
     representations into a single fixed-size vector
  4. DualHead MLP → verb & noun logits

Key optimisations carried over from the base script:
  • Batched GPU preprocessing (no per-sample CPU loop)
  • AMP (autocast + GradScaler) throughout
  • torch.compile on classifier (PyTorch ≥ 2.0)
  • cudnn.benchmark = True
  • Non-blocking host→device transfers
  • persistent_workers + prefetch_factor in DataLoaders
  • Fixed gradient-accumulation flush on last batch
  • weights_only=True on torch.load
  • Vectorised mean top-5 recall (scatter)

Usage:
    python train_vjepa2_epic_flow_fusion.py \\
        --train_csv /hadatasets/EPIC-KITCHENS/EPIC_100_train.csv \\
        --val_csv   /hadatasets/EPIC-KITCHENS/EPIC_100_validation.csv \\
        --rgb_dir   /hadatasets/EPIC-KITCHENS_rgb_crops \\
        --flow_u_dir /hadatasets/EPIC-KITCHENS_rgb_crops_flow_u \\
        --flow_v_dir /hadatasets/EPIC-KITCHENS_rgb_crops_flow_v \\
        --model_path ./experiments/vjepa2_flow_fusion \\
        [OPTIONS]
"""

import os
import sys
import argparse
import logging
import random
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

torch.backends.cudnn.benchmark = True

# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(model_path: str) -> logging.Logger:
    os.makedirs(model_path, exist_ok=True)
    log_file = os.path.join(
        model_path, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("vjepa2_flow_fusion")


# ── Path helpers ──────────────────────────────────────────────────────────────

def _crop_path(base_dir: str, row: pd.Series) -> str:
    pid = row["participant_id"]
    vid = row["video_id"]
    sf  = int(row["start_frame"])
    ef  = int(row["stop_frame"])
    return os.path.join(base_dir, pid, vid, f"{pid}_{vid}_{sf}_{ef}.pt")


# ── Dataset ───────────────────────────────────────────────────────────────────

class EpicKitchensFlowDataset(Dataset):
    """
    Loads pre-extracted (T, C, H, W) uint8 .pt clips for RGB, Flow-U and Flow-V.

    A sample is included only when ALL THREE modality files exist.
    Returns (rgb_clip, u_clip, v_clip, verb_label, noun_label).
    """

    TARGET_H = 384
    TARGET_W = 384

    def __init__(
        self,
        df: pd.DataFrame,
        rgb_dir: str,
        flow_u_dir: str,
        flow_v_dir: str,
        debug: bool = False,
    ):
        if debug:
            df = df.sample(n=min(200, len(df)), random_state=42).reset_index(drop=True)

        log = logging.getLogger("vjepa2_flow_fusion")

        def all_exist(r):
            return (
                os.path.isfile(_crop_path(rgb_dir,    r))
                and os.path.isfile(_crop_path(flow_u_dir, r))
                and os.path.isfile(_crop_path(flow_v_dir, r))
            )

        valid_mask = df.apply(all_exist, axis=1)
        missing = (~valid_mask).sum()
        if missing:
            log.warning("%d rows missing at least one modality and will be dropped.", missing)

        self.df         = df[valid_mask].reset_index(drop=True)
        self.rgb_dir    = rgb_dir
        self.flow_u_dir = flow_u_dir
        self.flow_v_dir = flow_v_dir

    def __len__(self):
        return len(self.df)

    def _load(self, path: str) -> torch.Tensor:
        """Load a .pt clip and normalise shape/dtype."""
        clip = torch.load(path, map_location="cpu", weights_only=True)
        if clip.ndim == 3:
            clip = clip.unsqueeze(0)  # (T, C, H, W)

        # Flow tensors saved by the RAFT extractor are already float32 in [-1,1].
        # RGB tensors are uint8. We keep them as-is and let _preprocess handle it.
        if clip.shape[-2:] != (self.TARGET_H, self.TARGET_W):
            orig_dtype = clip.dtype
            clip = F.interpolate(
                clip.float(),
                size=(self.TARGET_H, self.TARGET_W),
                mode="bilinear",
                align_corners=False,
            )
            # Only snap back to uint8 if the clip was originally uint8
            if orig_dtype == torch.uint8:
                clip = clip.to(torch.uint8)

        return clip

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rgb = self._load(_crop_path(self.rgb_dir,    row))
        u   = self._load(_crop_path(self.flow_u_dir, row))
        v   = self._load(_crop_path(self.flow_v_dir, row))
        return rgb, u, v, int(row["verb_class"]), int(row["noun_class"])


# ── GPU preprocessing ─────────────────────────────────────────────────────────

def _preprocess_batch_gpu(
    processor,
    clips: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    clips : (B, T, C, H, W)  uint8 or float32
    Returns (B, C, T, H', W') float32 on `device`.

    Flow clips arrive as float32 in [-1, 1]; the V-JEPA preprocessor
    expects uint8 in [0, 255]. We rescale flow to [0, 255] uint8 before
    preprocessing so normalisation is well-defined, then the encoder sees
    it as a regular 'image' stream.
    """
    if clips.dtype != torch.uint8:
        # Rescale [-1, 1] float32 → [0, 255] uint8
        clips = ((clips.float().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)

    clips_dev = clips.to(device, non_blocking=True)

    try:
        out = processor(clips_dev)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.ndim == 4:
            out = out.unsqueeze(0)
        return out
    except Exception:
        pass

    # Fallback: per-sample
    processed = []
    for b in range(clips_dev.shape[0]):
        out = processor(clips_dev[b])
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.ndim == 4:
            out = out.unsqueeze(0)
        processed.append(out)
    return torch.cat(processed, dim=0)


def _encode(encoder, clip_proc: torch.Tensor) -> torch.Tensor:
    """Run encoder and unwrap output → (B, N, D)."""
    out = encoder(clip_proc)
    if isinstance(out, (tuple, list)):
        return out[0]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    return out


# ── Cross-Modal Attention Alignment ──────────────────────────────────────────

class CrossModalAttentionAlignment(nn.Module):
    """
    Aligns a *query* modality (flow) to an *anchor* modality (RGB) via
    multi-head cross-attention.

        query  : (B, N, D)  — flow patch embeddings
        anchor : (B, N, D)  — RGB patch embeddings

    The anchor acts as key/value; the query attends to it, producing a
    flow representation that is semantically grounded in the RGB context.
    The original query is added as a residual so flow-specific signal is
    not lost.

    Output : (B, N, D)
    """

    def __init__(self, embed_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ff   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,   # flow   (B, N, D)
        anchor: torch.Tensor,  # RGB    (B, N, D)
    ) -> torch.Tensor:
        # Cross-attention: flow queries attend to RGB keys/values
        attended, _ = self.cross_attn(query, anchor, anchor)
        query = self.norm(query + attended)              # residual + LN
        query = self.norm2(query + self.ff(query))       # FFN + LN
        return query


# ── Statistical Attentive Pooling (SAP) ──────────────────────────────────────

class StatisticalAttentivePooling(nn.Module):
    """
    Computes a fixed-size representation from a variable-length set of
    token embeddings by learning an attention score per token, then
    combining the weighted mean AND weighted std as the pooled vector.

    Given two aligned representations X_u (B, N, D) and X_v (B, N, D):
      1. Concatenate along the token axis → (B, 2N, D)
      2. Learn token-wise attention weights  α ∈ (0,1)  summing to 1
      3. Pool: μ = Σ αᵢ xᵢ,   σ² = Σ αᵢ (xᵢ - μ)²
      4. Output = [μ ; σ]  → (B, 2D)

    The statistics carry complementary info (mean = what, std = how much),
    which has been shown to help speaker/action verification tasks.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        # Learnable attention projection: D → 1 score per token
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(
        self,
        x_u: torch.Tensor,  # (B, N, D)
        x_v: torch.Tensor,  # (B, N, D)
    ) -> torch.Tensor:
        # 1. Concatenate modalities along token dimension
        x = torch.cat([x_u, x_v], dim=1)      # (B, 2N, D)

        # 2. Attention weights
        alpha = self.attn(x)                   # (B, 2N, 1)
        alpha = torch.softmax(alpha, dim=1)    # normalise over tokens

        # 3. Weighted statistics
        mu  = (alpha * x).sum(dim=1)           # (B, D)
        var = (alpha * (x - mu.unsqueeze(1)).pow(2)).sum(dim=1)
        sigma = (var + 1e-8).sqrt()            # (B, D)

        # 4. Concatenate mean and std
        return torch.cat([mu, sigma], dim=1)   # (B, 2D)


# ── Full Fusion Classifier ────────────────────────────────────────────────────

class FlowRGBFusionClassifier(nn.Module):
    """
    Full fusion pipeline:
      RGB patches  (B, N, D)
      U   patches  (B, N, D)   ──► CrossModalAttentionAlignment(U, RGB) ──► X_u
      V   patches  (B, N, D)   ──► CrossModalAttentionAlignment(V, RGB) ──► X_v
                                               │
                              StatisticalAttentivePooling(X_u, X_v)
                                               │
                                          (B, 2D)
                                               │
                                    MLP → verb_logits, noun_logits
    """

    def __init__(
        self,
        encoder_dim: int,
        n_verbs: int,
        n_nouns: int,
        n_heads: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        attn_dropout: float = 0.1,
    ):
        super().__init__()

        self.u_align = CrossModalAttentionAlignment(encoder_dim, n_heads, attn_dropout)
        self.v_align = CrossModalAttentionAlignment(encoder_dim, n_heads, attn_dropout)
        self.sap     = StatisticalAttentivePooling(encoder_dim)

        # Input to MLP is 2D (mean + std from SAP)
        fused_dim = encoder_dim * 2

        if hidden_dim > 0:
            self.shared = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            head_in = hidden_dim
        else:
            self.shared = nn.Identity()
            head_in     = fused_dim

        self.verb_head = nn.Linear(head_in, n_verbs)
        self.noun_head = nn.Linear(head_in, n_nouns)

    def forward(
        self,
        rgb_emb: torch.Tensor,  # (B, N, D)
        u_emb:   torch.Tensor,  # (B, N, D)
        v_emb:   torch.Tensor,  # (B, N, D)
    ):
        # Cross-modal alignment: flow attends to RGB
        x_u = self.u_align(u_emb, rgb_emb)    # (B, N, D)
        x_v = self.v_align(v_emb, rgb_emb)    # (B, N, D)

        # Statistical attentive pooling over both aligned representations
        fused = self.sap(x_u, x_v)            # (B, 2D)

        x = self.shared(fused)
        return self.verb_head(x), self.noun_head(x)


# ── Metrics ───────────────────────────────────────────────────────────────────

def mean_top5_recall(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Vectorised mean per-class top-5 recall."""
    n, n_cls = logits.shape
    k        = min(5, n_cls)
    top5     = logits.topk(k, dim=1).indices
    hit      = (top5 == labels.unsqueeze(1)).any(1).float()

    hit_sum  = torch.zeros(n_cls, dtype=torch.float)
    cnt      = torch.zeros(n_cls, dtype=torch.float)
    hit_sum.scatter_add_(0, labels, hit)
    cnt.scatter_add_(0, labels, torch.ones(n, dtype=torch.float))

    mask    = cnt > 0
    recalls = hit_sum[mask] / cnt[mask]
    return recalls.mean().item() if recalls.numel() > 0 else 0.0


def combined_top5_recall(
    verb_logits: torch.Tensor,
    noun_logits: torch.Tensor,
    verb_labels: torch.Tensor,
    noun_labels: torch.Tensor,
) -> float:
    """Vectorised mean per-(verb,noun)-pair top-5 recall."""
    k        = min(5, verb_logits.shape[1], noun_logits.shape[1])
    verb_hit = (verb_logits.topk(k, 1).indices == verb_labels.unsqueeze(1)).any(1)
    noun_hit = (noun_logits.topk(k, 1).indices == noun_labels.unsqueeze(1)).any(1)
    both_hit = (verb_hit & noun_hit).float()

    n_nouns  = noun_logits.shape[1]
    pair_idx = verb_labels * n_nouns + noun_labels

    hit_sum  = torch.zeros(pair_idx.max().item() + 1, dtype=torch.float)
    cnt      = torch.zeros_like(hit_sum)
    hit_sum.scatter_add_(0, pair_idx, both_hit)
    cnt.scatter_add_(0, pair_idx, torch.ones(len(pair_idx), dtype=torch.float))

    mask    = cnt > 0
    recalls = hit_sum[mask] / cnt[mask]
    return recalls.mean().item() if recalls.numel() > 0 else 0.0


# ── Training / Eval loop ──────────────────────────────────────────────────────

def run_epoch(
    mode: str,
    encoder,
    classifier: FlowRGBFusionClassifier,
    processor,
    loader: DataLoader,
    device: str,
    scaler: GradScaler,
    optimizer=None,
    accumulation_steps: int = 1,
    verb_weight: torch.Tensor = None,
    noun_weight: torch.Tensor = None,
    freeze_encoder: bool = True,
) -> dict:
    is_train = mode == "train"
    classifier.train() if is_train else classifier.eval()
    if not freeze_encoder:
        encoder.train() if is_train else encoder.eval()
    else:
        encoder.eval()

    total_loss = total_verb_loss = total_noun_loss = 0.0
    n_batches  = 0

    all_verb_logits, all_noun_logits = [], []
    all_verb_labels, all_noun_labels = [], []

    n_total                = len(loader)
    optimizer_step_pending = 0

    if is_train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, (rgb_clips, u_clips, v_clips, verb_labels, noun_labels) in enumerate(
        tqdm(loader, desc=mode, leave=False)
    ):
        verb_labels = verb_labels.to(device, non_blocking=True)
        noun_labels = noun_labels.to(device, non_blocking=True)

        # ── Preprocess all three modalities ──────────────────────────────
        rgb_proc = _preprocess_batch_gpu(processor, rgb_clips, device)
        u_proc   = _preprocess_batch_gpu(processor, u_clips,   device)
        v_proc   = _preprocess_batch_gpu(processor, v_clips,   device)

        # ── Encode (no_grad when frozen) ─────────────────────────────────
        enc_ctx = torch.no_grad() if (freeze_encoder or not is_train) else torch.enable_grad()
        with enc_ctx:
            with (autocast() if not freeze_encoder else torch.no_grad()):
                rgb_emb = _encode(encoder, rgb_proc)  # (B, N, D)
                u_emb   = _encode(encoder, u_proc)
                v_emb   = _encode(encoder, v_proc)

        # ── Fusion + classification (AMP) ────────────────────────────────
        with autocast():
            verb_logits, noun_logits = classifier(rgb_emb, u_emb, v_emb)
            v_loss = F.cross_entropy(verb_logits, verb_labels, weight=verb_weight)
            n_loss = F.cross_entropy(noun_logits, noun_labels, weight=noun_weight)
            loss   = v_loss + n_loss

        # ── Backward ─────────────────────────────────────────────────────
        if is_train:
            scaler.scale(loss / accumulation_steps).backward()
            optimizer_step_pending += 1

            is_last_batch = (batch_idx == n_total - 1)
            if optimizer_step_pending >= accumulation_steps or is_last_batch:
                scaler.unscale_(optimizer)
                all_params = list(classifier.parameters())
                if not freeze_encoder:
                    all_params += list(encoder.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step_pending = 0

        total_loss      += loss.item()
        total_verb_loss += v_loss.item()
        total_noun_loss += n_loss.item()
        n_batches       += 1

        all_verb_logits.append(verb_logits.detach().cpu().float())
        all_noun_logits.append(noun_logits.detach().cpu().float())
        all_verb_labels.append(verb_labels.detach().cpu())
        all_noun_labels.append(noun_labels.detach().cpu())

    all_vl = torch.cat(all_verb_logits)
    all_nl = torch.cat(all_noun_logits)
    all_vt = torch.cat(all_verb_labels)
    all_nt = torch.cat(all_noun_labels)

    return {
        "loss":           total_loss / max(n_batches, 1),
        "verb_loss":      total_verb_loss / max(n_batches, 1),
        "noun_loss":      total_noun_loss / max(n_batches, 1),
        "verb_recall5":   mean_top5_recall(all_vl, all_vt),
        "noun_recall5":   mean_top5_recall(all_nl, all_nt),
        "action_recall5": combined_top5_recall(all_vl, all_nl, all_vt, all_nt),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="V-JEPA 2.1 RGB+Flow fusion fine-tuning on EPIC-KITCHENS."
    )
    # Data
    p.add_argument("--train_csv",    required=True)
    p.add_argument("--val_csv",      required=True)
    p.add_argument("--rgb_dir",      required=True,
                   help="Root dir of pre-extracted RGB .pt crops")
    p.add_argument("--flow_u_dir",   required=True,
                   help="Root dir of RAFT flow-U .pt crops")
    p.add_argument("--flow_v_dir",   required=True,
                   help="Root dir of RAFT flow-V .pt crops")
    # Model
    p.add_argument("--cache_dir",    default="/home/lucas.ueda/github/AIMS_3rd_EgoVis/cache_models")
    p.add_argument("--hidden_dim",   type=int,   default=512,
                   help="MLP hidden dim (0 = linear head)")
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--n_heads",      type=int,   default=8,
                   help="Number of attention heads in cross-modal alignment")
    p.add_argument("--attn_dropout", type=float, default=0.1)
    # Training
    p.add_argument("--batch_size",         type=int,   default=8,
                   help="Per-GPU batch size (3 encoder passes, so keep lower than RGB-only)")
    p.add_argument("--accumulation_steps", type=int,   default=2)
    p.add_argument("--epochs",             type=int,   default=20)
    p.add_argument("--lr",                 type=float, default=1e-3)
    p.add_argument("--encoder_lr",         type=float, default=0.0,
                   help="LR for encoder (0 = frozen)")
    p.add_argument("--num_workers",        type=int,   default=2)
    p.add_argument("--prefetch_factor",    type=int,   default=2)
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--model_path",         default="./experiments/vjepa2_flow_fusion")
    p.add_argument("--debug",          action="store_true")
    p.add_argument("--weighted_loss",  action="store_true")
    p.add_argument("--compile",        action="store_true",
                   help="torch.compile the classifier (PyTorch ≥ 2.0)")
    p.add_argument("--no_amp",         action="store_true")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    logger = setup_logging(args.model_path)
    set_seed(args.seed)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and (not args.no_amp)

    logger.info("=" * 70)
    logger.info("V-JEPA 2.1  EPIC-KITCHENS  RGB+Flow Fusion  Verb+Noun Classifier")
    logger.info("=" * 70)
    for k, v in vars(args).items():
        logger.info("  %-25s %s", k, v)
    logger.info("  device                    %s", device)
    logger.info("  AMP enabled               %s", use_amp)
    logger.info("=" * 70)

    if args.debug:
        args.epochs = 2
        logger.info("[DEBUG] Limiting to 200 samples and 2 epochs.")

    # ── V-JEPA 2.1 ────────────────────────────────────────────────────────
    LOCAL_REPO = os.path.join(args.cache_dir, "facebookresearch_vjepa2_main")
    os.environ["TORCH_HOME"] = args.cache_dir
    torch.hub.set_dir(args.cache_dir)

    logger.info("Loading V-JEPA 2.1 processor …")
    processor = torch.hub.load(LOCAL_REPO, "vjepa2_preprocessor", source="local")

    logger.info("Loading V-JEPA 2.1 ViT-B/384 encoder …")
    (encoder, _predictor) = torch.hub.load(
        LOCAL_REPO, "vjepa2_1_vit_base_384", source="local"
    )
    encoder.eval()
    encoder_dim    = 768
    freeze_encoder = args.encoder_lr == 0.0

    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder FROZEN.")
    else:
        logger.info("Encoder UNFROZEN  LR=%.2e", args.encoder_lr)

    encoder.to(device)

    # ── Data ──────────────────────────────────────────────────────────────
    train_df = pd.read_csv(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)

    n_verbs = int(train_df["verb_class"].max()) + 1
    n_nouns = int(train_df["noun_class"].max()) + 1
    logger.info("Verb classes : %d  |  Noun classes : %d", n_verbs, n_nouns)

    train_set = EpicKitchensFlowDataset(
        train_df, args.rgb_dir, args.flow_u_dir, args.flow_v_dir, debug=args.debug
    )
    val_set = EpicKitchensFlowDataset(
        val_df, args.rgb_dir, args.flow_u_dir, args.flow_v_dir, debug=args.debug
    )
    logger.info("Train : %d  |  Val : %d", len(train_set), len(val_set))

    loader_kwargs = dict(
        num_workers        = args.num_workers,
        pin_memory         = True,
        persistent_workers = args.num_workers > 0,
        prefetch_factor    = args.prefetch_factor if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(
        train_set,
        batch_size = args.batch_size,
        shuffle    = True,
        drop_last  = True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size = args.batch_size * 2,
        shuffle    = False,
        **loader_kwargs,
    )

    # ── Class-balanced loss weights (optional) ────────────────────────────
    verb_weight = noun_weight = None
    if args.weighted_loss:
        def inv_freq_weights(series, n_classes):
            counts = series.value_counts().reindex(range(n_classes), fill_value=1)
            w = len(series) / (n_classes * counts.values.astype(float))
            return torch.tensor(w, dtype=torch.float, device=device)

        verb_weight = inv_freq_weights(train_df["verb_class"], n_verbs)
        noun_weight = inv_freq_weights(train_df["noun_class"], n_nouns)
        logger.info("Using inverse-frequency loss weights.")

    # ── Fusion classifier ─────────────────────────────────────────────────
    classifier = FlowRGBFusionClassifier(
        encoder_dim  = encoder_dim,
        n_verbs      = n_verbs,
        n_nouns      = n_nouns,
        n_heads      = args.n_heads,
        hidden_dim   = args.hidden_dim,
        dropout      = args.dropout,
        attn_dropout = args.attn_dropout,
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        logger.info("Compiling classifier with torch.compile …")
        classifier = torch.compile(classifier)

    n_cls_params = sum(p.numel() for p in classifier.parameters()) / 1e6
    n_enc_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    logger.info("Classifier parameters : %.2f M", n_cls_params)
    logger.info("Encoder parameters    : %.2f M", n_enc_params)

    # ── Optimiser + scheduler + scaler ───────────────────────────────────
    param_groups = [{"params": classifier.parameters(), "lr": args.lr}]
    if not freeze_encoder:
        param_groups.append({"params": encoder.parameters(), "lr": args.encoder_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler(enabled=use_amp)

    # ── Training loop ─────────────────────────────────────────────────────
    best_action_r5 = -1.0
    best_epoch     = -1

    logger.info("\n%s\nSTART TRAINING\n%s", "=" * 70, "=" * 70)

    for epoch in range(args.epochs):
        logger.info("\n── Epoch %d / %d ──", epoch + 1, args.epochs)

        train_m = run_epoch(
            mode="train",
            encoder=encoder,
            classifier=classifier,
            processor=processor,
            loader=train_loader,
            device=device,
            scaler=scaler,
            optimizer=optimizer,
            accumulation_steps=args.accumulation_steps,
            verb_weight=verb_weight,
            noun_weight=noun_weight,
            freeze_encoder=freeze_encoder,
        )
        scheduler.step()

        logger.info(
            "  TRAIN  loss=%.4f  v_loss=%.4f  n_loss=%.4f"
            "  verb_R5=%.4f  noun_R5=%.4f  action_R5=%.4f",
            train_m["loss"], train_m["verb_loss"], train_m["noun_loss"],
            train_m["verb_recall5"], train_m["noun_recall5"], train_m["action_recall5"],
        )

        val_m = run_epoch(
            mode="val",
            encoder=encoder,
            classifier=classifier,
            processor=processor,
            loader=val_loader,
            device=device,
            scaler=scaler,
            optimizer=None,
            verb_weight=verb_weight,
            noun_weight=noun_weight,
            freeze_encoder=freeze_encoder,
        )

        logger.info(
            "  VAL    loss=%.4f  v_loss=%.4f  n_loss=%.4f"
            "  verb_R5=%.4f  noun_R5=%.4f  action_R5=%.4f",
            val_m["loss"], val_m["verb_loss"], val_m["noun_loss"],
            val_m["verb_recall5"], val_m["noun_recall5"], val_m["action_recall5"],
        )

        lrs = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
        logger.info("  LR : %s", " | ".join(lrs))

        # ── Checkpoint ────────────────────────────────────────────────────
        action_r5 = val_m["action_recall5"]
        if action_r5 > best_action_r5:
            best_action_r5 = action_r5
            best_epoch     = epoch + 1

            raw_cls = (
                classifier._orig_mod
                if hasattr(classifier, "_orig_mod")
                else classifier
            )
            ckpt = {
                "epoch":            epoch + 1,
                "classifier_state": raw_cls.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "scheduler_state":  scheduler.state_dict(),
                "scaler_state":     scaler.state_dict(),
                "val_metrics":      val_m,
                "n_verbs":          n_verbs,
                "n_nouns":          n_nouns,
                "encoder_dim":      encoder_dim,
                "hidden_dim":       args.hidden_dim,
                "dropout":          args.dropout,
                "n_heads":          args.n_heads,
                "attn_dropout":     args.attn_dropout,
            }
            if not freeze_encoder:
                ckpt["encoder_state"] = encoder.state_dict()

            ckpt_path = os.path.join(args.model_path, "best_classifier.pt")
            torch.save(ckpt, ckpt_path)
            logger.info("  ★ New best  action_R5=%.4f  → %s", action_r5, ckpt_path)

    logger.info("\n%s\nTRAINING COMPLETE", "=" * 70)
    logger.info("  Best epoch      : %d", best_epoch)
    logger.info("  Best action_R@5 : %.4f", best_action_r5)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()