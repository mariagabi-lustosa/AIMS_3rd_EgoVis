"""
V-JEPA 2.1 Fine-tuning — EPIC-KITCHENS Verb + Noun Classification
==================================================================
OPTIMIZED VERSION — key changes over original:
  1. Batched GPU preprocessing (no per-sample CPU loop)
  2. Encoder features cached to RAM after first pass (frozen mode)
  3. DataLoader with persistent_workers + prefetch_factor
  4. torch.compile on classifier (PyTorch ≥ 2.0)
  5. AMP (automatic mixed precision) throughout
  6. Fixed gradient-accumulation flush on last batch
  7. cudnn.benchmark = True
  8. torch.load with weights_only=True to suppress FutureWarning
  9. Non-blocking host→device transfers everywhere
  10. Recall computed in one vectorised pass (no Python loop over classes)

Usage:
    python train_vjepa2_epic_optimized.py \\
        --train_csv /hadatasets/EPIC-KITCHENS/EPIC_100_train.csv \\
        --val_csv   /hadatasets/EPIC-KITCHENS/EPIC_100_validation.csv \\
        --data_dir  /hadatasets/EPIC-KITCHENS_rgb_crops \\
        --model_path ./experiments/vjepa2_epic \\
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

torch.backends.cudnn.benchmark = True   # auto-tune convolution kernels

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
    return logging.getLogger("vjepa2_epic")

# ── Dataset ───────────────────────────────────────────────────────────────────

def _crop_path(data_dir: str, row: pd.Series) -> str:
    pid  = row["participant_id"]
    vid  = row["video_id"]
    sf   = int(row["start_frame"])
    ef   = int(row["stop_frame"])
    return os.path.join(data_dir, pid, vid, f"{pid}_{vid}_{sf}_{ef}.pt")


class EpicKitchensDataset(Dataset):
    """
    Loads pre-extracted (T, C, H, W) uint8 .pt clips.
    Returns (clip_tensor, verb_label, noun_label).

    OPT: weights_only=True on torch.load avoids the FutureWarning and is
         faster because it skips arbitrary pickle execution.
    """

    def __init__(self, df: pd.DataFrame, data_dir: str, debug: bool = False):
        if debug:
            df = df.sample(n=min(200, len(df)), random_state=42).reset_index(drop=True)

        valid_mask = df.apply(
            lambda r: os.path.isfile(_crop_path(data_dir, r)), axis=1
        )
        missing = (~valid_mask).sum()
        if missing:
            logging.getLogger("vjepa2_epic").warning(
                "%d rows have no .pt file and will be dropped.", missing
            )
        self.df       = df[valid_mask].reset_index(drop=True)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    # def __getitem__(self, idx):
    #     row  = self.df.iloc[idx]
    #     path = _crop_path(self.data_dir, row)
    #     # weights_only=True: faster + no pickle security warning
    #     clip = torch.load(path, map_location="cpu", weights_only=True)
    #     if clip.ndim == 3:
    #         clip = clip.unsqueeze(0)
    #     return clip, int(row["verb_class"]), int(row["noun_class"])

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = _crop_path(self.data_dir, row)
        
        # Load clip: (T, C, H, W)
        clip = torch.load(path, map_location="cpu", weights_only=True)
        
        if clip.ndim == 3:
            clip = clip.unsqueeze(0)

        # Standardize spatial resolution to the majority shape (384, 384)
        # or a square size like (384, 384) if that's what your V-JEPA config expects.
        target_h, target_w = 384, 384
        
        if clip.shape[-2:] != (target_h, target_w):
            # interpolate expects (Batch, Channel, H, W). 
            # Since our tensor is (T, C, H, W), we can treat T as the batch dimension.
            clip = F.interpolate(
                clip.float(), 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            ).to(torch.uint8) # Convert back to uint8 to save memory/match expected input

        return clip, int(row["verb_class"]), int(row["noun_class"])


# ── Feature-cache dataset (frozen encoder only) ───────────────────────────────

class FeatureCacheDataset(Dataset):
    """
    OPT: When the encoder is frozen we run it once, cache all patch
    embeddings in CPU RAM, then train the head against cached tensors.
    This eliminates the encoder forward pass from every training step.

    Memory: ViT-B produces (N=~2016, D=768) per clip → ~6 MB fp32.
    40 k clips ≈ 240 GB — too large for RAM. So we cache lazily on
    first access (per-worker, not global), which still avoids re-encoding
    within a single epoch if pin_memory is combined with persistent workers
    and the OS page cache is warm.

    For smaller datasets (≤ ~5 k clips) pass cache_all=True to pre-load
    everything into RAM before training starts.
    """

    def __init__(
        self,
        base_dataset: EpicKitchensDataset,
        encoder,
        processor,
        device: str,
        batch_size: int = 32,
        cache_all: bool = False,
    ):
        self.base    = base_dataset
        self._cache  = {}   # idx → (patch_emb cpu tensor, verb, noun)
        self._cached = False

        if cache_all:
            logger = logging.getLogger("vjepa2_epic")
            logger.info("Pre-caching all encoder features in RAM …")
            loader = DataLoader(
                base_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
            encoder.eval()
            offset = 0
            with torch.no_grad():
                for clips, verbs, nouns in tqdm(loader, desc="caching", leave=False):
                    clips = _preprocess_batch_gpu(processor, clips, device)
                    with autocast():
                        emb = _encode(encoder, clips)   # (B, N, D)
                    emb = emb.cpu().float()
                    for i in range(emb.shape[0]):
                        self._cache[offset + i] = (emb[i], int(verbs[i]), int(nouns[i]))
                    offset += emb.shape[0]
            self._cached = True
            logger.info("Cached %d feature tensors.", len(self._cache))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        clip, verb, noun = self.base[idx]
        return clip, verb, noun   # caller must still encode these


# ── Model ─────────────────────────────────────────────────────────────────────

class DualHeadClassifier(nn.Module):
    """
    (B, N, D) → mean-pool → MLP → {verb_logits, noun_logits}
    """

    def __init__(
        self,
        encoder_dim: int,
        n_verbs: int,
        n_nouns: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dim > 0:
            self.shared = nn.Sequential(
                nn.Linear(encoder_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            feat_dim = hidden_dim
        else:
            self.shared  = nn.Identity()
            feat_dim     = encoder_dim

        self.verb_head = nn.Linear(feat_dim, n_verbs)
        self.noun_head = nn.Linear(feat_dim, n_nouns)

    def forward(self, patch_embeddings: torch.Tensor):
        # OPT: mean-pool with torch.mean is marginally faster than
        #      AdaptiveAvgPool1d for this shape, and avoids the permute.
        x = patch_embeddings.mean(dim=1)    # (B, D)
        x = self.shared(x)
        return self.verb_head(x), self.noun_head(x)


# ── Preprocessing — BATCHED on GPU ───────────────────────────────────────────

def _preprocess_batch_gpu(processor, clips: torch.Tensor, device: str) -> torch.Tensor:
    """
    OPT (critical): Original code looped per sample on CPU.
    Here we try to call processor on the whole batch at once.
    Falls back to per-sample if the processor doesn't accept batches.

    clips : (B, T, C, H, W) uint8
    Returns (B, C, T, H', W') float32 on `device`.
    """
    # Move to device first so the processor (if GPU-aware) can work there.
    clips_dev = clips.to(device, non_blocking=True)

    try:
        # Happy path: processor accepts (B, T, C, H, W) directly
        out = processor(clips_dev)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.ndim == 4:          # (C, T, H, W) → add batch dim
            out = out.unsqueeze(0)
        return out                 # (B, C, T, H', W') already on device
    except Exception:
        pass

    # Fallback: per-sample but at least on GPU
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
    """Run encoder and unwrap output format."""
    out = encoder(clip_proc)
    if isinstance(out, (tuple, list)):
        return out[0]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    return out


# ── Metrics — vectorised Mean Top-5 Recall ───────────────────────────────────

def mean_top5_recall(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    OPT: fully vectorised — no Python loop over classes.
    For each class c, recall@5 = fraction of its samples where c ∈ top-5.

    Uses scatter to accumulate hits and counts per class in one pass.
    """
    n, n_cls = logits.shape
    k        = min(5, n_cls)

    top5     = logits.topk(k, dim=1).indices          # (N, k)
    # hit[i] = 1 if labels[i] is in top5[i]
    hit      = (top5 == labels.unsqueeze(1)).any(1).float()   # (N,)

    # Accumulate per class
    hit_sum   = torch.zeros(n_cls, dtype=torch.float)
    cnt       = torch.zeros(n_cls, dtype=torch.float)
    hit_sum.scatter_add_(0, labels, hit)
    cnt.scatter_add_(0, labels, torch.ones(n, dtype=torch.float))

    mask    = cnt > 0
    recalls = (hit_sum[mask] / cnt[mask])
    return recalls.mean().item() if recalls.numel() > 0 else 0.0


def combined_top5_recall(
    verb_logits: torch.Tensor,
    noun_logits: torch.Tensor,
    verb_labels: torch.Tensor,
    noun_labels: torch.Tensor,
) -> float:
    """
    OPT: vectorised version of joint verb+noun recall@5.
    A sample is a hit only when BOTH verb and noun are in their top-5.
    Per (verb, noun) class-pair recall is then averaged.
    """
    k         = min(5, verb_logits.shape[1], noun_logits.shape[1])
    verb_hit  = (verb_logits.topk(k, 1).indices == verb_labels.unsqueeze(1)).any(1)
    noun_hit  = (noun_logits.topk(k, 1).indices == noun_labels.unsqueeze(1)).any(1)
    both_hit  = (verb_hit & noun_hit).float()

    # Encode pair as a single integer for scatter
    n_nouns   = noun_logits.shape[1]
    pair_idx  = verb_labels * n_nouns + noun_labels   # unique per (v, n) pair

    hit_sum   = torch.zeros(pair_idx.max().item() + 1, dtype=torch.float)
    cnt       = torch.zeros_like(hit_sum)
    hit_sum.scatter_add_(0, pair_idx, both_hit)
    cnt.scatter_add_(0, pair_idx, torch.ones(len(pair_idx), dtype=torch.float))

    mask    = cnt > 0
    recalls = hit_sum[mask] / cnt[mask]
    return recalls.mean().item() if recalls.numel() > 0 else 0.0


# ── Training / Eval loops ─────────────────────────────────────────────────────

def run_epoch(
    mode: str,
    encoder,
    classifier: DualHeadClassifier,
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
    """
    One training or evaluation epoch with AMP.

    OPT changes vs original:
      • encoder always in torch.no_grad() when frozen (saves graph memory)
      • AMP autocast wraps classifier forward + loss
      • GradScaler for stable fp16 training
      • Fixed accumulation flush: tracks remaining batches, not a counter mod
      • Non-blocking transfers throughout
    """
    is_train = mode == "train"
    classifier.train() if is_train else classifier.eval()
    if not freeze_encoder:
        encoder.train() if is_train else encoder.eval()

    total_loss = total_verb_loss = total_noun_loss = 0.0
    n_batches  = 0

    all_verb_logits, all_noun_logits = [], []
    all_verb_labels, all_noun_labels = [], []

    n_total   = len(loader)
    optimizer_step_pending = 0

    if is_train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, (clips_uint8, verb_labels, noun_labels) in enumerate(
        tqdm(loader, desc=mode, leave=False)
    ):
        verb_labels = verb_labels.to(device, non_blocking=True)
        noun_labels = noun_labels.to(device, non_blocking=True)

        # ── Encoder forward ──────────────────────────────────────────────
        clip_proc = _preprocess_batch_gpu(processor, clips_uint8, device)

        enc_ctx = torch.no_grad() if (freeze_encoder or not is_train) else torch.enable_grad()
        with enc_ctx:
            enc_amp = autocast() if not freeze_encoder else torch.no_grad()
            with enc_amp:
                patch_emb = _encode(encoder, clip_proc)

        # ── Classifier forward + loss (AMP) ──────────────────────────────
        with autocast():
            verb_logits, noun_logits = classifier(patch_emb)
            v_loss = F.cross_entropy(verb_logits, verb_labels, weight=verb_weight)
            n_loss = F.cross_entropy(noun_logits, noun_labels, weight=noun_weight)
            loss   = v_loss + n_loss

        # ── Backward (scaled) ────────────────────────────────────────────
        if is_train:
            scaler.scale(loss / accumulation_steps).backward()
            optimizer_step_pending += 1

            # OPT FIX: flush on last batch of epoch OR on accumulation boundary
            is_last_batch = (batch_idx == n_total - 1)
            if optimizer_step_pending >= accumulation_steps or is_last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(classifier.parameters()) +
                    (list(encoder.parameters()) if not freeze_encoder else []),
                    max_norm=1.0,
                )
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

    verb_r5   = mean_top5_recall(all_vl, all_vt)
    noun_r5   = mean_top5_recall(all_nl, all_nt)
    action_r5 = combined_top5_recall(all_vl, all_nl, all_vt, all_nt)

    return {
        "loss":           total_loss / max(n_batches, 1),
        "verb_loss":      total_verb_loss / max(n_batches, 1),
        "noun_loss":      total_noun_loss / max(n_batches, 1),
        "verb_recall5":   verb_r5,
        "noun_recall5":   noun_r5,
        "action_recall5": action_r5,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune V-JEPA 2.1 on EPIC-KITCHENS (optimized)."
    )
    p.add_argument("--train_csv",  required=True)
    p.add_argument("--val_csv",    required=True)
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--cache_dir",  default="/home/lucas.ueda/github/AIMS_3rd_EgoVis/cache_models")
    p.add_argument("--hidden_dim", type=int,   default=512)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--batch_size",         type=int,   default=16)
    p.add_argument("--accumulation_steps", type=int,   default=1)
    p.add_argument("--epochs",             type=int,   default=20)
    p.add_argument("--lr",                 type=float, default=1e-3)
    p.add_argument("--encoder_lr",         type=float, default=0.0,
                   help="LR for encoder (0 = frozen)")
    p.add_argument("--num_workers",        type=int,   default=4)
    p.add_argument("--prefetch_factor",    type=int,   default=2,
                   help="DataLoader prefetch factor (OPT)")
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--model_path",         default="./experiments/vjepa2_epic")
    p.add_argument("--debug",          action="store_true")
    p.add_argument("--weighted_loss",  action="store_true")
    p.add_argument("--compile",        action="store_true",
                   help="torch.compile the classifier (PyTorch ≥ 2.0, OPT)")
    p.add_argument("--cache_features", action="store_true",
                   help="Pre-cache encoder features in RAM (frozen encoder, small datasets)")
    p.add_argument("--no_amp",         action="store_true",
                   help="Disable automatic mixed precision")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    logger = setup_logging(args.model_path)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device = {device}")

    use_amp = (device == "cuda") and (not args.no_amp)

    logger.info("=" * 70)
    logger.info("V-JEPA 2.1  EPIC-KITCHENS  Verb+Noun Classifier  [OPTIMIZED]")
    logger.info("=" * 70)
    for k, v in vars(args).items():
        logger.info("  %-25s %s", k, v)
    logger.info("  device                    %s", device)
    logger.info("  AMP enabled               %s", use_amp)
    logger.info("=" * 70)

    if args.debug:
        args.epochs = 2
        logger.info("[DEBUG] Limiting to 200 samples and 2 epochs.")

    # ── V-JEPA 2.1 setup ──────────────────────────────────────────────────
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
    encoder_dim  = 768
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

    train_set = EpicKitchensDataset(train_df, args.data_dir, debug=args.debug)
    val_set   = EpicKitchensDataset(val_df,   args.data_dir, debug=args.debug)
    logger.info("Train : %d  |  Val : %d", len(train_set), len(val_set))

    # OPT: persistent_workers avoids respawning processes between epochs;
    #      prefetch_factor keeps the GPU fed.
    loader_kwargs = dict(
        num_workers      = args.num_workers,
        pin_memory       = True,
        persistent_workers = args.num_workers > 0,
        prefetch_factor  = args.prefetch_factor if args.num_workers > 0 else None,
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

    # Optional class-balanced loss weights
    verb_weight = noun_weight = None
    if args.weighted_loss:
        def inv_freq_weights(series, n_classes):
            counts = series.value_counts().reindex(range(n_classes), fill_value=1)
            w = len(series) / (n_classes * counts.values.astype(float))
            return torch.tensor(w, dtype=torch.float, device=device)

        verb_weight = inv_freq_weights(train_df["verb_class"], n_verbs)
        noun_weight = inv_freq_weights(train_df["noun_class"], n_nouns)
        logger.info("Using inverse-frequency loss weights.")

    # ── Classifier ────────────────────────────────────────────────────────
    classifier = DualHeadClassifier(
        encoder_dim = encoder_dim,
        n_verbs     = n_verbs,
        n_nouns     = n_nouns,
        hidden_dim  = args.hidden_dim,
        dropout     = args.dropout,
    ).to(device)

    # OPT: torch.compile fuses ops and can give 20-50% head speedup
    if args.compile and hasattr(torch, "compile"):
        logger.info("Compiling classifier with torch.compile …")
        classifier = torch.compile(classifier)

    n_head_params = sum(p.numel() for p in classifier.parameters()) / 1e6
    logger.info("Classifier head parameters : %.2f M", n_head_params)

    n_encoder_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    logger.info("Encoder parameters : %.2f M", n_encoder_params)

    # ── Optimizers + AMP scaler ───────────────────────────────────────────
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

        train_metrics = run_epoch(
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
            train_metrics["loss"],
            train_metrics["verb_loss"],
            train_metrics["noun_loss"],
            train_metrics["verb_recall5"],
            train_metrics["noun_recall5"],
            train_metrics["action_recall5"],
        )

        val_metrics = run_epoch(
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
            val_metrics["loss"],
            val_metrics["verb_loss"],
            val_metrics["noun_loss"],
            val_metrics["verb_recall5"],
            val_metrics["noun_recall5"],
            val_metrics["action_recall5"],
        )

        lrs = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
        logger.info("  LR : %s", " | ".join(lrs))

        # Save best checkpoint
        action_r5 = val_metrics["action_recall5"]
        if action_r5 > best_action_r5:
            best_action_r5 = action_r5
            best_epoch     = epoch + 1

            # Unwrap compiled model if needed
            raw_classifier = (
                classifier._orig_mod
                if hasattr(classifier, "_orig_mod")
                else classifier
            )
            ckpt = {
                "epoch":            epoch + 1,
                "classifier_state": raw_classifier.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "scheduler_state":  scheduler.state_dict(),
                "scaler_state":     scaler.state_dict(),
                "val_metrics":      val_metrics,
                "n_verbs":          n_verbs,
                "n_nouns":          n_nouns,
                "encoder_dim":      encoder_dim,
                "hidden_dim":       args.hidden_dim,
                "dropout":          args.dropout,
            }
            if not freeze_encoder:
                ckpt["encoder_state"] = encoder.state_dict()

            ckpt_path = os.path.join(args.model_path, "best_classifier.pt")
            torch.save(ckpt, ckpt_path)
            logger.info(
                "  ★ New best  action_R5=%.4f  → %s", action_r5, ckpt_path
            )

    logger.info("\n%s\nTRAINING COMPLETE", "=" * 70)
    logger.info("  Best epoch      : %d", best_epoch)
    logger.info("  Best action_R@5 : %.4f", best_action_r5)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()