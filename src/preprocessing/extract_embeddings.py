import os
import argparse
import logging
import torch
import torch.nn.functional as F
import traceback
from tqdm import tqdm
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("vjepa_extractor")

@torch.no_grad()
def extract_raw_embeddings():
    parser = argparse.ArgumentParser(description="Extract raw V-JEPA 2.1 tokens from .pt clips")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to RGB, U, or V folder")
    parser.add_argument("--cache_dir", type=str, default="./cache_models")
    parser.add_argument("--keep_batch_dim", action="store_true", help="Save as (1, N, D) instead of (N, D)")
    parser.add_argument("--verbose", action="store_true", help="Log every step for every file")
    args = parser.parse_args()

    logger = setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_path = Path(args.input_dir).resolve()
    output_path = input_path.with_name(f"{input_path.name}_emb")
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info(f"DEVICE: {device}")
    logger.info(f"INPUT:  {input_path}")
    logger.info(f"OUTPUT: {output_path}")
    logger.info("=" * 70)

    # 1. Model Loading
    LOCAL_REPO = os.path.join(args.cache_dir, "facebookresearch_vjepa2_main")
    os.environ["TORCH_HOME"] = args.cache_dir
    torch.hub.set_dir(args.cache_dir)
    
    logger.info("Loading V-JEPA 2.1 Model Components...")
    processor = torch.hub.load(LOCAL_REPO, "vjepa2_preprocessor", source="local")
    encoder, _ = torch.hub.load(LOCAL_REPO, "vjepa2_1_vit_base_384", source="local")
    encoder.to(device).eval()

    pt_files = list(input_path.rglob("*.pt"))
    logger.info(f"Processing {len(pt_files)} files...")

    error_count = 0
    sample_shape = None

    for file_path in tqdm(pt_files, desc="Extracting Tokens"):
        try:
            if args.verbose: logger.info(f"Step 1: Loading {file_path.name}")
            clip = torch.load(file_path, map_location="cpu", weights_only=True)
            
            # Initial Shape Check
            if clip.ndim == 3:
                clip = clip.unsqueeze(0)
            
            if args.verbose: logger.info(f"  - Original Shape: {clip.shape} | Dtype: {clip.dtype}")

            # 2. Spatial Resizing
            target_h, target_w = 384, 384
            if clip.shape[-2:] != (target_h, target_w):
                if args.verbose: logger.info(f"Step 2: Resizing to {target_h}x{target_w}")
                clip = F.interpolate(
                    clip.float(), 
                    size=(target_h, target_w), 
                    mode="bilinear", 
                    align_corners=False
                ).to(clip.dtype)

            # 3. Dtype / Scaling
            if clip.dtype != torch.uint8:
                if clip.min() < 0:
                    if args.verbose: logger.info("Step 3: Scaling signed flow to uint8")
                    clip = ((clip.float().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
                else:
                    if args.verbose: logger.info("Step 3: Casting float RGB to uint8")
                    clip = clip.to(torch.uint8)
            
            # 4. Preparation for Processor
            # V-JEPA Processor expects (B, T, C, H, W)
            input_tensor = clip.to(device, non_blocking=True)
            if args.verbose: logger.info(f"Step 4: Input to processor shape: {input_tensor.shape}")

            # 5. Processor
            clip_proc = processor(input_tensor)
            if isinstance(clip_proc, (list, tuple)):
                clip_proc = clip_proc[0]
                
            if clip_proc.ndim == 4:
                clip_proc = clip_proc.unsqueeze(0)
            
            if args.verbose: logger.info(f"Step 5: Processor output shape: {clip_proc.shape}")

            # 6. Encoder
            emb_tokens = encoder(clip_proc)
            if isinstance(emb_tokens, (tuple, list)):
                emb_tokens = emb_tokens[0]
            
            if args.verbose: logger.info(f"Step 6: Encoder output shape: {emb_tokens.shape}")

            # 7. Final Squeeze and Save
            if not args.keep_batch_dim:
                emb_tokens = emb_tokens.squeeze(0) 
            
            if sample_shape is None:
                sample_shape = list(emb_tokens.shape)

            rel_path = file_path.relative_to(input_path)
            target_file = output_path / rel_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(emb_tokens.cpu(), target_file)
            
        except Exception as e:
            error_count += 1
            logger.error("-" * 30)
            logger.error(f"CRITICAL ERROR on {file_path.name}")
            logger.error(f"Exception Type: {type(e).__name__}")
            logger.error(f"Exception Message: {str(e)}")
            # We print the full traceback for the first 5 errors to find the line number
            if error_count <= 5:
                logger.error("FULL TRACEBACK:")
                logger.error(traceback.format_exc())
            logger.error("-" * 30)
            
            if error_count > 100:
                logger.error("Over 100 errors encountered. Stopping for safety.")
                break
            continue

    logger.info("\n" + "=" * 70)
    logger.info("EXTRACTION SUMMARY")
    logger.info(f"Successfully processed: {len(pt_files) - error_count}")
    logger.info(f"Errors encountered:     {error_count}")
    if sample_shape:
        logger.info(f"Last Saved Shape:       {sample_shape}")
    logger.info("=" * 70)

if __name__ == "__main__":
    extract_raw_embeddings()