import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from sta.bbox_prediction.models.siglip_cosmos_detr import SiglipCosmosDETR
from sta.bbox_prediction.read_data.filter_and_use_dataset import STANounDetectionDataset, Ego4DPaths
from sta.bbox_prediction.models.siglip_detr import SiglipDETR
from sta.bbox_prediction.models.criterion import SetCriterion
from sta.bbox_prediction.train.collate import collate
from sta.bbox_prediction.models.matcher import HungarianMatcher
from sta.bbox_prediction.read_data.transform import SiglipFrameTransform


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    model_name = "google/siglip-base-patch16-224"

    sta_json = Path("/hadatasets/ego4d_data/v1/annotations/fho_sta_train.json")
    full_scale_dir = Path("/hadatasets/ego4d_data/v1/video_540ss")

    data = json.load(open(sta_json, "r"))
    num_nouns = len(data["noun_categories"])

    tf = SiglipFrameTransform(model_name)
    paths = Ego4DPaths(full_scale_dir=full_scale_dir, sta_json_path=sta_json)

    ds = STANounDetectionDataset(paths, transform_query=tf, min_box_size=1, keep_metadata=False, cosmos_cache_dir=None)
    
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    
    dl = DataLoader(
        ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        collate_fn=collate,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    model = SiglipDETR(
        siglip_name=model_name, 
        num_classes=num_nouns, 
        num_queries=50,
        num_decoder_layers=6,
        dim_feedforward=2048,
        train_backbone=False
    )
    model.to(device)

    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    criterion = SetCriterion(num_classes=num_nouns, matcher=matcher, eos_coef=0.1).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    
    BASE_LR = 2.5e-4
    
    opt = torch.optim.AdamW(params, lr=BASE_LR, weight_decay=1e-4, betas=(0.9, 0.999))

    NUM_EPOCHS = 20
    num_training_steps = len(dl) * NUM_EPOCHS
    num_warmup_steps = min(1000, num_training_steps // 10)
    
    scheduler = get_cosine_schedule_with_warmup(
        opt, 
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=0.01
    )

    scaler = GradScaler()
    use_amp = True

    Path("runs").mkdir(exist_ok=True)
    log_path = Path("runs/train_log_optimized.jsonl")
    log_f = open(log_path, "w", encoding="utf-8")
    global_step = 0

    run_dir = Path("runs/detr_siglip_optimized")
    writer = SummaryWriter(log_dir=str(run_dir))
    
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 0.1

    model.train()
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_losses = []
        
        for batch_idx, (pixel_values, cosmos_feats, targets) in enumerate(pbar):
            pixel_values = pixel_values.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with autocast(enabled=use_amp):
                outputs = model(pixel_values)
                losses = criterion(outputs, targets)
                loss = losses["total_loss"]
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                scheduler.step()
            
            current_lr = float(scheduler.get_last_lr()[0])
            
            for k, v in losses.items():
                writer.add_scalar(f"train/{k}", float(v.detach().cpu()), global_step)
            writer.add_scalar("train/lr", current_lr, global_step)

            step_log = {k: float(v.detach().cpu()) for k, v in losses.items()}
            step_log.update({
                "epoch": int(epoch), 
                "step": int(global_step),
                "lr": float(current_lr),
                "batch_size": int(BATCH_SIZE)
            })
            log_f.write(json.dumps(step_log) + "\n")
            log_f.flush()
            
            epoch_losses.append(losses["total_loss"].item())
            global_step += 1

            pbar.set_postfix({
                **{k: f"{float(v.detach().cpu()):.3f}" for k, v in losses.items()},
                "lr": f"{current_lr:.2e}"
            })
        
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\nEpoch {epoch+1} avg loss: {avg_epoch_loss:.4f}")
        
        Path("checkpoints").mkdir(exist_ok=True)
        checkpoint_path = f"checkpoints/detr_siglip_optimized_epoch{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_epoch_loss,
            'global_step': global_step,
        }, checkpoint_path)
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), "checkpoints/detr_siglip_optimized_best.pt")
            print(f"Saved best model with loss: {best_loss:.4f}")

    writer.close()
    print("TensorBoard logs in:", run_dir)

    log_f.close()
    print("Wrote logs to:", log_path)


if __name__ == "__main__":
    main()
