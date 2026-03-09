import json
from pyexpat import model
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sta.bbox_prediction.models.siglip_cosmos_detr import SiglipCosmosDETR
from sta.bbox_prediction.read_data.filter_and_use_dataset import STANounDetectionDataset, Ego4DPaths
from sta.bbox_prediction.models.siglip_detr import SiglipDETR
from sta.bbox_prediction.models.criterion import SetCriterion
from sta.bbox_prediction.train.collate import collate
from sta.bbox_prediction.models.matcher import HungarianMatcher
from sta.bbox_prediction.read_data.transform import SiglipFrameTransform


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "google/siglip-base-patch16-224"

    sta_json= Path("/hadatasets/ego4d_data/v1/annotations/fho_sta_train.json")
    full_scale_dir = Path("/hadatasets/ego4d_data/v1/video_540ss")

    data = json.load(open(sta_json, "r"))
    num_nouns = len(data["noun_categories"])

    tf = SiglipFrameTransform(model_name)
    paths = Ego4DPaths(full_scale_dir=full_scale_dir, sta_json_path=sta_json)

    ds = STANounDetectionDataset(paths, transform_query=tf, min_box_size=1, keep_metadata=False, cosmos_cache_dir=None)
    #ds = STANounDetectionDataset(paths, transform_query=tf, min_box_size=1, keep_metadata=False, cosmos_cache_dir=Path("/ego4d/cosmos_cache_sta_train"))
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate)

    #model = SiglipCosmosDETR(siglip_name=model_name, num_classes=num_nouns, num_queries=50, train_backbone=False)
    model = SiglipDETR(siglip_name=model_name, num_classes=num_nouns, num_queries=50, train_backbone=False)
    model.to(device)

    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    criterion = SetCriterion(num_classes=num_nouns, matcher=matcher, eos_coef=0.1).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

    Path("runs").mkdir(exist_ok=True)
    log_path = Path("runs/train_log.jsonl")
    log_f = open(log_path, "w", encoding="utf-8")
    global_step = 0

    run_dir = Path("runs/detr_siglip")
    writer = SummaryWriter(log_dir=str(run_dir))

    model.train()
    for epoch in range(5):
        pbar = tqdm(dl, desc=f"Epoch {epoch}")
        for pixel_values, cosmos_feats, targets in pbar:
            pixel_values = pixel_values.to(device)
            #cosmos_feats = cosmos_feats.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(pixel_values)
            #outputs = model(pixel_values, cosmos_feats)
            losses = criterion(outputs, targets)

            #print("Available loss keys:", losses.keys())
            loss = losses["total_loss"]

            for k, v in losses.items():
                writer.add_scalar(f"train/{k}", float(v.detach().cpu()), global_step)
                #writer.add_scalar("train/lr", opt.param_groups[0]["lr"], global_step)

            step_log = {k: float(v.detach().cpu()) for k, v in losses.items()}
            step_log.update({"epoch": epoch, "step": global_step})
            log_f.write(json.dumps(step_log) + "\n")
            log_f.flush()
            global_step += 1

            opt.zero_grad()
            
            loss.backward()
            opt.step()

            pbar.set_postfix({k: float(v.detach().cpu()) for k, v in losses.items()})

    writer.close()
    print("TensorBoard logs in:", run_dir)

    log_f.close()
    print("Wrote logs to:", log_path)

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/detr_siglip.pt")


if __name__ == "__main__":
    main()