from pathlib import Path
import json
import torch

from sta.bbox_prediction.read_data.filter_and_use_dataset import Ego4DPaths, STANounDetectionDataset
from sta.bbox_prediction.read_data.transform import SiglipFrameTransform
from sta.bbox_prediction.models.siglip_query import SiglipQueryEncoder


def main():
    model_name = "google/siglip-base-patch16-224"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = Ego4DPaths(
        full_scale_dir=Path("/home/gabi/Documents/materiais_ic/ViT/ego4d/full_scale"),
        sta_json_path=Path("/home/gabi/Documents/materiais_ic/ViT/ego4d/annotations/fho_sta_train.json"),
    )

    tf = SiglipFrameTransform(model_name)
    ds = STANounDetectionDataset(paths, transform_query=tf, keep_metadata=True)

    encoder = SiglipQueryEncoder(model_name, train_backbone=False).to(device).eval()

    sample = ds[0]
    pv = sample["query_pixel_values"].unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        tokens, pooled = encoder(pv)

    print("query_pixel_values:", pv.shape)
    print("tokens:", tokens.shape)   # [1, N, D]
    print("pooled:", pooled.shape)   # [1, D]
    print("gt boxes:", len(sample["gt_boxes"]))
    print("gt nouns:", len(sample["gt_nouns"]))
    print("video:", sample["video_uid"], "frame:", sample["frame"])


if __name__ == "__main__":
    main()
