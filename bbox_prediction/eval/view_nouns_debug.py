import json
import torch

from pathlib import Path

from sta.bbox_prediction.read_data.filter_and_use_dataset import Ego4DPaths, STANounDetectionDataset
from sta.bbox_prediction.read_data.transform import SiglipFrameTransform
from sta.bbox_prediction.models.siglip_detr import SiglipDETR
from sta.bbox_prediction.eval.plot_utils import plot_prediction


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "google/siglip-base-patch16-224"

    ego4d_root = Path("/home/gabi/Documents/materiais_ic/ViT/ego4d")
    sta_json = ego4d_root / "annotations/fho_sta_train.json"
    full_scale_dir = ego4d_root / "full_scale"

    # label map
    data = json.load(open(sta_json, "r"))
    noun_map = {c["id"]: c["name"] for c in data["noun_categories"]}
    num_nouns = len(noun_map)

    # dataset (keep_metadata=True so we get query_rgb)
    tf = SiglipFrameTransform(model_name)
    paths = Ego4DPaths(full_scale_dir=full_scale_dir, sta_json_path=sta_json)

    ds = STANounDetectionDataset(
        paths=paths,
        transform_query=tf,
        min_box_size=1,
        keep_metadata=True,
    )

    # model
    model = SiglipDETR(
        siglip_name=model_name,
        num_classes=num_nouns,
        num_queries=50,
        train_backbone=False,
    ).to(device)

    # load checkpoint
    ckpt = Path("checkpoints/detr_siglip.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        print("Loaded checkpoint:", ckpt)
    else:
        print("WARNING: checkpoint not found; visualizing random weights:", ckpt)

    model.eval()

    # visualize a few samples
    indices = [0, 1, 2, 3, 4]  # they can be any indices in the dataset (important make sure to have keep_metadata=True in the dataset to get the query_rgb for plotting)

    for idx in indices:
        sample = ds[idx]
        pixel_values = sample["query_pixel_values"].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(pixel_values)

        print(f"\nSample idx={idx} video={sample['video_uid']} frame={sample['frame']}")
        print("GT objects:", len(sample["targets"]["labels"]))

        plot_prediction(sample, outputs, noun_map, threshold=0.01)


if __name__ == "__main__":
    main()
