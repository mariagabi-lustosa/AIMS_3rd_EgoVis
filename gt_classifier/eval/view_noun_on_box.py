import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from sta.gt_classifier.read_data.use_dataset import Ego4DPaths, STANounCropDataset
from sta.gt_classifier.read_data.transform import SiglipTransform
from sta.gt_classifier.models.siglip_noun import SiglipNounClassifier
from sta.gt_classifier.read_data.video_reader import read_frame


def draw_gt_box_with_pred(frame_rgb, box, gt_name, pred_name, score):
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)

    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], width=3)

    text = f"GT: {gt_name}\nPRED: {pred_name} ({score:.2f})"
    draw.text((x1, max(0, y1 - 28)), text)

    return img


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # paths
    sta_json = Path("/home/gabi/Documents/materiais_ic/ViT/ego4d/annotations/fho_sta_train.json")
    full_scale_dir = Path("/home/gabi/Documents/materiais_ic/ViT/ego4d/full_scale")
    ckpt_path = Path("/home/gabi/Documents/materiais_ic/ViT/sta/train/checkpoints/siglip_noun_head.pt")
    out_dir = Path("sta_pred_view")
    out_dir.mkdir(exist_ok=True)

    # load label names  
    data = json.load(open(sta_json, "r"))
    noun_id_to_name = {c["id"]: c["name"] for c in data["noun_categories"]}
    num_nouns = len(noun_id_to_name)

    # dataset (IMPORTANT: keep_metadata=True)  
    model_name = "google/siglip-base-patch16-224"
    transform = SiglipTransform(model_name)

    paths = Ego4DPaths(full_scale_dir=full_scale_dir, sta_json_path=sta_json)
    ds = STANounCropDataset(paths, transform=transform, keep_metadata=True)

    # model  
    model = SiglipNounClassifier(model_name=model_name, num_nouns=num_nouns, train_backbone=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    # visualize a few random-ish samples
    indices = list(range(300, min(350, len(ds))))

    for i, idx in enumerate(indices):
        item = ds[idx]
        pixel_values = item["pixel_values"].unsqueeze(0).to(device)  # [1,3,224,224]
        gt_id = int(item["label"])

        logits = model(pixel_values)
        probs = F.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())
        score = float(probs[pred_id].item())

        gt_name = noun_id_to_name.get(gt_id, str(gt_id))
        pred_name = noun_id_to_name.get(pred_id, str(pred_id))

        # load the *full frame* again for drawing
        video_uid = item["video_uid"]  # or "video_id" depending on your dataset output
        frame_idx = int(item["frame"])
        box = item["box"]  # already scaled + int in your dataset code

        video_path = full_scale_dir / f"{video_uid}.mp4"
        frame_rgb = read_frame(video_path, frame_idx)

        vis = draw_gt_box_with_pred(frame_rgb, box, gt_name, pred_name, score)

        # save with correctness in filename
        ok = (pred_id == gt_id)
        vis.save(out_dir / f"{i:03d}_idx{idx}_ok{int(ok)}_{video_uid}_{frame_idx}.jpg")
        print("saved", out_dir / f"{i:03d}_idx{idx}_ok{int(ok)}_{video_uid}_{frame_idx}.jpg")


if __name__ == "__main__":
    main()