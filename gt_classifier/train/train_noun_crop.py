import torch
import os
import json

import torch.nn.functional as F

from tqdm import tqdm
from xml.parsers.expat import model
from torch.utils.data import DataLoader
from sta.bbox_prediction.read_data.filter_and_use_dataset import Ego4DPaths, STANounCropDataset
from sta.gt_classifier.read_data.transform import SiglipTransform
from sta.gt_classifier.models.siglip_noun import SiglipNounClassifier


def collate(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return pixel_values, labels


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "google/siglip-base-patch16-224"  # choose what you used
    sta_json = "/home/gabi/Documents/materiais_ic/ViT/ego4d/annotations/fho_sta_train.json"
    full_scale_dir = "/home/gabi/Documents/materiais_ic/ViT/ego4d/full_scale"

    paths = Ego4DPaths(full_scale_dir=full_scale_dir, sta_json_path=sta_json)
    
    data = json.load(open(sta_json, "r"))
    num_nouns = len(data["noun_categories"])
    #print(num_nouns)

    transform = SiglipTransform(model_name)
    ds = STANounCropDataset(paths, transform=transform)

    batch_size = 8
    num_workers = min(4, os.cpu_count())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)

    model = SiglipNounClassifier(model_name=model_name, num_nouns=num_nouns, train_backbone=False)
    model.to(device)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print("Trainable params:", trainable[:20], "count:", len(trainable))

    params = [p for p in model.parameters() if p.requires_grad]

    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)

    model.train()
    print("Has get_image_features:", hasattr(model.model, "get_image_features"))
    print("Model class:", type(model.model))

    for epoch in range(3):
        total, correct, total_loss = 0, 0, 0.0
        for pixel_values, labels in tqdm(dl, desc=f"epoch {epoch}"):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            logits = model(pixel_values)
            loss = F.cross_entropy(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch}: loss={total_loss/total:.4f} acc={correct/total:.4f}")
    
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/siglip_noun_head.pt")

if __name__ == "__main__":
    main()
