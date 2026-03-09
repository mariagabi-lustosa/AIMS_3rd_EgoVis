import json
import torch
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from PIL import Image

from ...gt_classifier.read_data.video_reader import read_frame


@dataclass
class Ego4DPaths:
    full_scale_dir: Path
    sta_json_path: Path


def get_available_uids(full_scale_dir):
    return {p.stem for p in Path(full_scale_dir).glob("*.mp4")}


def filter_annotations_by_available_videos(annotations, available_uids):
    filtered = []
    for a in annotations:
        video = a.get("video_uid") or a.get("video_id")
        if video in available_uids:
            filtered.append(a)
    return filtered


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h-1))
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return [x1, y1, x2, y2]


def resolve_video_path(full_scale_dir, video_uid):
    p = Path(full_scale_dir) / f"{video_uid}.mp4"
    if p.exists():
        return p
    raise FileNotFoundError(f"Missing video: {p}")


def define_box(box):
    x1, y1, x2, y2 = box.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    return torch.stack([cx, cy, w, h], dim=-1)


class STANounDetectionDataset:
    def __init__(
            self,
            paths: Ego4DPaths,
            transform_query = None,
            min_box_size: int = 1,
            keep_metadata: bool = False,
            cosmos_cache_dir: Optional[Path] = None,
    ):
        
        self.paths = paths
        self.keep_metadata = keep_metadata
        self.min_box_size = min_box_size
        self.transform_query = transform_query
        self.cosmos_cache_dir = cosmos_cache_dir


        data = json.load(open(paths.sta_json_path, "r"))
        self.video_metadata = data["info"]["video_metadata"]

        available = get_available_uids(self.paths.full_scale_dir)
        print(f"Available videos: {len(available)}")
        
        annotations = data["annotations"]

        annotations = filter_annotations_by_available_videos(annotations, available)
        print(f"Annotations with local videos: {len(annotations)}")

        self.samples = []
        for a in annotations:
            if "frame" not in a:
                continue

            if not a.get("objects"):
                continue

            self.samples.append(a)

        if len(self.samples) == 0:
            raise RuntimeError("No samples found in the dataset.")
        

        if self.cosmos_cache_dir is not None:
            cache_dir = Path(self.cosmos_cache_dir)
            before = len(self.samples)
            self.samples = [
                a for a in self.samples
                if (cache_dir / f"{a.get('uid')}.pt").exists()
            ]
            after = len(self.samples)
            print(f"Samples with Cosmos cache: {after}/{before}")

            if after == 0:
                raise RuntimeError(f"No samples have Cosmos cache in: {cache_dir}")
        

    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, index):
        a = self.samples[index]
        video_uid = a.get("video_id") or a.get("video_uid")
        video_path = resolve_video_path(self.paths.full_scale_dir, video_uid)

        frame_index = a["frame"]
        frame_rgb = read_frame(video_path, frame_index)

        query_pixel_values = self.transform_query(frame_rgb) if self.transform_query else None

        metadata = self.video_metadata[video_uid]
        a_w, a_h = metadata["frame_width"], metadata["frame_height"]

        h, w = frame_rgb.shape[0], frame_rgb.shape[1]
        sx, sy = w / a_w, h / a_h

        uid = a.get("uid")
        cosmos_feats = None
        if self.cosmos_cache_dir is not None:
            p = Path(self.cosmos_cache_dir) / f"{uid}.pt"
            if not p.exists():
                raise FileNotFoundError(f"Missing Cosmos cache for annotation uid={uid}: {p}")
            cosmos_feats = torch.load(p)  # expected [6,3] float32

        boxes = []
        nouns = []
        for obj in a["objects"]:
            x1, y1, x2, y2 = obj["box"]
            x1, x2 = x1 * sx, x2 * sx
            y1, y2 = y1 * sy, y2 * sy
            x1, y1, x2, y2 = map(int, clamp_box([x1, y1, x2, y2], w, h))
            if (x2-x1) < self.min_box_size or (y2-y1) < self.min_box_size:
                continue
            
            boxes.append([x1, y1, x2, y2])
            nouns.append(int(obj["noun_category_id"]))
        
        h, w = frame_rgb.shape[:2]
        boxes = torch.tensor(boxes, dtype=torch.float32)
        boxes[:, [0, 2]] /= float(w)
        boxes[:, [1, 3]] /= float(h)
        boxes = define_box(boxes)

        labels = torch.tensor(nouns, dtype=torch.long)

        output = {
            "query_pixel_values": query_pixel_values,
            "cosmos_feats": cosmos_feats,
            "targets": {
                "labels": labels,
                "boxes": boxes,
            }
        }

        if self.keep_metadata:
            output |= {
                "query_rgb": frame_rgb,
                "video_uid": video_uid,
                "frame": frame_index,
                "annotation_uid": a.get("uid"),
            }
        
        return output