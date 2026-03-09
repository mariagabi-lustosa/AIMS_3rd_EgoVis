import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
from PIL import Image

from ...bbox_prediction.read_data.video_frame_reader import read_frame


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


def resolve_video_path(full_scale_dir, video_uid):
    p = Path(full_scale_dir) / f"{video_uid}.mp4"
    if p.exists():
        return p


def define_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    #x1, x2 = sorted([x1, x2])
    #y1, y2 = sorted([y1, y2])
    return [x1, y1, x2, y2]


class STANounCropDataset:
    def __init__(
            self,
            paths: Ego4DPaths,
            split_max_annotations: Optional[int] = None,
            min_box_size: int = 1,
            transform = None,
            keep_metadata: bool = False,
    ):
        
        self.paths = paths
        self.transform = transform
        self.keep_metadata = keep_metadata
        self.min_box_size = min_box_size

        data = json.load(open(paths.sta_json_path, "r"))
        self.video_metadata = data["info"]["video_metadata"]
        self.noun_categories = data["noun_categories"]

        available = get_available_uids(self.paths.full_scale_dir)
        print(f"Available videos: {len(available)}")
        
        annotations = filter_annotations_by_available_videos(data["annotations"], available)
        print(f"Annotations with local videos: {len(annotations)}")

        self.samples = []
        for a in annotations:
            if "frame" not in a:
                continue

            objects = a.get("objects", [])
            if not objects:
                continue

            for obj_index, obj in enumerate(objects):
                self.samples.append((a, obj_index))
        
        if len(self.samples) == 0:
            raise RuntimeError("No samples found in the dataset.")
        

    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, index):
        a, obj_index = self.samples[index]
        video_uid = a.get("video_id") or a.get("video_uid")
        video_path = resolve_video_path(self.paths.full_scale_dir, video_uid)

        frame_index = a["frame"]
        frame_rgb = read_frame(video_path, frame_index)

        metadata = self.video_metadata[video_uid]
        a_w, a_h = metadata["frame_width"], metadata["frame_height"]

        h, w = frame_rgb.shape[0], frame_rgb.shape[1]
        sx, sy = w / a_w, h / a_h

        obj = a["objects"][obj_index]
        x1, y1, x2, y2 = obj["box"]
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy

        box = define_box([x1, y1, x2, y2], w, h)
        x1, y1, x2, y2 = map(int, box)
        if (x2 - x1) < self.min_box_size or (y2 - y1) < self.min_box_size:
            return self.__getitem__((index + 1) % len(self.samples))
        
        crop = frame_rgb[y1:y2, x1:x2, :]
        crop_image = Image.fromarray(crop)

        label = int(obj["noun_category_id"])

        if self.transform is not None:
            pixel_values = self.transform(crop_image)
        else:
            pixel_values = crop_image # debug

        if self.keep_metadata:
            return {
                "pixel_values": pixel_values,
                "label": label,
                "video_uid": video_uid,
                "annotation_uid": a.get("uid"),
                "frame": frame_index,
                "box": [x1, y1, x2, y2],
            }
        
        return {"pixel_values": pixel_values, "label": label}