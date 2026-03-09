import torch
from transformers import AutoProcessor

class SiglipFrameTransform:
    def __init__(self, model_name):
        self.processor = AutoProcessor.from_pretrained(model_name)

    def __call__(self, frame_rgb_np):
        # returns dict with pixel_values; we want just the tensor
        inputs = self.processor(images=frame_rgb_np.copy(), return_tensors="pt")
        return inputs["pixel_values"][0]  # remove batch dim
