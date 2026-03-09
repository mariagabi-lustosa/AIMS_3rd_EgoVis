import torch
from transformers import AutoProcessor

class SiglipTransform:
    def __init__(self, model_name):
        self.processor = AutoProcessor.from_pretrained(model_name)

    def __call__(self, pil_img):
        # returns dict with pixel_values; we want just the tensor
        inputs = self.processor(images=pil_img, return_tensors="pt")
        return inputs["pixel_values"][0]  # remove batch dim
