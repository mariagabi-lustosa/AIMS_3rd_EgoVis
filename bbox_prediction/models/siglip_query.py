import torch
import torch.nn as nn
from transformers import AutoModel

class SiglipQueryEncoder(nn.Module):
    def __init__(self, model_name: str, train_backbone: bool = False):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

        # Freeze backbone if requested
        if not train_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

    
    def forward(self, query_pixel_vals):
        """
        query_pixel_vals: [B, 3, H, W]
        returns:
            last_hidden_state: [B, N, D]
            pooler_output: [B, D]
        """
        vision_out = self.model.vision_model(pixel_values=query_pixel_vals)
        return vision_out.last_hidden_state, vision_out.pooler_output