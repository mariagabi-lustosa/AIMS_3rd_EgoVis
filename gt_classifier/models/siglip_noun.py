import torch
import torch.nn as nn
from transformers import AutoModel

class SiglipNounClassifier(nn.Module):
    def __init__(self, model_name: str, num_nouns: int, train_backbone: bool = False):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

        # Freeze backbone if requested
        if not train_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

        # Infer embedding dim from config (works for SigLIP)
        cfg = self.model.config
        emb_dim = cfg.vision_config.hidden_size
        self.classifier = nn.Linear(emb_dim, num_nouns)

    def forward(self, pixel_values):
        # pixel_values: [B, 3, H, W]

        # SiglipModel returns structured outputs; use vision_model directly
        if hasattr(self.model, "vision_model"):
            out = self.model.vision_model(pixel_values=pixel_values)
            # out is BaseModelOutputWithPooling -> tensor is pooler_output
            feats = out.pooler_output  # [B, D]
        else:
            # fallback (shouldn't happen for SiglipModel)
            out = self.model(pixel_values=pixel_values)
            feats = out.pooler_output if hasattr(out, "pooler_output") else out.last_hidden_state[:, 0]

        if not isinstance(feats, torch.Tensor):
            raise TypeError(f"Expected tensor features, got: {type(feats)}")

        return self.classifier(feats)
