import torch
import torch.nn as nn
from sta.bbox_prediction.models.siglip_query import SiglipQueryEncoder


class SiglipCosmosDETR(nn.Module):
    def __init__(
            self,
            siglip_name: str,
            num_classes: int,
            num_queries: int = 50,
            d_model: int = 256,
            cosmos_dim: int = 3,
            train_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = SiglipQueryEncoder(siglip_name, train_backbone=train_backbone)
        siglip_dim = self.backbone.model.config.vision_config.hidden_size

        self.input_proj = nn.Linear(siglip_dim, d_model)
        self.cosmos_proj = nn.Linear(cosmos_dim, d_model)

        decode_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decode_layer, num_layers=6)

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_embed = nn.Linear(d_model, num_classes + 1) # no object case
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
        )


    def forward(self, pixel_values, cosmos_feats):
        tokens, _ = self.backbone(pixel_values)
        siglip_mem = self.input_proj(tokens)
        cosmos_mem = self.cosmos_proj(cosmos_feats)
        memory = torch.cat([siglip_mem, cosmos_mem], dim=1)

        B = memory.size(0)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        hs = self.decoder(tgt=queries, memory=memory)
        logits = self.class_embed(hs)
        bboxes = self.bbox_embed(hs).sigmoid() # normalize bbox coordinates to [0, 1]

        return {"pred_logits": logits, "pred_bboxes": bboxes}