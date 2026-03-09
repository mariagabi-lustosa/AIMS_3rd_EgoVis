import torch
import torch.nn as nn
from sta.bbox_prediction.models.siglip_query import SiglipQueryEncoder


class SiglipDETR(nn.Module):
    def __init__(
            self,
            siglip_name: str,
            num_classes: int,
            num_queries: int = 50,
            d_model: int = 256,
            nhead: int = 8,
            num_decoder_layers: int = 6,
            dim_feedforward: int = 1024,
            dropout: float = 0.1,
            train_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = SiglipQueryEncoder(siglip_name, train_backbone)

        siglip_dim = self.backbone.model.config.vision_config.hidden_size

        self.input_proj = nn.Linear(siglip_dim, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
            )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.query_embed = nn.Embedding(num_queries, d_model)

        self.class_embed = nn.Linear(d_model, num_classes + 1) # no object case (maybe remove this later)

        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
            #nn.Sigmoid(), # normalize bbox coordinates to [0, 1]
        )

        self.num_queries = num_queries

    
    def forward(self, pixel_values):
        tokens, _pooled = self.backbone(pixel_values)
        memory = self.input_proj(tokens)

        B = memory.size(0)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        hs = self.decoder(tgt=queries, memory=memory)

        logits = self.class_embed(hs)
        bboxes = self.bbox_embed(hs).sigmoid()

        return {"pred_logits": logits, "pred_boxes": bboxes}