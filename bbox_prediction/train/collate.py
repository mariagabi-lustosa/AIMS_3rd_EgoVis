import torch

def collate(batch):
    pixel_values = torch.stack([b["query_pixel_values"] for b in batch], dim=0)
    
    # Handle None cosmos_feats
    if batch[0]["cosmos_feats"] is not None:
        cosmos_feats = torch.stack([b["cosmos_feats"] for b in batch], 0)  # [B,6,3]
    else:
        cosmos_feats = None
    
    targets = [b["targets"] for b in batch]
    
    return pixel_values, cosmos_feats, targets