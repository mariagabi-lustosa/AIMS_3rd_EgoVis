import torch
import torch.nn.functional as F
from .matcher import HungarianMatcher, box_iou, define_boxes


class SetCriterion(torch.nn.Module):
    def __init__(self, num_classes, matcher: HungarianMatcher, eos_coef=0.1, loss_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.loss_weights = loss_weights or {
            "loss_ce": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        }

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)


    def loss_labels(self, outputs, targets, indices):
        logits = outputs["pred_logits"]
        B, Q, _ = logits.shape

        target_classes = torch.full((B, Q), self.num_classes, dtype=torch.long, device=logits.device)

        for b, (i, j) in enumerate(indices):
            if i.numel() == 0:
                continue
            target_classes[b, i] = targets[b]["labels"][j].to(logits.device)

        loss_ce = F.cross_entropy(logits.transpose(1, 2), target_classes, weight=self.empty_weight)
        return {"loss_ce": loss_ce} # classification loss
                                    # it means that is the cross-entropy loss between predicted noun classes and the ground-truth noun labels (plus the “no-object” class)
    

    def loss_boxes(self, outputs, targets, indices):
        pred_boxes = outputs["pred_boxes"]
        loss_bbox = torch.tensor(0.0, device=pred_boxes.device)
        loss_g_iou = torch.tensor(0.0, device=pred_boxes.device)
        n = 0

        for b, (i, j) in enumerate(indices):
            if i.numel() == 0:
                continue
            pb = pred_boxes[b, i]
            tb = targets[b]["boxes"][j].to(pred_boxes.device)
            loss_bbox += F.l1_loss(pb, tb, reduction="sum")

            pb_coords = define_boxes(pb)
            tb_coords = define_boxes(tb)
            g_iou = box_iou(pb_coords, tb_coords).diag()
            loss_g_iou += (1 - g_iou).sum()

            n += tb.shape[0]
        
        n = max(n, 1)
        return {
            "loss_bbox": loss_bbox / n,
            "loss_giou": loss_g_iou / n,
        }
    

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices))

        total_loss = 0.0
        for k, v in losses.items():
            total_loss += self.loss_weights.get(k, 1.0) * v

        losses["total_loss"] = total_loss

        return losses