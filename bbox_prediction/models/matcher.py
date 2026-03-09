import torch
from scipy.optimize import linear_sum_assignment


def define_boxes(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(boxes_1, boxes_2):
    eps = 1e-7

    x11, y11, x12, y12 = boxes_1[:, 0], boxes_1[:, 1], boxes_1[:, 2], boxes_1[:, 3]
    x21, y21, x22, y22 = boxes_2[:, 0], boxes_2[:, 1], boxes_2[:, 2], boxes_2[:, 3]

    xi1 = torch.max(x11[:, None], x21[None, :])
    yi1 = torch.max(y11[:, None], y21[None, :])
    xi2 = torch.min(x12[:, None], x22[None, :])
    yi2 = torch.min(y12[:, None], y22[None, :])

    intersection = (xi2 - xi1).clamp(min=0) * (yi2 - yi1).clamp(min=0)

    area_1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area_2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)

    union = area_1[:, None] + area_2[None, :] - intersection + eps

    iou = intersection / union
    
    xc1 = torch.min(x11[:, None], x21[None, :])
    yc1 = torch.min(y11[:, None], y21[None, :])
    xc2 = torch.max(x12[:, None], x22[None, :])
    yc2 = torch.max(y12[:, None], y22[None, :])

    area_c = (xc2 - xc1).clamp(min=0) * (yc2 - yc1).clamp(min=0) + eps
    g_iou = iou - (area_c - union) / area_c

    return g_iou


class HungarianMatcher(torch.nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou


    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        #print("Available output keys:", outputs.keys())
        out_prob = outputs["pred_logits"].softmax(-1)
        out_bbox = outputs["pred_boxes"]

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]

            if tgt_bbox.numel() == 0:
                #this means there are no objects in the image, so we return empty indices
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            cost_class = -out_prob[b][:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)

            out_box = define_boxes(out_bbox[b])
            tgt_box = define_boxes(tgt_bbox)
            cost_giou = -box_iou(out_box, tgt_box)

            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.cpu()
            i, j = linear_sum_assignment(C)
            indices.append((torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)))

            return indices