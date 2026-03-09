import matplotlib.pyplot as plt
import matplotlib.patches as patches


def convert_coords(box):
    cx, cy, w, h = box
    return [
        cx - w / 2,
        cy - h / 2,
        cx + w / 2,
        cy + h / 2,
    ]


def plot_prediction(sample, outputs, noun_map, threshold=0.01):
    img = sample["query_rgb"]
    h, w = img.shape[:2]

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(img)
    ax.axis("off")

    gt_boxes = sample["targets"]["boxes"]
    gt_labels = sample["targets"]["labels"]

    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = convert_coords(box.tolist())
        x1 *= w
        x2 *= w
        y1 *= h
        y2 *= h

        rect = patches.Rectangle(
            (x1, y1), 
            x2-x1, 
            y2-y1,
            linewidth=2,
            edgecolor="green",
            facecolor="none",
        )
        ax.add_patch(rect)
        
        ax.text(
            x1,
            max(0, y1 - 5),
            noun_map.get(int(label), str(int(label))),
            color="green",
            fontsize=10,
            backgroundcolor="black",
        )


    logits = outputs["pred_logits"][0]
    boxes = outputs["pred_boxes"][0]

    scores = logits.softmax(-1)
    conf, labels = scores[..., :-1].max(-1)

    keep = conf > threshold

    boxes = boxes[keep]
    labels = labels[keep]
    conf = conf[keep]

    for box, label, score in zip(boxes, labels, conf):
        x1, y1, x2, y2 = convert_coords(box.tolist())
        x1 *= w
        x2 *= w
        y1 *= h
        y2 *= h

        rect = patches.Rectangle(
            (x1, y1), 
            x2-x1, 
            y2-y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            max(0, y1 - 5),
            f"{noun_map.get(int(label), str(int(label)))}: {score:.2f}",
            color="red",
            fontsize=10,
            backgroundcolor="black",
        )

    plt.show()