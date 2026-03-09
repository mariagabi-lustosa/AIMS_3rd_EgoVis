import json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    log_path = Path("runs/train_log.jsonl")
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]

    steps = [r["step"] for r in rows]

    for key in ["loss_total", "loss_ce", "loss_bbox", "loss_giou"]:
        if key in rows[0]:
            vals = [r[key] for r in rows]
            plt.figure()
            plt.plot(steps, vals)
            plt.title(key)
            plt.xlabel("step")
            plt.ylabel(key)
            plt.show()

if __name__ == "__main__":
    main()