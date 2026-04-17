import wandb


class WandbWriter:
    def __init__(self):
        self._defined_metrics = set()

    def _step_metric_for_tag(self, tag):
        if "/epoch/" in tag or "_per_epoch" in tag:
            return "epoch"
        return "iter"

    def _ensure_metric_defined(self, tag):
        step_metric = self._step_metric_for_tag(tag)
        if tag in self._defined_metrics:
            return step_metric
        wandb.define_metric(tag, step_metric=step_metric)
        self._defined_metrics.add(tag)
        return step_metric

    def add_scalar(self, tag, value, step):
        step_metric = self._ensure_metric_defined(tag)
        wandb.log({step_metric: step, tag: value})

    def add_text(self, tag, text, step):
        step_metric = self._ensure_metric_defined(tag)
        wandb.log({step_metric: step, tag: text})

    def add_video(self, tag, vid_tensor, step, fps=4):
        """
        vid_tensor: (N, T, C, H, W)
        """
        step_metric = self._ensure_metric_defined(tag)
        videos = []

        for i in range(vid_tensor.shape[0]):
            videos.append(
                wandb.Video(
                    vid_tensor[i].cpu().numpy(),
                    fps=fps,
                    format="mp4"
                )
            )

        wandb.log({step_metric: step, tag: videos})
