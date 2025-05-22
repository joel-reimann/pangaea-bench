import operator
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt  # [WANDB_IMG_LOGGING] for quick GT/pred visualization
import wandb  # [WANDB_IMG_LOGGING] for quick GT/pred visualization
from pangaea.engine.trainer import RegTrainer
from pangaea.engine.evaluator import RegEvaluator

class AGBDRegTrainer(RegTrainer):
    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Spatially average logits before MSE
        logits = torch.mean(logits, dim=(-2, -1))
        return self.criterion(logits, target)

    @torch.no_grad()
    def compute_logging_metrics(self, logits: torch.Tensor, target: torch.Tensor):
        logits = torch.mean(logits, dim=(-2, -1))
        mse = F.mse_loss(logits, target, reduction="mean").item()
        return {"MSE": mse}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_metric_comp = operator.lt  # For MSE minimization

class AGBDRegEvaluator(RegEvaluator):
    @torch.no_grad()
    def evaluate(self, model, model_name='model', model_ckpt_path=None):
        self.model = model
        self.model.eval()
        mse_sum = 0.0
        n_samples = 0
        # [WANDB_IMG_LOGGING] Begin block
        images_to_log = []
        logged_images = False
        # [WANDB_IMG_LOGGING] End block
        for i, batch in enumerate(self.val_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            logits = self.model(images)
            logits = torch.mean(logits, dim=(-2, -1))
            mse_sum += F.mse_loss(logits, labels, reduction="sum").item()
            n_samples += labels.size(0)
            # [WANDB_IMG_LOGGING] Log GT/pred for first batch only
            if not logged_images:
                for j in range(min(4, images.size(0))):
                    img = images[j].detach().cpu().numpy()
                    # Prithvi expects S2 bands: B02, B03, B04, B8A, B11, B12
                    # Use B04 (red), B03 (green), B02 (blue) for RGB
                    rgb = img[[2, 1, 0], :, :]
                    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
                    gt = labels[j].cpu().item()
                    pred = logits[j].cpu().item()
                    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
                    ax.imshow(rgb.transpose(1, 2, 0))
                    ax.set_title(f"GT: {gt:.1f}, Pred: {pred:.1f}")
                    ax.axis('off')
                    plt.tight_layout()
                    images_to_log.append(wandb.Image(fig, caption=f"GT: {gt:.1f}, Pred: {pred:.1f}"))
                    plt.close(fig)
                if wandb.run is not None and images_to_log:
                    wandb.log({"examples": images_to_log})
                logged_images = True
            # [WANDB_IMG_LOGGING] End block
        final_mse = mse_sum / n_samples if n_samples > 0 else float('nan')
        metrics = {"MSE": final_mse}
        self.log_metrics(metrics)
        return metrics