import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothTransitionLoss(nn.Module):
    """Smoothly interpolate between CE and focal loss over epochs."""

    def __init__(
        self,
        class_weights=None,
        focal_alpha=None,
        focal_gamma=2.5,
        label_smoothing=0.1,
        warmup_epochs=15,
        total_epochs=50,
    ):
        super().__init__()
        self.class_weights = class_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def update_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, inputs, targets):
        transition = min(1.0, max(0.0, self.current_epoch / max(1, self.total_epochs)))
        ce_weight = max(0.2, 1.0 - transition)
        fl_weight = 1.0 - ce_weight

        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )
        ce_unreduced = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_unreduced)
        if self.focal_alpha is not None:
            alpha_t = self.focal_alpha if isinstance(self.focal_alpha, (float, int)) else self.focal_alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * ce_unreduced
        else:
            focal_loss = (1 - pt) ** self.focal_gamma * ce_unreduced

        return ce_weight * ce_loss + fl_weight * focal_loss.mean()

    def get_loss_info(self):
        transition = min(1.0, max(0.0, self.current_epoch / max(1, self.total_epochs)))
        ce_weight = max(0.2, 1.0 - transition)
        return {"epoch": self.current_epoch, "ce_weight": ce_weight, "fl_weight": 1.0 - ce_weight}
