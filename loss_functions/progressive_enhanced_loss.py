import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressiveEnhancedLoss(nn.Module):
    """Transition from weighted CE to focal loss as training progresses."""

    def __init__(
        self,
        num_classes=2,
        class_weights=None,
        focal_alpha=None,
        focal_gamma=2.0,
        label_smoothing=0.1,
        warmup_epochs=10,
        total_epochs=50,
    ):
        super().__init__()
        self.num_classes = num_classes
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
        warmup = min(self.warmup_epochs, self.total_epochs // 3)
        if self.current_epoch <= warmup:
            ce_weight, fl_weight = 1.0, 0.0
        else:
            progress = (self.current_epoch - warmup) / max(1, self.total_epochs - warmup)
            ce_weight = max(0.4, 1.0 - progress)
            fl_weight = 1.0 - ce_weight

        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )

        ce_loss_unreduced = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss_unreduced)
        if self.focal_alpha is not None:
            alpha_t = self.focal_alpha if isinstance(self.focal_alpha, (float, int)) else self.focal_alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * ce_loss_unreduced
        else:
            focal_loss = (1 - pt) ** self.focal_gamma * ce_loss_unreduced

        return ce_weight * ce_loss + fl_weight * focal_loss.mean()

    def get_loss_weights(self):
        warmup = min(self.warmup_epochs, self.total_epochs // 3)
        if self.current_epoch <= warmup:
            ce_weight, fl_weight = 1.0, 0.0
        else:
            progress = (self.current_epoch - warmup) / max(1, self.total_epochs - warmup)
            ce_weight = max(0.4, 1.0 - progress)
            fl_weight = 1.0 - ce_weight
        return {"w_ce": ce_weight, "w_fl": fl_weight}
