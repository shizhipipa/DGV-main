import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicWeightedLoss(nn.Module):
    """Adjust CE/focal weights using training progress and recent validation F1."""

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
        self.f1_history = []

    def update_epoch(self, epoch, f1_score=None):
        self.current_epoch = epoch
        if f1_score is not None:
            self.f1_history.append(f1_score)

    def forward(self, inputs, targets):
        warmup = min(self.warmup_epochs, self.total_epochs // 3)
        ce_weight = max(0.4, 1.0 - max(0, self.current_epoch - warmup) / max(1, self.total_epochs - warmup))
        fl_weight = 1.0 - ce_weight

        if self.f1_history and self.f1_history[-1] < 0.6:
            adjustment = (0.6 - self.f1_history[-1]) / 0.6 * 0.2
            ce_weight = max(0.3, ce_weight - adjustment)
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
        warmup = min(self.warmup_epochs, self.total_epochs // 3)
        ce_weight = max(0.4, 1.0 - max(0, self.current_epoch - warmup) / max(1, self.total_epochs - warmup))
        fl_weight = 1.0 - ce_weight
        adjusted = False
        if self.f1_history and self.f1_history[-1] < 0.6:
            adjustment = (0.6 - self.f1_history[-1]) / 0.6 * 0.2
            ce_weight = max(0.3, ce_weight - adjustment)
            fl_weight = 1.0 - ce_weight
            adjusted = True
        return {"epoch": self.current_epoch, "ce_weight": ce_weight, "fl_weight": fl_weight, "f1_adjusted": adjusted}
