import torch.nn as nn
import torch.nn.functional as F


class WeightedCELoss(nn.Module):
    """Cross-entropy with optional class weights and label smoothing."""

    def __init__(self, class_weights=None, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        return F.cross_entropy(
            inputs,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )
