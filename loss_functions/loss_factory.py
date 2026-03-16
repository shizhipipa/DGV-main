import torch

from .dynamic_weighted_loss import DynamicWeightedLoss
from .focal_loss import FocalLoss
from .progressive_enhanced_loss import ProgressiveEnhancedLoss
from .smooth_transition_loss import SmoothTransitionLoss
from .weighted_ce_loss import WeightedCELoss


def create_loss_function(loss_strategy, device, total_epochs=50, **kwargs):
    class_weight_false = kwargs.get("class_weight_false", 1.2)
    class_weight_true = kwargs.get("class_weight_true", 0.8)
    focal_gamma = kwargs.get("focal_gamma", 2.5)
    label_smoothing = kwargs.get("label_smoothing", 0.1)
    warmup_epochs = kwargs.get("warmup_epochs", min(15, total_epochs // 3))
    disable_loss_balance = kwargs.get("disable_loss_balance", False)

    ce_weights = None if disable_loss_balance else torch.tensor([class_weight_false, class_weight_true], device=device)
    focal_weights = None if disable_loss_balance else torch.tensor([0.8, 0.2], device=device)
    if disable_loss_balance:
        label_smoothing = 0.0

    if loss_strategy == "crossentropy":
        if ce_weights is None:
            print("Created cross-entropy loss without class weighting.")
        else:
            print(f"Created weighted cross-entropy loss with weights={ce_weights.tolist()} and label_smoothing={label_smoothing}.")
        return WeightedCELoss(class_weights=ce_weights, label_smoothing=label_smoothing, reduction="mean")

    if loss_strategy == "focal":
        if focal_weights is None:
            print(f"Created focal loss without alpha weighting. gamma={focal_gamma}")
        else:
            print(f"Created focal loss with alpha={focal_weights.tolist()} and gamma={focal_gamma}.")
        return FocalLoss(alpha=focal_weights, gamma=focal_gamma, reduction="mean")

    if loss_strategy == "progressive":
        print(f"Created progressive enhanced loss. warmup_epochs={warmup_epochs}, total_epochs={total_epochs}")
        return ProgressiveEnhancedLoss(
            num_classes=2,
            class_weights=ce_weights,
            focal_alpha=focal_weights,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
        )

    if loss_strategy == "smooth_transition":
        print(f"Created smooth transition loss. warmup_epochs={warmup_epochs}, total_epochs={total_epochs}")
        return SmoothTransitionLoss(
            class_weights=ce_weights,
            focal_alpha=focal_weights,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
        )

    if loss_strategy == "dynamic_weighted":
        print(f"Created dynamic weighted loss. warmup_epochs={warmup_epochs}, total_epochs={total_epochs}")
        return DynamicWeightedLoss(
            class_weights=ce_weights,
            focal_alpha=focal_weights,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
        )

    raise ValueError(
        "Unsupported loss strategy. Supported values are: "
        "crossentropy, focal, progressive, smooth_transition, dynamic_weighted."
    )


def get_loss_info(loss_function):
    if hasattr(loss_function, "get_loss_info"):
        return loss_function.get_loss_info()
    if hasattr(loss_function, "get_loss_weights"):
        return loss_function.get_loss_weights()
    return {"type": type(loss_function).__name__}
