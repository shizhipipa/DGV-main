import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedDynamicFusion(nn.Module):
    """Fuse semantic and structural logits with an interaction-aware gate."""

    def __init__(self, feature_dim, hidden_dim=128, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.interaction_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, feature_dim),
        )
        self.weight_net = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.semantic_transform = nn.Linear(feature_dim, feature_dim)
        self.structural_transform = nn.Linear(feature_dim, feature_dim)

    def forward(self, semantic_feat, structural_feat):
        interaction_feat = self.interaction_net(torch.cat([semantic_feat, structural_feat], dim=1))
        semantic_feat_t = self.semantic_transform(semantic_feat)
        structural_feat_t = self.structural_transform(structural_feat)
        raw_weights = self.weight_net(torch.cat([semantic_feat, structural_feat, interaction_feat], dim=1))
        weights = F.softmax(raw_weights / self.temperature, dim=1)
        fused_feat = (
            weights[:, 0].unsqueeze(1) * semantic_feat_t
            + weights[:, 1].unsqueeze(1) * structural_feat_t
            + weights[:, 2].unsqueeze(1) * interaction_feat
        )
        metadata = {"weights": weights}
        return fused_feat, metadata
