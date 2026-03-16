import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDualStreamFusion(nn.Module):
    """Fuse semantic and structural query streams with learned gates."""

    def __init__(self, feature_dim, use_interaction=True, temperature=1.0, dropout=0.1):
        super().__init__()
        self.use_interaction = use_interaction
        self.temperature = temperature
        self.num_branches = 3 if use_interaction else 2
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, self.num_branches),
        )
        if use_interaction:
            self.interaction_net = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, feature_dim),
            )

    def forward(self, queries_sem, queries_struct):
        concat = torch.cat([queries_sem, queries_struct], dim=-1)
        gate_logits = self.gate_net(concat)
        gate_weights = F.softmax(gate_logits / self.temperature, dim=-1)
        if self.use_interaction:
            interaction = self.interaction_net(concat)
            fused = (
                gate_weights[..., 0:1] * queries_sem
                + gate_weights[..., 1:2] * queries_struct
                + gate_weights[..., 2:3] * interaction
            )
        else:
            fused = gate_weights[..., 0:1] * queries_sem + gate_weights[..., 1:2] * queries_struct
        metadata = {
            "gate_weights": gate_weights.detach(),
            "sem_weight_mean": gate_weights[..., 0].mean().item(),
            "struct_weight_mean": gate_weights[..., 1].mean().item(),
        }
        return fused, metadata


class GatedVulnQFormer(nn.Module):
    """Lightweight query-based semantic-structural fusion module."""

    def __init__(self, feature_dim=768, num_queries=8, num_heads=8, dropout=0.1, use_interaction=True, use_importance_weighting=True):
        super().__init__()
        self.num_queries = num_queries
        self.feature_dim = feature_dim
        self.use_importance_weighting = use_importance_weighting
        self.query_generator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_queries * feature_dim),
        )
        self.sem_cross_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm_sem = nn.LayerNorm(feature_dim)
        self.struct_cross_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm_struct = nn.LayerNorm(feature_dim)
        self.gated_fusion = GatedDualStreamFusion(feature_dim, use_interaction=use_interaction, temperature=1.0, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm_self = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.norm_ffn = nn.LayerNorm(feature_dim)
        if use_importance_weighting:
            self.query_importance = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.GELU(),
                nn.Linear(feature_dim // 4, 1),
            )

    def _convert_mask(self, mask):
        if mask is None:
            return None
        return mask == 0

    def forward(self, semantic_seq, graph_nodes, semantic_cls, semantic_mask=None, graph_mask=None):
        batch_size = semantic_cls.size(0)
        semantic_mask_mha = self._convert_mask(semantic_mask)
        graph_mask_mha = self._convert_mask(graph_mask)
        queries = self.query_generator(semantic_cls).view(batch_size, self.num_queries, self.feature_dim)
        q_sem_out, _ = self.sem_cross_attn(queries, semantic_seq, semantic_seq, key_padding_mask=semantic_mask_mha)
        queries_sem = self.norm_sem(queries + q_sem_out)
        q_struct_out, _ = self.struct_cross_attn(queries, graph_nodes, graph_nodes, key_padding_mask=graph_mask_mha)
        queries_struct = self.norm_struct(queries + q_struct_out)
        fused_queries, _ = self.gated_fusion(queries_sem, queries_struct)
        q_final, _ = self.self_attn(fused_queries, fused_queries, fused_queries)
        fused_queries = self.norm_self(fused_queries + q_final)
        fused_queries = self.norm_ffn(fused_queries + self.ffn(fused_queries))
        if self.use_importance_weighting:
            importance_scores = self.query_importance(fused_queries)
            importance_weights = F.softmax(importance_scores, dim=1)
            doc_embedding = (fused_queries * importance_weights).sum(dim=1)
        else:
            doc_embedding = fused_queries.mean(dim=1)
        return doc_embedding, fused_queries


class VulnClassifierHead(nn.Module):
    """MLP classifier applied on top of the Q-Former document embedding."""

    def __init__(self, input_dim=768, hidden_dims=None, num_classes=2, dropout=0.2):
        super().__init__()
        hidden_dims = hidden_dims or [256, 64]
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.classifier(x)


def batch_to_padded(x, batch, max_nodes=None):
    batch_size = batch.max().item() + 1
    if max_nodes is None:
        max_nodes = max((batch == i).sum().item() for i in range(batch_size))
    padded = torch.zeros(batch_size, max_nodes, x.size(-1), device=x.device, dtype=x.dtype)
    mask = torch.zeros(batch_size, max_nodes, device=x.device)
    for i in range(batch_size):
        node_idx = batch == i
        num_nodes = node_idx.sum().item()
        actual_nodes = min(num_nodes, max_nodes)
        padded[i, :actual_nodes] = x[node_idx][:actual_nodes]
        mask[i, :actual_nodes] = 1
    return padded, mask
