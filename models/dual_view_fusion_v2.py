import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalGlobalAdaptiveFusionV2(nn.Module):
    """V2 fusion that only uses structural complexity features to infer weights."""

    def __init__(self, hidden_dim, max_nodes=500, temperature=0.5, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.temperature = temperature
        self.complexity_gate = nn.Sequential(
            nn.Linear(4, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )
        self.weight_history = []

    def compute_complexity_features(self, x_ggnn, x_appnp, edge_index, batch=None):
        num_nodes = x_ggnn.size(0)
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=x_ggnn.device)
        batch_size = batch.max().item() + 1
        complexity_features = []
        for b in range(batch_size):
            node_mask = batch == b
            n_nodes = node_mask.sum().item()
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            n_edges = edge_mask.sum().item()
            scale = np.log(n_nodes + 1) / np.log(self.max_nodes + 1)
            max_edges = n_nodes * (n_nodes - 1) + 1e-8
            density = n_edges / max_edges
            avg_degree = n_edges / (n_nodes + 1e-8)
            avg_degree_normalized = min(avg_degree / 10.0, 1.0)
            pooled_ggnn = x_ggnn[node_mask].mean(dim=0)
            pooled_appnp = x_appnp[node_mask].mean(dim=0)
            cos_sim = F.cosine_similarity(pooled_ggnn.unsqueeze(0), pooled_appnp.unsqueeze(0))
            divergence = (1 - cos_sim.item()) / 2
            complexity_features.append([scale, density, avg_degree_normalized, divergence])
        return torch.tensor(complexity_features, dtype=torch.float32, device=x_ggnn.device)

    def forward(self, x_ggnn, x_appnp, edge_index, batch=None):
        num_nodes = x_ggnn.size(0)
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=x_ggnn.device)
        complexity_features = self.compute_complexity_features(x_ggnn, x_appnp, edge_index, batch)
        raw_weights = self.complexity_gate(complexity_features)
        weights = F.softmax(raw_weights / self.temperature, dim=1)
        w_local = weights[:, 0]
        w_global = weights[:, 1]
        x_fused = w_local[batch].unsqueeze(1) * x_ggnn + w_global[batch].unsqueeze(1) * x_appnp
        metadata = {
            "w_local": w_local.detach(),
            "w_global": w_global.detach(),
            "complexity_features": complexity_features.detach(),
            "weight_std": weights.std(dim=1).mean().item(),
        }
        if self.training:
            self.weight_history.append({"w_local_mean": w_local.mean().item(), "w_global_mean": w_global.mean().item()})
        return x_fused, metadata

    def get_weight_statistics(self):
        if not self.weight_history:
            return None
        w_local_means = [item["w_local_mean"] for item in self.weight_history]
        w_global_means = [item["w_global_mean"] for item in self.weight_history]
        return {
            "w_local_mean": np.mean(w_local_means),
            "w_local_std": np.std(w_local_means),
            "w_global_mean": np.mean(w_global_means),
            "w_global_std": np.std(w_global_means),
            "total_samples": len(self.weight_history),
        }


class ExplicitComplexityFusion(nn.Module):
    """Formula-based local/global fusion for interpretability ablations."""

    def __init__(self, hidden_dim, max_nodes=500):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.alpha = nn.Parameter(torch.tensor(2.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_ggnn, x_appnp, edge_index, batch=None):
        num_nodes = x_ggnn.size(0)
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=x_ggnn.device)
        batch_size = batch.max().item() + 1
        w_global_list = []
        for b in range(batch_size):
            node_mask = batch == b
            n_nodes = node_mask.sum().item()
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            n_edges = edge_mask.sum().item()
            scale = np.log(n_nodes + 1) / np.log(self.max_nodes + 1)
            density = n_edges / (n_nodes * (n_nodes - 1) + 1e-8)
            logit = self.alpha * scale + self.beta * density + self.gamma
            w_global_list.append(torch.sigmoid(logit))
        w_global = torch.stack(w_global_list)
        w_local = 1 - w_global
        x_fused = w_local[batch].unsqueeze(1) * x_ggnn + w_global[batch].unsqueeze(1) * x_appnp
        metadata = {"w_local": w_local.detach(), "w_global": w_global.detach()}
        return x_fused, metadata
