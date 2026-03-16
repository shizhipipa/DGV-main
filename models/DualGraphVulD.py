import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import GatedGraphConv
from transformers import AutoModel, AutoTokenizer

from models.appnp_conv import APPNPConv
from models.dual_view_fusion import LocalGlobalAdaptiveFusion
from models.dual_view_fusion_v2 import ExplicitComplexityFusion, LocalGlobalAdaptiveFusionV2
from models.improved_fusion import ImprovedDynamicFusion
from models.layers import Conv
from models.vuln_qformer import GatedVulnQFormer, VulnClassifierHead, batch_to_padded


class DualGraphVulD(nn.Module):
    """Core dual-view vulnerability detector used in the paper."""

    def __init__(
        self,
        pred_lambda,
        gated_graph_conv_args,
        conv_args,
        emb_size,
        device,
        fusion_type="dynamic",
        appnp_args=None,
        use_residual=True,
        dropout=0.2,
        model_path="models/unixcoder",
        dual_fusion_type="v1",
        use_qformer=False,
        qformer_num_queries=32,
        qformer_num_heads=8,
    ):
        super().__init__()
        self.k = pred_lambda
        self.device = device
        self.dropout_rate = dropout
        self.nb_class = 2
        self.use_residual = use_residual
        self.model_path = model_path
        self.fusion_type = fusion_type
        self.use_qformer = use_qformer

        self.local_encoder = GatedGraphConv(**gated_graph_conv_args).to(device)
        if appnp_args is None:
            appnp_args = {"K": 10, "alpha": 0.1, "dropout": dropout}
        self.global_encoder = APPNPConv(**appnp_args).to(device)

        hidden_dim = gated_graph_conv_args["out_channels"]
        self.dual_fusion_type = dual_fusion_type
        if dual_fusion_type == "v2":
            self.dual_view_fusion = LocalGlobalAdaptiveFusionV2(hidden_dim=hidden_dim, max_nodes=500, temperature=0.5, dropout=dropout).to(device)
            print("Using dual-view fusion: V2 complexity gating.")
        elif dual_fusion_type == "explicit":
            self.dual_view_fusion = ExplicitComplexityFusion(hidden_dim=hidden_dim, max_nodes=500).to(device)
            print("Using dual-view fusion: explicit formula mode.")
        else:
            self.dual_view_fusion = LocalGlobalAdaptiveFusion(hidden_dim=hidden_dim, max_nodes=500, temperature=1.0, dropout=dropout).to(device)
            print("Using dual-view fusion: V1 learned gating.")

        self.conv = Conv(
            **conv_args,
            fc_1_size=gated_graph_conv_args["out_channels"] + emb_size,
            fc_2_size=gated_graph_conv_args["out_channels"],
        ).to(device)

        self.tokenizer, self.code_lm, self.model_path = self._load_codelm(model_path, device)
        self.feat_dim = self.code_lm.config.hidden_size
        self.semantic_classifier = nn.Linear(self.feat_dim, self.nb_class).to(device)
        self.semantic_node_init = nn.Linear(self.feat_dim + 1, emb_size).to(device)

        self.struct_to_logits = nn.Sequential(
            nn.Linear(1, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout * 0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, self.nb_class),
        ).to(device)
        self.semantic_struct_fusion = ImprovedDynamicFusion(feature_dim=self.nb_class, hidden_dim=128, temperature=1.0).to(device)

        if self.use_qformer:
            print("Using Q-Former fusion path.")
            self.graph_proj = nn.Linear(hidden_dim, self.feat_dim).to(device)
            self.qformer = GatedVulnQFormer(
                feature_dim=self.feat_dim,
                num_queries=qformer_num_queries,
                num_heads=qformer_num_heads,
                dropout=dropout,
                use_interaction=True,
                use_importance_weighting=True,
            ).to(device)
            self.qformer_classifier = VulnClassifierHead(
                input_dim=self.feat_dim,
                hidden_dims=[256, 64],
                num_classes=self.nb_class,
                dropout=dropout,
            ).to(device)

        self.dropout = nn.Dropout(dropout)
        self.bn_struct = nn.BatchNorm1d(hidden_dim).to(device)
        self.bn_semantic = nn.BatchNorm1d(self.feat_dim).to(device)
        self.bn_struct_output = nn.BatchNorm1d(1).to(device)
        self.pos_bias_logit = nn.Parameter(torch.tensor([-0.15]), requires_grad=False)
        self.bias_history_f1 = []
        self.bias_adjustment_cool_down = 0
        self.precision_history = []
        self.apply(self._init_weights)
        self.aux_struct_head = nn.Linear(hidden_dim, self.nb_class).to(device)

    def _load_codelm(self, model_path, device):
        candidate_paths = [model_path, "models/unixcoder", "microsoft/unixcoder-base"]
        last_error = None
        for candidate in candidate_paths:
            try:
                print(f"Loading CodeLM from: {candidate}")
                tokenizer = AutoTokenizer.from_pretrained(candidate)
                model = AutoModel.from_pretrained(candidate).to(device)
                return tokenizer, model, candidate
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Unable to load a CodeLM model. Last error: {last_error}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def encode_input(self, text, tokenizer, device):
        if isinstance(text, list):
            return tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        return tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    def forward(self, data):
        data.x = self.reduce_embedding(data)
        x, edge_index, text = data.x, data.edge_index, data.func
        batch = data.batch if hasattr(data, "batch") else None

        h_local = self.local_encoder(x, edge_index)
        h_global = self.global_encoder(x, edge_index)
        h_struct, _ = self.dual_view_fusion(h_local, h_global, edge_index, batch)

        if h_struct.dim() > 1 and h_struct.size(0) > 1:
            h_struct = self.bn_struct(h_struct)
            h_struct = F.layer_norm(h_struct, h_struct.size()[1:])
        else:
            h_struct = (h_struct - h_struct.mean(dim=0)) / (h_struct.std(dim=0) + 1e-5)
        h_struct = F.dropout(h_struct, p=self.dropout_rate * 0.8, training=self.training)
        x_struct = self.conv(h_struct, data.x)

        inputs = self.encode_input(text, self.tokenizer, self.device)
        lm_outputs = self.code_lm(**inputs)
        h_semantic = lm_outputs.last_hidden_state[:, 0, :]
        if h_semantic.dim() > 1 and h_semantic.size(0) > 1:
            h_semantic = self.bn_semantic(h_semantic)
        h_semantic = self.dropout(h_semantic)

        if self.use_qformer:
            graph_nodes_padded, graph_mask = batch_to_padded(h_local, batch, max_nodes=256)
            graph_nodes_proj = self.graph_proj(graph_nodes_padded)
            doc_embedding, _ = self.qformer(
                semantic_seq=lm_outputs.last_hidden_state,
                graph_nodes=graph_nodes_proj,
                semantic_cls=h_semantic,
                semantic_mask=inputs.attention_mask,
                graph_mask=graph_mask,
            )
            pred = self.qformer_classifier(doc_embedding)
            if not self.training:
                return self.calibrate_confidence(pred)
            aux_struct = self.aux_struct_head(global_mean_pool(h_local, batch))
            aux_sem = self.semantic_classifier(h_semantic)
            return pred, aux_struct, aux_sem

        z_semantic = self.semantic_classifier(h_semantic)
        x_struct_raw = torch.clamp(x_struct, min=-5.0, max=5.0)
        if x_struct_raw.size(0) > 1:
            x_struct_norm = self.bn_struct_output(x_struct_raw)
        else:
            x_struct_norm = (x_struct_raw - x_struct_raw.mean(dim=0)) / (x_struct_raw.std(dim=0) + 1e-5)
        z_struct = self.struct_to_logits(x_struct_norm)
        z_fused, _ = self.semantic_struct_fusion(z_semantic, z_struct)
        if self.use_residual and z_semantic.size() == z_struct.size():
            z_fused = z_fused + 0.1 * (z_semantic + z_struct)
        bias_tensor = torch.tensor([[0.0, self.pos_bias_logit.item()]], device=self.device)
        pred = z_fused + bias_tensor
        return self.calibrate_confidence(pred) if not self.training else pred

    def reduce_embedding(self, data):
        expected_dim = self.feat_dim + 1
        actual_dim = data.x.size(-1)
        if actual_dim != expected_dim:
            return data.x.to(self.device)
        return self.semantic_node_init(data.x.to(self.device))

    def calibrate_confidence(self, logits):
        calibrated = logits / 1.5
        if calibrated.size(1) >= 2:
            calibrated[:, 1] += -0.1
        return calibrated

    def adjust_pos_bias_logit(self, f1_score, precision=None):
        if self.bias_adjustment_cool_down > 0:
            self.bias_adjustment_cool_down -= 1
            return
        self.bias_history_f1.append(f1_score)
        if precision is not None:
            self.precision_history.append(precision)
        if len(self.bias_history_f1) < 3:
            return
        current_precision = precision if precision is not None else 0.5
        adjustment_step = 0.01
        old_bias = self.pos_bias_logit.item()
        new_bias = old_bias
        triggered_rule = "none"
        if current_precision < 0.60 and f1_score > 0.1:
            new_bias -= adjustment_step
            triggered_rule = f"precision<0.60 ({current_precision:.2f})"
        elif f1_score < 0.4 and all(f < 0.45 for f in self.bias_history_f1[-3:]):
            new_bias += adjustment_step
            triggered_rule = f"f1<0.40 ({f1_score:.2f})"
        elif f1_score > 0.70 and current_precision < 0.75:
            new_bias -= adjustment_step * 0.5
            triggered_rule = f"high_f1_low_precision ({current_precision:.2f})"
        if triggered_rule != "none":
            self.pos_bias_logit.data.fill_(np.clip(new_bias, -0.35, 0.15))
            print(
                f"Bias adjustment triggered ({triggered_rule}): "
                f"F1={f1_score:.3f}, precision={current_precision:.3f}, "
                f"old_bias={old_bias:.4f}, new_bias={self.pos_bias_logit.item():.4f}"
            )
            self.bias_adjustment_cool_down = 2

    def get_dual_view_statistics(self):
        if hasattr(self.dual_view_fusion, "get_weight_statistics"):
            return self.dual_view_fusion.get_weight_statistics()
        return None

    def save(self, path):
        print(f"Saving model to: {path}")
        torch.save(self.state_dict(), path)
        print("Model saved successfully.")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print(f"Loaded model from: {path}")


UniXcoderLMGNN = DualGraphVulD
