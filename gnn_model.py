"""
DrugForge AI graph neural network model

This module defines:
1. An edge-aware message-passing layer
2. A graph + descriptor fusion model for classification
3. Helper methods for probability prediction and uncertainty estimation
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, dropout=0.15):
        super().__init__()
        self.msg_linear = nn.Linear(in_feats, out_feats)
        self.self_linear = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = (in_feats == out_feats)

    def _safe_batch_norm(self, x):
        if x.dim() == 2 and x.size(0) > 1:
            return self.bn(x)
        return x

    def forward(self, g, h, e):
        with g.local_scope():
            g.ndata["h"] = h

            if e is None or e.numel() == 0:
                edge_weights = torch.ones((g.num_edges(), 1), device=h.device, dtype=h.dtype)
            else:
                edge_weights = e.to(device=h.device, dtype=h.dtype)
                if edge_weights.dim() == 1:
                    edge_weights = edge_weights.unsqueeze(1)

                edge_weights = torch.nan_to_num(
                    edge_weights,
                    nan=0.0,
                    posinf=5.0,
                    neginf=0.0
                )
                edge_weights = torch.clamp(edge_weights, min=0.0, max=5.0)

            g.edata["w"] = edge_weights

            g.update_all(
                dgl.function.u_mul_e("h", "w", "m"),
                dgl.function.mean("m", "agg_h")
            )

            agg_h = g.ndata["agg_h"]
            out = self.msg_linear(agg_h) + self.self_linear(h)
            out = self._safe_batch_norm(out)
            out = F.relu(out)
            out = self.dropout(out)

            if self.use_residual:
                out = out + h

            return out


class GNN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, dropout=0.25, feature_size=1197):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.dropout_rate = dropout

        self.gnn1 = GNNLayer(in_feats, hidden_size, dropout=dropout)
        self.gnn2 = GNNLayer(hidden_size, hidden_size, dropout=dropout)
        self.gnn3 = GNNLayer(hidden_size, hidden_size, dropout=dropout)
        self.gnn4 = GNNLayer(hidden_size, hidden_size, dropout=dropout)

        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _safe_sequential_forward(self, module, x):
        out = x
        for layer in module:
            if isinstance(layer, nn.BatchNorm1d):
                if out.dim() == 2 and out.size(0) > 1:
                    out = layer(out)
            else:
                out = layer(out)
        return out

    def _encode_graph(self, g):
        if "h" not in g.ndata:
            raise ValueError("Graph is missing node features 'h'.")

        h = g.ndata["h"]
        e = g.edata["e"] if "e" in g.edata else None

        h = self.gnn1(g, h, e)
        h = self.gnn2(g, h, e)
        h = self.gnn3(g, h, e)
        h = self.gnn4(g, h, e)

        with g.local_scope():
            g.ndata["h_final"] = h
            h_mean = dgl.mean_nodes(g, "h_final")
            h_max = dgl.max_nodes(g, "h_final")

        graph_repr = torch.cat([h_mean, h_max], dim=1)
        return graph_repr

    def _encode_features(self, features):
        if features.dim() == 1:
            features = features.unsqueeze(0)

        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        return self._safe_sequential_forward(self.feature_mlp, features)

    def get_features(self, g, features):
        graph_repr = self._encode_graph(g)
        feature_repr = self._encode_features(features)

        combined = torch.cat([graph_repr, feature_repr], dim=1)
        combined = self._safe_sequential_forward(self.combine_mlp, combined)
        return combined

    def forward(self, g, features):
        combined = self.get_features(g, features)
        logits = self.classifier(combined)
        return logits

    @torch.no_grad()
    def predict_proba(self, g, features):
        was_training = self.training
        self.eval()

        logits = self.forward(g, features)
        probs = torch.softmax(logits, dim=1)

        if was_training:
            self.train()

        return probs

    @torch.no_grad()
    def predict_with_uncertainty(self, g, features, n_passes=20):
        n_passes = max(int(n_passes), 2)
        was_training = self.training
        self.train()

        probs_list = []
        for _ in range(n_passes):
            logits = self.forward(g, features)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs.unsqueeze(0))

        stacked = torch.cat(probs_list, dim=0)
        mean_probs = stacked.mean(dim=0)
        std_probs = stacked.std(dim=0)

        if not was_training:
            self.eval()

        return mean_probs, std_probs