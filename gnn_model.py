"""
DrugForge AI graph neural network model - Enhanced 2D version

This module defines:
1. Edge-feature-aware attention message-passing layer
2. Mean + max + attention graph pooling
3. Graph + descriptor/fingerprint fusion model
4. Classification, probability prediction, embedding extraction
5. Monte-Carlo dropout uncertainty estimation

Compatible with enhanced gnn_utils.py:
- ATOM_FEATURE_DIM = 31
- BOND_FEATURE_DIM = 15
- FEATURE_VECTOR_DIM = 1197

Expected graph fields:
- g.ndata["h"] : atom features, shape [num_nodes, in_feats]
- g.edata["e"] : bond features, shape [num_edges, edge_feats]
"""

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from dgl.nn.functional import edge_softmax
except Exception:  # pragma: no cover
    edge_softmax = None


class SafeBatchNorm1d(nn.BatchNorm1d):
    """
    BatchNorm1d that safely skips normalization when batch size is 1.

    This avoids runtime problems during single-molecule inference.
    """

    def forward(self, x):
        if x.dim() == 2 and x.size(0) > 1:
            return super().forward(x)
        return x


class GNNLayer(nn.Module):
    """
    Edge-aware attention message-passing layer.

    Difference from the old layer:
    - Old: edge feature was treated mostly as scalar bond weight.
    - New: full bond feature vector is encoded and used for message gating
      and attention over neighboring atoms.

    Message idea:
        source atom embedding + bond embedding
        -> edge gate
        -> attention normalization
        -> aggregated atom update
    """

    def __init__(self, in_feats, out_feats, edge_feats=15, dropout=0.15):
        super().__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.edge_feats = edge_feats

        self.msg_linear = nn.Linear(in_feats, out_feats)
        self.self_linear = nn.Linear(in_feats, out_feats)

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feats, out_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_feats, out_feats),
            nn.ReLU()
        )

        self.edge_gate = nn.Sequential(
            nn.Linear(edge_feats, out_feats),
            nn.Sigmoid()
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(out_feats * 3, out_feats),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Linear(out_feats, 1)
        )

        self.bn = SafeBatchNorm1d(out_feats)
        self.dropout = nn.Dropout(dropout)

        if in_feats == out_feats:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(in_feats, out_feats)

    def _prepare_edge_features(self, e, num_edges, device, dtype):
        """
        Make edge feature tensor robust to missing/old edge features.

        If old graph has edge dimension 1, it is padded to edge_feats.
        If edge dimension is larger, it is truncated.
        """
        if e is None or e.numel() == 0:
            return torch.zeros(
                (num_edges, self.edge_feats),
                device=device,
                dtype=dtype
            )

        e = e.to(device=device, dtype=dtype)

        if e.dim() == 1:
            e = e.unsqueeze(1)

        e = torch.nan_to_num(e, nan=0.0, posinf=1.0, neginf=0.0)

        if e.size(1) < self.edge_feats:
            pad = torch.zeros(
                (e.size(0), self.edge_feats - e.size(1)),
                device=device,
                dtype=dtype
            )
            e = torch.cat([e, pad], dim=1)
        elif e.size(1) > self.edge_feats:
            e = e[:, :self.edge_feats]

        if e.size(0) != num_edges:
            # Safety fallback if graph/edge-feature mismatch occurs.
            e_fixed = torch.zeros(
                (num_edges, self.edge_feats),
                device=device,
                dtype=dtype
            )
            n = min(num_edges, e.size(0))
            e_fixed[:n] = e[:n]
            e = e_fixed

        return e

    def forward(self, g, h, e=None):
        """
        Parameters
        ----------
        g : dgl.DGLGraph
        h : torch.Tensor
            Node features/embeddings.
        e : torch.Tensor or None
            Edge features.

        Returns
        -------
        torch.Tensor
            Updated node embeddings.
        """
        if h is None:
            raise ValueError("Node features h cannot be None.")

        with g.local_scope():
            h = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=0.0)

            num_edges = g.num_edges()
            e = self._prepare_edge_features(
                e=e,
                num_edges=num_edges,
                device=h.device,
                dtype=h.dtype
            )

            h_msg = self.msg_linear(h)
            h_self = self.self_linear(h)
            edge_emb = self.edge_encoder(e)
            edge_gate = self.edge_gate(e)

            g.ndata["h_msg"] = h_msg
            g.edata["edge_emb"] = edge_emb
            g.edata["edge_gate"] = edge_gate

            def edge_attention(edges):
                src_h = edges.src["h_msg"]
                dst_h = edges.dst["h_msg"]
                e_h = edges.data["edge_emb"]
                gate = edges.data["edge_gate"]

                raw_msg = (src_h + e_h) * gate
                attn_input = torch.cat([src_h, dst_h, e_h], dim=1)
                attn_logits = self.attn_mlp(attn_input)

                return {
                    "raw_msg": raw_msg,
                    "attn_logits": attn_logits
                }

            g.apply_edges(edge_attention)

            if edge_softmax is not None and g.num_edges() > 0:
                attn = edge_softmax(g, g.edata["attn_logits"])
            else:
                # Fallback: if edge_softmax is unavailable, use unnormalized messages.
                attn = torch.ones_like(g.edata["attn_logits"])

            g.edata["msg"] = g.edata["raw_msg"] * attn

            g.update_all(
                fn.copy_e("msg", "m"),
                fn.sum("m", "agg_h")
            )

            agg_h = g.ndata["agg_h"]

            out = agg_h + h_self
            out = self.bn(out)
            out = F.relu(out)
            out = self.dropout(out)

            out = out + self.residual(h)

            return out


class AttentionPooling(nn.Module):
    """
    Learnable graph-level attention pooling.

    It learns which atoms are more important for molecular activity.
    """

    def __init__(self, hidden_size, dropout=0.25):
        super().__init__()

        attn_hidden = max(hidden_size // 2, 32)

        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_size, attn_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, 1)
        )

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h_pool"] = h
            g.ndata["gate_logits"] = self.gate_nn(h)

            try:
                weights = dgl.softmax_nodes(g, "gate_logits")
            except Exception:
                # Safe fallback if softmax_nodes fails.
                weights = torch.ones_like(g.ndata["gate_logits"])

            g.ndata["weighted_h"] = g.ndata["h_pool"] * weights
            pooled = dgl.sum_nodes(g, "weighted_h")

            return pooled

    @torch.no_grad()
    def get_attention_weights(self, g, h):
        """
        Return atom-level attention weights for explainability.
        """
        with g.local_scope():
            g.ndata["gate_logits"] = self.gate_nn(h)

            try:
                weights = dgl.softmax_nodes(g, "gate_logits")
            except Exception:
                weights = torch.ones_like(g.ndata["gate_logits"])

            return weights.detach()


class GNN(nn.Module):
    """
    Enhanced 2D GNN for ligand activity prediction.

    Architecture:
        molecular graph
        -> 4 edge-aware attention message-passing layers
        -> mean pooling + max pooling + attention pooling
        -> descriptor/fingerprint MLP
        -> fusion MLP
        -> classifier

    Parameters
    ----------
    in_feats : int
        Atom feature dimension. Use ATOM_FEATURE_DIM from gnn_utils.py.
    hidden_size : int
        Hidden embedding size.
    num_classes : int
        Usually 2 for inactive/active classification.
    dropout : float
        Dropout rate.
    feature_size : int
        Molecular descriptor/fingerprint vector size. Default 1197.
    edge_feats : int
        Bond feature dimension. Use BOND_FEATURE_DIM from gnn_utils.py.
    """

    def __init__(
        self,
        in_feats,
        hidden_size,
        num_classes,
        dropout=0.25,
        feature_size=1197,
        edge_feats=15
    ):
        super().__init__()

        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.edge_feats = edge_feats
        self.dropout_rate = dropout

        self.gnn1 = GNNLayer(
            in_feats=in_feats,
            out_feats=hidden_size,
            edge_feats=edge_feats,
            dropout=dropout
        )
        self.gnn2 = GNNLayer(
            in_feats=hidden_size,
            out_feats=hidden_size,
            edge_feats=edge_feats,
            dropout=dropout
        )
        self.gnn3 = GNNLayer(
            in_feats=hidden_size,
            out_feats=hidden_size,
            edge_feats=edge_feats,
            dropout=dropout
        )
        self.gnn4 = GNNLayer(
            in_feats=hidden_size,
            out_feats=hidden_size,
            edge_feats=edge_feats,
            dropout=dropout
        )

        self.attention_pool = AttentionPooling(
            hidden_size=hidden_size,
            dropout=dropout
        )

        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_size, hidden_size * 2),
            SafeBatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size * 2, hidden_size),
            SafeBatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # graph_repr = mean + max + attention = hidden_size * 3
        # feature_repr = hidden_size
        fusion_input_dim = hidden_size * 4

        self.combine_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_size * 2),
            SafeBatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size * 2, hidden_size),
            SafeBatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, max(hidden_size // 2, 32)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_size // 2, 32), num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """
        Xavier initialization for linear layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _safe_sequential_forward(self, module, x):
        """
        Forward pass through Sequential while safely handling BatchNorm.
        """
        out = x
        for layer in module:
            out = layer(out)
        return out

    def _encode_graph(self, g):
        """
        Encode DGL graph into graph-level representation.
        """
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
            h_attn = self.attention_pool(g, h)

        graph_repr = torch.cat([h_mean, h_max, h_attn], dim=1)
        return graph_repr

    def _encode_features(self, features):
        """
        Encode RDKit descriptor/fingerprint vector.
        """
        if features is None:
            raise ValueError("Molecular feature tensor cannot be None.")

        if features.dim() == 1:
            features = features.unsqueeze(0)

        features = torch.nan_to_num(
            features,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        if features.size(1) != self.feature_size:
            if features.size(1) < self.feature_size:
                pad = torch.zeros(
                    (features.size(0), self.feature_size - features.size(1)),
                    device=features.device,
                    dtype=features.dtype
                )
                features = torch.cat([features, pad], dim=1)
            else:
                features = features[:, :self.feature_size]

        return self._safe_sequential_forward(self.feature_mlp, features)

    def get_features(self, g, features):
        """
        Return fused molecular embedding before classifier.

        This is useful for PCA, t-SNE, UMAP, clustering, and downstream ranking.
        """
        graph_repr = self._encode_graph(g)
        feature_repr = self._encode_features(features)

        combined = torch.cat([graph_repr, feature_repr], dim=1)
        combined = self._safe_sequential_forward(self.combine_mlp, combined)

        return combined

    def forward(self, g, features):
        """
        Forward classification pass.

        Returns
        -------
        logits : torch.Tensor
            Shape [batch_size, num_classes].
        """
        combined = self.get_features(g, features)
        logits = self.classifier(combined)
        return logits

    @torch.no_grad()
    def predict_proba(self, g, features):
        """
        Predict class probabilities.
        """
        was_training = self.training
        self.eval()

        logits = self.forward(g, features)
        probs = torch.softmax(logits, dim=1)

        if was_training:
            self.train()

        return probs

    @torch.no_grad()
    def predict_with_uncertainty(self, g, features, n_passes=20):
        """
        Monte-Carlo dropout uncertainty estimation.

        Returns
        -------
        mean_probs : torch.Tensor
            Mean class probabilities across stochastic passes.
        std_probs : torch.Tensor
            Standard deviation of class probabilities across passes.
        """
        n_passes = max(int(n_passes), 2)

        was_training = self.training

        # Keep dropout active for MC dropout.
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

    @torch.no_grad()
    def extract_embeddings(self, g, features):
        """
        Extract fused molecular embeddings for visualization or clustering.
        """
        was_training = self.training
        self.eval()

        embeddings = self.get_features(g, features)

        if was_training:
            self.train()

        return embeddings

    @torch.no_grad()
    def get_atom_attention_weights(self, g):
        """
        Return atom-level attention weights from the final graph encoder.

        This can be used later for explainability/visualization.
        """
        was_training = self.training
        self.eval()

        if "h" not in g.ndata:
            raise ValueError("Graph is missing node features 'h'.")

        h = g.ndata["h"]
        e = g.edata["e"] if "e" in g.edata else None

        h = self.gnn1(g, h, e)
        h = self.gnn2(g, h, e)
        h = self.gnn3(g, h, e)
        h = self.gnn4(g, h, e)

        weights = self.attention_pool.get_attention_weights(g, h)

        if was_training:
            self.train()

        return weights
