"""
DrugForge AI Advanced 3D GNN model

This module defines:
1. Distance-aware 3D message passing
2. EGNN-style coordinate-aware graph updates
3. SchNet-style radial distance encoding
4. Mean + max + attention pooling
5. Graph + RDKit descriptor/fingerprint fusion
6. Classification, probability prediction, uncertainty estimation
7. Embedding extraction for PCA/t-SNE/UMAP/clustering

Compatible with gnn3d_utils.py:
- NODE_FEATURE_DIM_3D
- EDGE_FEATURE_DIM_3D
- FEATURE_VECTOR_DIM

Expected graph fields:
- g.ndata["h"]   : atom features, shape [num_nodes, node_feature_dim]
- g.ndata["pos"] : 3D coordinates, shape [num_nodes, 3]
- g.edata["e"]   : edge features, shape [num_edges, edge_feature_dim]
- g.edata["dist"]: distances, shape [num_edges, 1]
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeBatchNorm1d(nn.BatchNorm1d):
    """
    BatchNorm1d that safely skips normalization when batch size is 1.

    This prevents errors during single-molecule inference.
    """

    def forward(self, x):
        if x.dim() == 2 and x.size(0) > 1:
            return super().forward(x)
        return x


class RadialBasisExpansion(nn.Module):
    """
    SchNet-style radial basis expansion for interatomic distances.

    Converts scalar distance d into a smooth vector representation.
    """

    def __init__(
        self,
        num_rbf: int = 32,
        cutoff: float = 10.0,
        gamma: Optional[float] = None,
    ):
        super().__init__()

        self.num_rbf = int(num_rbf)
        self.cutoff = float(cutoff)

        centers = torch.linspace(0.0, self.cutoff, self.num_rbf)
        self.register_buffer("centers", centers)

        if gamma is None:
            if self.num_rbf > 1:
                spacing = centers[1] - centers[0]
                gamma = 1.0 / float(spacing ** 2)
            else:
                gamma = 1.0

        self.gamma = float(gamma)

    def forward(self, distances):
        """
        Parameters
        ----------
        distances : torch.Tensor
            Shape [num_edges, 1] or [num_edges]

        Returns
        -------
        torch.Tensor
            Shape [num_edges, num_rbf]
        """
        if distances.dim() == 1:
            distances = distances.unsqueeze(1)

        distances = torch.nan_to_num(
            distances,
            nan=0.0,
            posinf=self.cutoff,
            neginf=0.0,
        )

        distances = distances.clamp(min=0.0, max=self.cutoff)

        diff = distances - self.centers.view(1, -1)
        rbf = torch.exp(-self.gamma * diff * diff)

        # Smooth cosine cutoff.
        cutoff_weight = 0.5 * (
            torch.cos(math.pi * distances / self.cutoff) + 1.0
        )
        cutoff_weight = cutoff_weight * (distances <= self.cutoff).float()

        return rbf * cutoff_weight


class EGNN3DLayer(nn.Module):
    """
    EGNN/SchNet-style 3D message passing layer.

    This layer uses:
    - atom embeddings
    - 3D edge features
    - radial distance encoding
    - coordinate differences
    - optional coordinate update

    It is not a pure SchNet layer and not a full formal EGNN implementation,
    but it follows the practical idea of distance-aware and coordinate-aware
    molecular message passing.
    """

    def __init__(
        self,
        hidden_size: int,
        edge_feats: int,
        num_rbf: int = 32,
        dropout: float = 0.20,
        update_coords: bool = True,
        coord_update_scale: float = 0.10,
    ):
        super().__init__()

        self.hidden_size = int(hidden_size)
        self.edge_feats = int(edge_feats)
        self.num_rbf = int(num_rbf)
        self.update_coords = bool(update_coords)
        self.coord_update_scale = float(coord_update_scale)

        edge_input_dim = hidden_size * 2 + edge_feats + num_rbf

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_size * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
        )

        self.edge_gate = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_size),
            nn.Sigmoid(),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            SafeBatchNorm1d(hidden_size * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _prepare_edge_features(self, e, num_edges, device, dtype):
        """
        Robustly pad/truncate edge features.
        """
        if e is None or e.numel() == 0:
            return torch.zeros(
                (num_edges, self.edge_feats),
                device=device,
                dtype=dtype,
            )

        e = e.to(device=device, dtype=dtype)

        if e.dim() == 1:
            e = e.unsqueeze(1)

        e = torch.nan_to_num(e, nan=0.0, posinf=1.0, neginf=0.0)

        if e.size(1) < self.edge_feats:
            pad = torch.zeros(
                (e.size(0), self.edge_feats - e.size(1)),
                device=device,
                dtype=dtype,
            )
            e = torch.cat([e, pad], dim=1)
        elif e.size(1) > self.edge_feats:
            e = e[:, :self.edge_feats]

        if e.size(0) != num_edges:
            fixed = torch.zeros(
                (num_edges, self.edge_feats),
                device=device,
                dtype=dtype,
            )
            n = min(num_edges, e.size(0))
            fixed[:n] = e[:n]
            e = fixed

        return e

    def _prepare_distances(self, g, pos, dist):
        """
        Use stored distances if available; otherwise calculate from coordinates.
        """
        if dist is not None and dist.numel() > 0:
            if dist.dim() == 1:
                dist = dist.unsqueeze(1)
            return torch.nan_to_num(dist, nan=0.0, posinf=10.0, neginf=0.0)

        with g.local_scope():
            g.ndata["pos_tmp"] = pos

            def calc_dist(edges):
                diff = edges.dst["pos_tmp"] - edges.src["pos_tmp"]
                d = torch.norm(diff, p=2, dim=1, keepdim=True)
                return {"computed_dist": d}

            g.apply_edges(calc_dist)
            return g.edata["computed_dist"]

    def forward(
        self,
        g,
        h,
        pos,
        edge_attr=None,
        dist=None,
        rbf_layer: Optional[RadialBasisExpansion] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        g : dgl.DGLGraph
        h : torch.Tensor
            Node embeddings [num_nodes, hidden_size]
        pos : torch.Tensor
            Coordinates [num_nodes, 3]
        edge_attr : torch.Tensor
            Edge features [num_edges, edge_feats]
        dist : torch.Tensor
            Edge distances [num_edges, 1]
        rbf_layer : RadialBasisExpansion

        Returns
        -------
        h_new, pos_new
        """
        if h is None:
            raise ValueError("Node embedding h cannot be None.")

        if pos is None:
            raise ValueError("3D graph is missing coordinates g.ndata['pos'].")

        if rbf_layer is None:
            raise ValueError("rbf_layer cannot be None.")

        h = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=0.0)
        pos = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)

        num_edges = g.num_edges()

        edge_attr = self._prepare_edge_features(
            edge_attr,
            num_edges=num_edges,
            device=h.device,
            dtype=h.dtype,
        )

        dist = self._prepare_distances(g, pos, dist)
        dist = dist.to(device=h.device, dtype=h.dtype)
        rbf = rbf_layer(dist)

        with g.local_scope():
            g.ndata["h"] = h
            g.ndata["pos"] = pos
            g.edata["edge_attr"] = edge_attr
            g.edata["dist"] = dist
            g.edata["rbf"] = rbf

            def edge_message(edges):
                src_h = edges.src["h"]
                dst_h = edges.dst["h"]
                edge_h = edges.data["edge_attr"]
                rbf_h = edges.data["rbf"]

                edge_input = torch.cat([src_h, dst_h, edge_h, rbf_h], dim=1)

                msg = self.edge_mlp(edge_input)
                gate = self.edge_gate(edge_input)
                msg = msg * gate

                coord_diff = edges.dst["pos"] - edges.src["pos"]

                if self.update_coords:
                    coord_weight = self.coord_mlp(msg) * self.coord_update_scale
                    coord_msg = coord_diff * coord_weight
                else:
                    coord_msg = torch.zeros_like(coord_diff)

                return {
                    "msg": msg,
                    "coord_msg": coord_msg,
                }

            g.apply_edges(edge_message)

            g.update_all(
                fn.copy_e("msg", "m"),
                fn.mean("m", "agg_msg"),
            )

            if self.update_coords:
                g.update_all(
                    fn.copy_e("coord_msg", "cm"),
                    fn.mean("cm", "agg_coord"),
                )
                coord_update = g.ndata["agg_coord"]
            else:
                coord_update = torch.zeros_like(pos)

            agg_msg = g.ndata["agg_msg"]

            node_input = torch.cat([h, agg_msg], dim=1)
            dh = self.node_mlp(node_input)
            dh = self.dropout(dh)

            h_new = self.norm(h + dh)
            pos_new = pos + coord_update

            # Keep coordinates numerically stable.
            pos_new = torch.nan_to_num(pos_new, nan=0.0, posinf=0.0, neginf=0.0)

            return h_new, pos_new


class AttentionPooling(nn.Module):
    """
    Learnable graph-level attention pooling.

    This learns which atoms contribute most to predicted activity.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.20):
        super().__init__()

        attn_hidden = max(hidden_size // 2, 32)

        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_size, attn_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, 1),
        )

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h_pool"] = h
            g.ndata["gate_logits"] = self.gate_nn(h)

            try:
                weights = dgl.softmax_nodes(g, "gate_logits")
            except Exception:
                weights = torch.ones_like(g.ndata["gate_logits"])

            g.ndata["weighted_h"] = g.ndata["h_pool"] * weights
            pooled = dgl.sum_nodes(g, "weighted_h")

            return pooled

    @torch.no_grad()
    def get_attention_weights(self, g, h):
        with g.local_scope():
            g.ndata["gate_logits"] = self.gate_nn(h)

            try:
                weights = dgl.softmax_nodes(g, "gate_logits")
            except Exception:
                weights = torch.ones_like(g.ndata["gate_logits"])

            return weights.detach()


class Advanced3DGNN(nn.Module):
    """
    Advanced 3D GNN for ligand activity prediction.

    Architecture:
        3D molecular graph
        -> atom encoder
        -> distance-aware EGNN layers
        -> mean + max + attention pooling
        -> RDKit descriptor/fingerprint MLP
        -> fused representation
        -> classifier

    Parameters
    ----------
    node_feats : int
        Atom feature dimension from gnn3d_utils.NODE_FEATURE_DIM_3D.
    edge_feats : int
        3D edge feature dimension from gnn3d_utils.EDGE_FEATURE_DIM_3D.
    hidden_size : int
        Hidden representation dimension.
    num_classes : int
        Usually 2 for inactive/active.
    num_layers : int
        Number of 3D message-passing layers.
    dropout : float
        Dropout rate.
    feature_size : int
        RDKit descriptor/fingerprint vector dimension. Default 1197.
    num_rbf : int
        Number of radial basis functions.
    cutoff : float
        Distance cutoff used in radial basis expansion.
    update_coords : bool
        Whether to update coordinates inside EGNN layers.
    """

    def __init__(
        self,
        node_feats: int,
        edge_feats: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 4,
        dropout: float = 0.25,
        feature_size: int = 1197,
        num_rbf: int = 32,
        cutoff: float = 10.0,
        update_coords: bool = True,
    ):
        super().__init__()

        self.node_feats = int(node_feats)
        self.edge_feats = int(edge_feats)
        self.hidden_size = int(hidden_size)
        self.num_classes = int(num_classes)
        self.num_layers = int(num_layers)
        self.dropout_rate = float(dropout)
        self.feature_size = int(feature_size)
        self.num_rbf = int(num_rbf)
        self.cutoff = float(cutoff)
        self.update_coords = bool(update_coords)

        self.atom_encoder = nn.Sequential(
            nn.Linear(self.node_feats, self.hidden_size),
            SafeBatchNorm1d(self.hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )

        self.rbf_layer = RadialBasisExpansion(
            num_rbf=self.num_rbf,
            cutoff=self.cutoff,
        )

        self.layers = nn.ModuleList([
            EGNN3DLayer(
                hidden_size=self.hidden_size,
                edge_feats=self.edge_feats,
                num_rbf=self.num_rbf,
                dropout=dropout,
                update_coords=update_coords,
                coord_update_scale=0.10,
            )
            for _ in range(self.num_layers)
        ])

        self.attention_pool = AttentionPooling(
            hidden_size=self.hidden_size,
            dropout=dropout,
        )

        self.feature_mlp = nn.Sequential(
            nn.Linear(self.feature_size, self.hidden_size * 2),
            SafeBatchNorm1d(self.hidden_size * 2),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(self.hidden_size * 2, self.hidden_size),
            SafeBatchNorm1d(self.hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # graph representation = mean + max + attention = hidden_size * 3
        # descriptor representation = hidden_size
        fusion_input_dim = self.hidden_size * 4

        self.combine_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_size * 2),
            SafeBatchNorm1d(self.hidden_size * 2),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(self.hidden_size * 2, self.hidden_size),
            SafeBatchNorm1d(self.hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, max(self.hidden_size // 2, 32)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(max(self.hidden_size // 2, 32), self.num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _safe_encode_features(self, features):
        if features is None:
            raise ValueError("Molecular descriptor/fingerprint tensor cannot be None.")

        if features.dim() == 1:
            features = features.unsqueeze(0)

        features = torch.nan_to_num(
            features,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        if features.size(1) != self.feature_size:
            if features.size(1) < self.feature_size:
                pad = torch.zeros(
                    (features.size(0), self.feature_size - features.size(1)),
                    device=features.device,
                    dtype=features.dtype,
                )
                features = torch.cat([features, pad], dim=1)
            else:
                features = features[:, :self.feature_size]

        return self.feature_mlp(features)

    def _prepare_graph_inputs(self, g):
        if "h" not in g.ndata:
            raise ValueError("3D graph is missing node features g.ndata['h'].")

        if "pos" not in g.ndata:
            raise ValueError("3D graph is missing coordinates g.ndata['pos'].")

        h = g.ndata["h"]
        pos = g.ndata["pos"]

        edge_attr = g.edata["e"] if "e" in g.edata else None
        dist = g.edata["dist"] if "dist" in g.edata else None

        h = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=0.0)
        pos = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)

        return h, pos, edge_attr, dist

    def _encode_graph(self, g):
        """
        Encode 3D molecular graph into graph-level representation.
        """
        h, pos, edge_attr, dist = self._prepare_graph_inputs(g)

        h = self.atom_encoder(h)

        for layer in self.layers:
            h, pos = layer(
                g=g,
                h=h,
                pos=pos,
                edge_attr=edge_attr,
                dist=dist,
                rbf_layer=self.rbf_layer,
            )

            # Recalculate edge distances after coordinate update.
            # If update_coords=False, this has negligible effect.
            with g.local_scope():
                g.ndata["pos_tmp"] = pos

                def calc_updated_dist(edges):
                    diff = edges.dst["pos_tmp"] - edges.src["pos_tmp"]
                    d = torch.norm(diff, p=2, dim=1, keepdim=True)
                    return {"updated_dist": d}

                g.apply_edges(calc_updated_dist)
                dist = g.edata["updated_dist"]

        with g.local_scope():
            g.ndata["h_final"] = h

            h_mean = dgl.mean_nodes(g, "h_final")
            h_max = dgl.max_nodes(g, "h_final")
            h_attn = self.attention_pool(g, h)

        graph_repr = torch.cat([h_mean, h_max, h_attn], dim=1)

        return graph_repr

    def get_features(self, g, features):
        """
        Return fused molecular embedding before classifier.

        Useful for:
        - PCA
        - t-SNE
        - UMAP
        - clustering
        - similarity analysis
        """
        graph_repr = self._encode_graph(g)
        feature_repr = self._safe_encode_features(features)

        combined = torch.cat([graph_repr, feature_repr], dim=1)
        combined = self.combine_mlp(combined)

        return combined

    def forward(self, g, features):
        """
        Forward pass.

        Returns
        -------
        logits : torch.Tensor
            Shape [batch_size, num_classes]
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
    def predict_with_uncertainty(self, g, features, n_passes: int = 20):
        """
        Monte-Carlo dropout uncertainty estimation.

        Returns
        -------
        mean_probs : torch.Tensor
            Mean probabilities across MC dropout passes.
        std_probs : torch.Tensor
            Standard deviation of probabilities across MC dropout passes.
        """
        n_passes = max(int(n_passes), 2)

        was_training = self.training

        # Keep dropout active.
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
        Extract fused embeddings for visualization/clustering.
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
        Return atom-level attention weights from final graph representation.

        Can be used later for explainable GNN visualization.
        """
        was_training = self.training
        self.eval()

        h, pos, edge_attr, dist = self._prepare_graph_inputs(g)
        h = self.atom_encoder(h)

        for layer in self.layers:
            h, pos = layer(
                g=g,
                h=h,
                pos=pos,
                edge_attr=edge_attr,
                dist=dist,
                rbf_layer=self.rbf_layer,
            )

            with g.local_scope():
                g.ndata["pos_tmp"] = pos

                def calc_updated_dist(edges):
                    diff = edges.dst["pos_tmp"] - edges.src["pos_tmp"]
                    d = torch.norm(diff, p=2, dim=1, keepdim=True)
                    return {"updated_dist": d}

                g.apply_edges(calc_updated_dist)
                dist = g.edata["updated_dist"]

        weights = self.attention_pool.get_attention_weights(g, h)

        if was_training:
            self.train()

        return weights


# Backward-compatible alias if you want to import GNN3D.
GNN3D = Advanced3DGNN


__all__ = [
    "SafeBatchNorm1d",
    "RadialBasisExpansion",
    "EGNN3DLayer",
    "AttentionPooling",
    "Advanced3DGNN",
    "GNN3D",
]