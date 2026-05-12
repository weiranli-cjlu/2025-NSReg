# -*- coding: utf-8 -*-
"""A compact NSReg implementation for GAD .mat benchmark datasets.

This file intentionally contains only the code needed by run.py:
- GraphSAGE encoder
- binary anomaly classifier
- normal-structure regularisation loss
No checkpointing, plotting, tensorboard, debug printing, or saved-index logic is kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from tqdm import trange


class GraphSAGEEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_dim, out_dim))
        else:
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.convs.append(SAGEConv(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i != len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NSRegModel(nn.Module):
    """Normal Structure Regularisation model.

    The supervised branch learns a binary anomaly score. The NSReg branch pulls
    normal-node embeddings toward their normal-neighbour structural prototype,
    making normal representations compact and less overfitted to the observed
    anomaly labels.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        emb_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=emb_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.classifier = MLP(emb_dim, hidden_dim, 1, dropout=dropout)
        self.projector = MLP(emb_dim, emb_dim, emb_dim, dropout=dropout)

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(data.x, data.edge_index)
        logits = self.classifier(z).view(-1)
        return logits, z


def normal_structure_regularisation(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    y_train: torch.Tensor,
    train_mask: torch.Tensor,
) -> torch.Tensor:
    """Compactness regularisation for labelled normal nodes.

    For each labelled normal node, aggregate embeddings of its labelled normal
    neighbours, then minimise cosine distance between the node and the aggregate.
    If a node has no labelled normal neighbours, it is skipped.
    """
    device = z.device
    normal_mask = (y_train == 0) & train_mask
    if int(normal_mask.sum()) <= 1 or edge_index.numel() == 0:
        return z.new_tensor(0.0)

    row, col = edge_index
    valid_edge = normal_mask[row] & normal_mask[col]
    row = row[valid_edge]
    col = col[valid_edge]
    if row.numel() == 0:
        return z.new_tensor(0.0)

    z_norm = F.normalize(z, p=2, dim=-1)
    proto = torch.zeros_like(z_norm)
    deg = torch.zeros(z.size(0), device=device, dtype=z.dtype)
    proto.index_add_(0, row, z_norm[col])
    deg.index_add_(0, row, torch.ones_like(row, dtype=z.dtype))

    has_proto = normal_mask & (deg > 0)
    if int(has_proto.sum()) == 0:
        return z.new_tensor(0.0)

    proto = proto / deg.clamp_min(1.0).unsqueeze(-1)
    proto = F.normalize(proto, p=2, dim=-1)
    loss = 1.0 - (z_norm[has_proto] * proto[has_proto]).sum(dim=-1)
    return loss.mean()


@dataclass
class FitOutput:
    scores: torch.Tensor
    logits: torch.Tensor
    loss: float


class NSRegDetector:
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        emb_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 200,
        nsreg_weight: float = 1.0,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.nsreg_weight = nsreg_weight
        self.model = NSRegModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            emb_dim=emb_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def fit_predict_score(
        self,
        data: Data,
        train_mask: torch.Tensor,
        y_train: torch.Tensor,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> FitOutput:
        data = data.to(self.device)
        train_mask = train_mask.to(self.device)
        y_train = y_train.to(self.device).float()
        if pos_weight is not None:
            pos_weight = pos_weight.to(self.device)

        last_loss = 0.0
        for _ in trange(self.epochs, desc="Epoch", position=1, leave=False):
            self.model.train()
            self.optim.zero_grad()
            logits, z = self.model(data)
            clf_loss = F.binary_cross_entropy_with_logits(
                logits[train_mask], y_train[train_mask], pos_weight=pos_weight
            )
            reg_loss = normal_structure_regularisation(
                z=z,
                edge_index=data.edge_index,
                y_train=y_train.long(),
                train_mask=train_mask,
            )
            loss = clf_loss + self.nsreg_weight * reg_loss
            loss.backward()
            self.optim.step()
            last_loss = float(loss.detach().cpu())

        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(data)
            scores = torch.sigmoid(logits).detach().cpu()
        return FitOutput(scores=scores, logits=logits.detach().cpu(), loss=last_loss)
