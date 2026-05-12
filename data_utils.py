# -*- coding: utf-8 -*-
"""Utilities for loading graph anomaly detection .mat datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected


FEATURE_KEYS = ("x", "X", "features", "Attributes", "attr", "node_feat", "node_features")
LABEL_KEYS = ("y", "Y", "label", "labels", "gnd", "truth", "Class")
EDGE_INDEX_KEYS = ("edge_index", "edges", "edge", "EdgeIndex")
ADJ_KEYS = ("adj", "A", "network", "Network", "graph", "Graph")


def _first_existing(mat: Dict[str, Any], keys: Iterable[str]) -> Tuple[Optional[str], Optional[Any]]:
    for key in keys:
        if key in mat:
            return key, mat[key]
    return None, None


def _to_numpy_dense(value: Any) -> np.ndarray:
    if sp.issparse(value):
        return value.toarray()
    arr = np.asarray(value)
    if arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.item())
    return arr


def _load_features(mat: Dict[str, Any]) -> torch.Tensor:
    key, value = _first_existing(mat, FEATURE_KEYS)
    if value is None:
        raise KeyError(f"No feature matrix found. Tried keys: {FEATURE_KEYS}")
    x = _to_numpy_dense(value)
    if x.ndim != 2:
        raise ValueError(f"Feature field {key!r} must be 2-D, got shape {x.shape}.")
    return torch.as_tensor(x, dtype=torch.float32)


def _load_labels(mat: Dict[str, Any], num_nodes: int) -> torch.Tensor:
    key, value = _first_existing(mat, LABEL_KEYS)
    if value is None:
        raise KeyError(f"No label vector found. Tried keys: {LABEL_KEYS}")
    y = _to_numpy_dense(value).reshape(-1)
    if y.shape[0] != num_nodes:
        raise ValueError(
            f"Label field {key!r} length {y.shape[0]} does not match num_nodes={num_nodes}."
        )
    # GAD benchmark convention: 0=normal, non-zero=anomaly.
    y = (y != 0).astype(np.int64)
    return torch.as_tensor(y, dtype=torch.long)


def _edge_index_from_adj(adj: Any) -> torch.Tensor:
    if not sp.issparse(adj):
        adj = sp.coo_matrix(np.asarray(adj))
    else:
        adj = adj.tocoo()
    row = torch.as_tensor(adj.row, dtype=torch.long)
    col = torch.as_tensor(adj.col, dtype=torch.long)
    return torch.stack([row, col], dim=0)


def _edge_index_from_raw(value: Any, num_nodes: int) -> torch.Tensor:
    arr = _to_numpy_dense(value)
    arr = np.asarray(arr)

    if arr.ndim != 2:
        raise ValueError(f"edge_index/edges must be 2-D, got shape {arr.shape}.")

    # Accept [2, E] or [E, 2].
    if arr.shape[0] == 2:
        edge_index = torch.as_tensor(arr, dtype=torch.long)
    elif arr.shape[1] == 2:
        edge_index = torch.as_tensor(arr.T, dtype=torch.long)
    else:
        # Some datasets store adjacency-like dense matrices under edge keys.
        if arr.shape[0] == num_nodes and arr.shape[1] == num_nodes:
            edge_index = _edge_index_from_adj(arr)
        else:
            raise ValueError(f"Cannot interpret edge field with shape {arr.shape}.")

    # Convert possible 1-based Matlab indices to 0-based.
    if edge_index.numel() > 0 and int(edge_index.min()) == 1 and int(edge_index.max()) == num_nodes:
        edge_index = edge_index - 1
    return edge_index


def _load_edge_index(mat: Dict[str, Any], num_nodes: int) -> torch.Tensor:
    _, value = _first_existing(mat, EDGE_INDEX_KEYS)
    if value is not None:
        edge_index = _edge_index_from_raw(value, num_nodes)
    else:
        _, adj = _first_existing(mat, ADJ_KEYS)
        if adj is None:
            raise KeyError(f"No graph structure found. Tried edge keys {EDGE_INDEX_KEYS} and adj keys {ADJ_KEYS}")
        edge_index = _edge_index_from_adj(adj)

    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    return edge_index.long()


def load_mat_dataset(dataset: str, data_dir: str | Path = "~/datasets/GAD/mat") -> Data:
    data_dir = Path(data_dir).expanduser()
    path = data_dir / dataset
    if path.suffix != ".mat":
        path = path.with_suffix(".mat")
    if not path.exists():
        available = sorted(p.stem for p in data_dir.glob("*.mat")) if data_dir.exists() else []
        raise FileNotFoundError(
            f"Dataset not found: {path}\nAvailable datasets: {available}"
        )

    mat = sio.loadmat(path)
    mat = {k: v for k, v in mat.items() if not k.startswith("__")}

    x = _load_features(mat)
    y = _load_labels(mat, num_nodes=x.size(0))
    edge_index = _load_edge_index(mat, num_nodes=x.size(0))

    data = Data(x=x, edge_index=edge_index, y=y)
    data.name = path.stem
    return data
