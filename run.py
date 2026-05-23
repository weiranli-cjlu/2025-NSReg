# -*- coding: utf-8 -*-
"""Command line entry for NSReg on GAD .mat datasets.

Example:
    python run.py --dataset ACM --n_trials 5 --lr 0.001
"""

from __future__ import annotations

import os
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from pandas import DataFrame
from tqdm import trange

from data_utils import load_mat_dataset
from model import NSRegDetector


@dataclass
class TrialResult:
    trial: int
    seed: int
    auc: float
    auprc: float
    loss: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_train_mask(
    y: torch.Tensor,
    train_ratio: float,
    num_train_anomaly: int,
    seed: int,
) -> torch.Tensor:
    """Create a reproducible supervised training split.

    - A fixed ratio of normal nodes is used as labelled normal training data.
    - Up to num_train_anomaly anomaly nodes are used as labelled seen anomalies.
    - Evaluation is always on all nodes, matching common GAD reporting.
    """
    if not (0.0 < train_ratio <= 1.0):
        raise ValueError("train_ratio must be in (0, 1].")

    rng = np.random.default_rng(seed)
    y_np = y.detach().cpu().numpy().astype(int)
    normal_idx = np.where(y_np == 0)[0]
    anomaly_idx = np.where(y_np != 0)[0]

    if len(normal_idx) == 0:
        raise ValueError("No normal nodes found; labels must use 0 for normal.")

    n_normal_train = max(1, int(round(len(normal_idx) * train_ratio)))
    normal_train = rng.choice(normal_idx, size=n_normal_train, replace=False)

    if num_train_anomaly < 0:
        raise ValueError("num_train_anomaly must be >= 0")
    n_anom_train = min(num_train_anomaly, len(anomaly_idx))
    if n_anom_train > 0:
        anom_train = rng.choice(anomaly_idx, size=n_anom_train, replace=False)
        train_idx = np.concatenate([normal_train, anom_train])
    else:
        train_idx = normal_train

    mask = torch.zeros(y.numel(), dtype=torch.bool)
    mask[torch.as_tensor(train_idx, dtype=torch.long)] = True
    return mask


def evaluate(y_true: torch.Tensor, scores: torch.Tensor) -> Tuple[float, float]:
    y_np = y_true.detach().cpu().numpy().astype(int)
    s_np = scores.detach().cpu().numpy().astype(float)
    if len(np.unique(y_np)) < 2:
        raise ValueError("AUC/AUPRC require both normal and anomaly labels.")
    auc_score = roc_auc_score(y_np, s_np)
    precision, recall, _ = precision_recall_curve(y_np, s_np)
    auprc_score = auc(recall, precision)
    return float(auc_score), float(auprc_score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NSReg on GAD .mat datasets.")

    # Required/common arguments
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name, e.g. ACM or ACM.mat"
    )
    parser.add_argument("--data_dir", type=str, default="~/datasets/GAD/mat")
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # Optimisation/model hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--nsreg_weight", type=float, default=1.0)

    # Supervised split hyperparameters
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.4,
        help="Ratio of normal nodes used as labelled normal training data.",
    )
    parser.add_argument(
        "--num_train_anomaly",
        type=int,
        default=10,
        help="Number of labelled anomaly nodes used as seen anomalies.",
    )
    parser.add_argument(
        "--balanced_loss",
        action="store_true",
        help="Use BCE pos_weight computed from the training split.",
    )

    parser.add_argument(
        "--result-csv",
        dest="result_csv",
        default=None,
        help="Append a summary row to a CSV file (e.g. results.csv)",
    )

    return parser.parse_args()


def run_one_trial(args: argparse.Namespace, trial: int, data) -> TrialResult:
    seed = args.seed + trial
    set_seed(seed)

    train_mask = make_train_mask(
        y=data.y,
        train_ratio=args.train_ratio,
        num_train_anomaly=args.num_train_anomaly,
        seed=seed,
    )

    pos_weight = None
    if args.balanced_loss:
        y_train = data.y[train_mask]
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        if n_pos > 0:
            pos_weight = torch.tensor([max(n_neg / n_pos, 1.0)], dtype=torch.float32)

    detector = NSRegDetector(
        in_dim=data.x.size(1),
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        num_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        nsreg_weight=args.nsreg_weight,
        device=args.device,
    )

    output = detector.fit_predict_score(
        data=data,
        train_mask=train_mask,
        y_train=data.y,
        pos_weight=pos_weight,
    )
    auc, auprc = evaluate(data.y, output.scores)
    return TrialResult(trial=trial + 1, seed=seed, auc=auc, auprc=auprc, loss=output.loss)


def main() -> None:
    args = parse_args()
    dataset = load_mat_dataset(args.dataset, args.data_dir)

    results: List[TrialResult] = []
    for trial in trange(args.n_trials, desc="Trial", position=0, leave=True):
        results.append(run_one_trial(args, trial, dataset))

    aucs = np.array([r.auc for r in results], dtype=float)
    auprcs = np.array([r.auprc for r in results], dtype=float)

    if args.result_csv is not None:
        data = {
            "datetime": datetime.now().isoformat(sep=" ", timespec="minutes"),
            "dataset": dataset.name,
            "trials": args.n_trials,
            "auc":f"{aucs.mean()*100:.2f}±{aucs.std(ddof=1)*100 if len(aucs) > 1 else 0.0:.2f}({aucs.max()*100:.2f})",
            "aucprc":f"{auprcs.mean()*100:.2f}±{auprcs.std(ddof=1)*100 if len(auprcs) > 1 else 0.0:.2f}({auprcs.max()*100:.2f})",
        }
        DataFrame([data]).to_csv(args.result_csv, index=False, mode="a", header=not os.path.exists(args.result_csv))

    print("=" * 80)
    print("NSReg finished")
    print(f"dataset      : {dataset.name}")
    print(f"num_nodes    : {dataset.num_nodes}")
    print(f"num_edges    : {dataset.edge_index.size(1)}")
    print(f"num_features : {dataset.x.size(1)}")
    print(f"anomaly_rate : {float(dataset.y.float().mean()):.6f}")
    print(f"n_trials     : {args.n_trials}")
    print(f"seed_base    : {args.seed}")
    print("-" * 80)
    for r in results:
        print(
            f"trial={r.trial:02d} seed={r.seed:<6d} auc={r.auc:.6f} auprc={r.auprc:.6f} loss={r.loss:.6f}"
        )
    print("-" * 80)
    print(f"AUC: {aucs.mean():.6f} ± {aucs.std(ddof=1) if len(aucs) > 1 else 0.0:.6f}")
    print(f"AUPRC: {auprcs.mean():.6f} ± {auprcs.std(ddof=1) if len(auprcs) > 1 else 0.0:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
