# -*- coding: utf-8 -*-
"""
Optuna hyperparameter tuning script for the simplified 2025-NSReg repo.

Place this file in the repository root, then run for example:

    python tune_optuna.py --dataset ACM --n_trials 50 --eval_trials 3 --device cuda

The script calls run.py as a subprocess, so it is compatible with the current
command-line interface:

    python run.py --dataset xx --n_trials xx --lr xx ...

It does not save model checkpoints. It only saves tuning records:
    tune_results/<dataset>/study.db
    tune_results/<dataset>/trials.csv
    tune_results/<dataset>/best_params.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "Optuna is not installed. Install it with:\n"
        "  pip install optuna pandas\n"
        "or:\n"
        "  uv pip install optuna pandas"
    ) from exc


AUC_RE = re.compile(r"^AUC:\s*([0-9]*\.?[0-9]+)\s*±\s*([0-9]*\.?[0-9]+)", re.MULTILINE)
AUPRC_RE = re.compile(r"^AUPRC:\s*([0-9]*\.?[0-9]+)\s*±\s*([0-9]*\.?[0-9]+)", re.MULTILINE)


def parse_run_output(stdout: str) -> Dict[str, float]:
    auc_match = AUC_RE.search(stdout)
    auprc_match = AUPRC_RE.search(stdout)
    if auc_match is None or auprc_match is None:
        tail = "\n".join(stdout.strip().splitlines()[-40:])
        raise RuntimeError(
            "Failed to parse final AUC/AUPRC from run.py output. "
            "Expected lines like:\n"
            "  AUC: 0.xxxxxx ± 0.xxxxxx\n"
            "  AUPRC: 0.xxxxxx ± 0.xxxxxx\n\n"
            f"Output tail:\n{tail}"
        )
    return {
        "auc_mean": float(auc_match.group(1)),
        "auc_std": float(auc_match.group(2)),
        "auprc_mean": float(auprc_match.group(1)),
        "auprc_std": float(auprc_match.group(2)),
    }


def run_nsreg_once(args: argparse.Namespace, params: Dict[str, Any], optuna_trial_id: int) -> Dict[str, float]:
    run_py = Path(args.run_py).expanduser().resolve()
    if not run_py.exists():
        raise FileNotFoundError(f"run.py not found: {run_py}")

    # 每组 Optuna 参数使用不同 seed base；run.py 内部会继续使用 seed+i 做多 trial。
    seed = args.seed + optuna_trial_id * args.seed_stride

    cmd: List[str] = [
        sys.executable,
        str(run_py),
        "--dataset", args.dataset,
        "--data_dir", args.data_dir,
        "--n_trials", str(args.eval_trials),
        "--seed", str(seed),
        "--device", args.device,
        "--lr", str(params["lr"]),
        "--weight_decay", str(params["weight_decay"]),
        "--epochs", str(params["epochs"]),
        "--hidden_dim", str(params["hidden_dim"]),
        "--emb_dim", str(params["emb_dim"]),
        "--n_layers", str(params["n_layers"]),
        "--dropout", str(params["dropout"]),
        "--nsreg_weight", str(params["nsreg_weight"]),
        "--train_ratio", str(params["train_ratio"]),
        "--num_train_anomaly", str(params["num_train_anomaly"]),
    ]
    if params["balanced_loss"]:
        cmd.append("--balanced_loss")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    completed = subprocess.run(
        cmd,
        cwd=str(run_py.parent),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"run.py failed with return code {completed.returncode}.\n"
            f"Command:\n{' '.join(cmd)}\n\nOutput:\n{completed.stdout}"
        )

    metrics = parse_run_output(completed.stdout)
    metrics["seed"] = seed
    return metrics


def suggest_params(trial: optuna.Trial, args: argparse.Namespace) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "lr": trial.suggest_float("lr", args.lr_low, args.lr_high, log=True),
        "weight_decay": trial.suggest_float("weight_decay", args.weight_decay_low, args.weight_decay_high, log=True),
        "epochs": trial.suggest_categorical("epochs", args.epochs_choices),
        "hidden_dim": trial.suggest_categorical("hidden_dim", args.hidden_dim_choices),
        "emb_dim": trial.suggest_categorical("emb_dim", args.emb_dim_choices),
        "n_layers": trial.suggest_categorical("n_layers", args.n_layers_choices),
        "dropout": trial.suggest_float("dropout", args.dropout_low, args.dropout_high),
        "nsreg_weight": trial.suggest_float("nsreg_weight", args.nsreg_weight_low, args.nsreg_weight_high, log=True),
        "train_ratio": trial.suggest_float("train_ratio", args.train_ratio_low, args.train_ratio_high),
        "num_train_anomaly": trial.suggest_categorical("num_train_anomaly", args.num_train_anomaly_choices),
        "balanced_loss": trial.suggest_categorical("balanced_loss", [False, True]),
    }
    if args.tie_hidden_emb:
        params["emb_dim"] = params["hidden_dim"]
    return params


def objective_factory(args: argparse.Namespace):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, args)
        metrics = run_nsreg_once(args, params, trial.number)

        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("command_params", params)

        if args.objective == "auc":
            score = metrics["auc_mean"]
        elif args.objective == "auprc":
            score = metrics["auprc_mean"]
        elif args.objective == "auc_auprc":
            score = args.auc_weight * metrics["auc_mean"] + (1.0 - args.auc_weight) * metrics["auprc_mean"]
        else:
            raise ValueError(f"Unknown objective: {args.objective}")

        if args.std_penalty > 0:
            if args.objective == "auc":
                score -= args.std_penalty * metrics["auc_std"]
            elif args.objective == "auprc":
                score -= args.std_penalty * metrics["auprc_std"]
            else:
                score -= args.std_penalty * (
                    args.auc_weight * metrics["auc_std"] + (1.0 - args.auc_weight) * metrics["auprc_std"]
                )
        return float(score)

    return objective


def parse_int_choices(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_best_command(args: argparse.Namespace, best_params: Dict[str, Any]) -> str:
    parts = [
        "python run.py",
        f"--dataset {args.dataset}",
        f"--data_dir {args.data_dir}",
        f"--n_trials {args.final_trials}",
        f"--seed {args.seed}",
        f"--device {args.device}",
        f"--lr {best_params['lr']}",
        f"--weight_decay {best_params['weight_decay']}",
        f"--epochs {best_params['epochs']}",
        f"--hidden_dim {best_params['hidden_dim']}",
        f"--emb_dim {best_params['emb_dim']}",
        f"--n_layers {best_params['n_layers']}",
        f"--dropout {best_params['dropout']}",
        f"--nsreg_weight {best_params['nsreg_weight']}",
        f"--train_ratio {best_params['train_ratio']}",
        f"--num_train_anomaly {best_params['num_train_anomaly']}",
    ]
    if best_params.get("balanced_loss", False):
        parts.append("--balanced_loss")
    return " \\\n  ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune NSReg hyperparameters with Optuna.")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="~/datasets/GAD/mat")
    parser.add_argument("--run_py", type=str, default="run.py")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_stride", type=int, default=1000)

    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna hyperparameter trials.")
    parser.add_argument("--eval_trials", type=int, default=3, help="NSReg repeated trials per Optuna trial.")
    parser.add_argument("--final_trials", type=int, default=10, help="Recommended n_trials for final command.")
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URI. Default: local sqlite under output_dir.")
    parser.add_argument("--output_dir", type=str, default="tune_results")
    parser.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random"])
    parser.add_argument("--timeout", type=int, default=None, help="Optuna timeout in seconds.")
    parser.add_argument("--objective", type=str, default="auc_auprc", choices=["auc", "auprc", "auc_auprc"])
    parser.add_argument("--auc_weight", type=float, default=0.5, help="Used only when objective=auc_auprc.")
    parser.add_argument("--std_penalty", type=float, default=0.0, help="Subtract std_penalty * metric_std from objective.")

    parser.add_argument("--lr_low", type=float, default=1e-4)
    parser.add_argument("--lr_high", type=float, default=5e-3)
    parser.add_argument("--weight_decay_low", type=float, default=1e-7)
    parser.add_argument("--weight_decay_high", type=float, default=1e-3)
    parser.add_argument("--epochs_choices", type=parse_int_choices, default=parse_int_choices("100,200,300,400"))
    parser.add_argument("--hidden_dim_choices", type=parse_int_choices, default=parse_int_choices("32,64,128,256"))
    parser.add_argument("--emb_dim_choices", type=parse_int_choices, default=parse_int_choices("32,64,128,256"))
    parser.add_argument("--n_layers_choices", type=parse_int_choices, default=parse_int_choices("1,2,3"))
    parser.add_argument("--dropout_low", type=float, default=0.0)
    parser.add_argument("--dropout_high", type=float, default=0.6)
    parser.add_argument("--nsreg_weight_low", type=float, default=1e-2)
    parser.add_argument("--nsreg_weight_high", type=float, default=10.0)
    parser.add_argument("--train_ratio_low", type=float, default=0.1)
    parser.add_argument("--train_ratio_high", type=float, default=0.6)
    parser.add_argument("--num_train_anomaly_choices", type=parse_int_choices, default=parse_int_choices("0,5,10,20,50"))
    parser.add_argument("--tie_hidden_emb", action="store_true", help="Force emb_dim = hidden_dim.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_name = Path(args.dataset).stem
    output_dir = Path(args.output_dir).expanduser() / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    study_name = args.study_name or f"nsreg_{dataset_name}_{args.objective}"
    storage = args.storage or f"sqlite:///{output_dir / 'study.db'}"

    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    else:
        sampler = optuna.samplers.RandomSampler(seed=args.seed)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    print("=" * 80)
    print("NSReg Optuna tuning")
    print(f"dataset      : {args.dataset}")
    print(f"objective    : {args.objective}")
    print(f"study_name   : {study_name}")
    print(f"storage      : {storage}")
    print(f"optuna trials: {args.n_trials}")
    print(f"eval_trials  : {args.eval_trials}")
    print("=" * 80)

    study.optimize(objective_factory(args), n_trials=args.n_trials, timeout=args.timeout, gc_after_trial=True)

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials_csv = output_dir / "trials.csv"
    trials_df.to_csv(trials_csv, index=False)

    best_params = dict(study.best_trial.params)
    if args.tie_hidden_emb:
        best_params["emb_dim"] = best_params["hidden_dim"]

    best_payload = {
        "dataset": args.dataset,
        "objective": args.objective,
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "best_params": best_params,
        "best_user_attrs": study.best_trial.user_attrs,
        "final_command": build_best_command(args, best_params),
    }

    best_json = output_dir / "best_params.json"
    best_json.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 80)
    print("Best trial")
    print(f"number: {study.best_trial.number}")
    print(f"value : {study.best_value:.6f}")
    print(f"AUC   : {study.best_trial.user_attrs.get('auc_mean', float('nan')):.6f} ± "
          f"{study.best_trial.user_attrs.get('auc_std', float('nan')):.6f}")
    print(f"AUPRC : {study.best_trial.user_attrs.get('auprc_mean', float('nan')):.6f} ± "
          f"{study.best_trial.user_attrs.get('auprc_std', float('nan')):.6f}")
    print("-" * 80)
    print("Best params:")
    print(json.dumps(best_params, indent=2, ensure_ascii=False))
    print("-" * 80)
    print("Recommended final command:")
    print(best_payload["final_command"])
    print("-" * 80)
    print(f"Saved trials     : {trials_csv}")
    print(f"Saved best params: {best_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
