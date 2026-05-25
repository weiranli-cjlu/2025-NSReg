# 2025-NSReg 简化复现版本

## Setup
```shell
uv venv -p 3.12
uv pip install torch==2.11.0 torch_geometric scikit-learn tqdm optuna pandas --torch-backend=cu128
```

本版本将原仓库的 `bash run_scripts/... + yaml config` 运行方式改为统一命令行：

```bash
python run.py --dataset ACM --n_trials 5 --lr 0.001
```

默认数据集目录：

```bash
~/datasets/GAD/mat
```

支持数据集名称直接使用文件名去掉 `.mat` 后的名称，例如：

```bash
python run.py --dataset ACM
python run.py --dataset Amazon
python run.py --dataset YelpChi
python run.py --dataset t_finance
```

## 主要参数

```bash
python run.py \
  --dataset ACM \
  --data_dir ~/datasets/GAD/mat \
  --n_trials 5 \
  --seed 42 \
  --device cuda \
  --lr 0.001 \
  --weight_decay 0.0 \
  --epochs 200 \
  --hidden_dim 64 \
  --emb_dim 64 \
  --n_layers 2 \
  --dropout 0.3 \
  --nsreg_weight 1.0 \
  --train_ratio 0.4 \
  --num_train_anomaly 10 \
  --balanced_loss
```

## 参数含义

| 参数 | 说明 | 默认值 |
|---|---|---:|
| `--dataset` | `.mat` 数据集名称 | 必填 |
| `--data_dir` | 数据集目录 | `~/datasets/GAD/mat` |
| `--n_trials` | 重复实验次数 | `5` |
| `--seed` | 初始随机种子，第 `i` 次 trial 使用 `seed+i` | `42` |
| `--lr` | 学习率 | `0.001` |
| `--weight_decay` | 权重衰减 | `0.0` |
| `--epochs` | 训练轮数 | `200` |
| `--hidden_dim` | GraphSAGE 隐藏层维度 | `64` |
| `--emb_dim` | 节点嵌入维度 | `64` |
| `--n_layers` | GraphSAGE 层数 | `2` |
| `--dropout` | dropout | `0.3` |
| `--nsreg_weight` | Normal Structure Regularisation 损失权重 | `1.0` |
| `--train_ratio` | 作为有标签正常点训练的正常节点比例 | `0.4` |
| `--num_train_anomaly` | 作为 seen anomaly 训练的异常节点数 | `10` |
| `--balanced_loss` | 是否使用 BCE 类别权重 | 关闭 |

## 原作者配置改为命令行示例

原仓库 README 使用：

```bash
bash run_scripts/mag_cs/run.sh run meta_mag_cs
```

现在改为：

```bash
python run.py --dataset ACM --n_trials 5 --lr 0.001 --weight_decay 0.0 --epochs 200 --hidden_dim 64 --emb_dim 64 --n_layers 2 --dropout 0.3 --nsreg_weight 1.0 --train_ratio 0.4 --num_train_anomaly 10 --seed 42
```

可为不同数据集手动调参，例如：

```bash
python run.py --dataset cora --n_trials 10 --lr 0.001 --weight_decay 5e-4 --epochs 300 --hidden_dim 128 --emb_dim 128 --dropout 0.3 --nsreg_weight 0.5 --seed 42
python run.py --dataset citeseer --n_trials 10 --lr 0.001 --weight_decay 5e-4 --epochs 300 --hidden_dim 128 --emb_dim 128 --dropout 0.4 --nsreg_weight 0.5 --seed 42
python run.py --dataset Amazon --n_trials 10 --lr 0.0005 --weight_decay 1e-5 --epochs 200 --hidden_dim 64 --emb_dim 64 --dropout 0.3 --nsreg_weight 1.0 --seed 42
python run.py --dataset YelpChi --n_trials 10 --lr 0.0005 --weight_decay 1e-5 --epochs 200 --hidden_dim 64 --emb_dim 64 --dropout 0.3 --nsreg_weight 1.0 --seed 42
```

## 文件说明

```text
run.py        # 主入口，负责命令行参数、多 trial、随机种子、最终指标打印
data_utils.py # 读取 ~/datasets/GAD/mat/*.mat，并转为 PyG Data
model.py      # 精简版 NSReg：GraphSAGE + BCE + normal structure regularisation
README.md     # 命令行说明与配置示例
```

## 注意

`.mat` 文件字段通常包括 `Network/Attributes/Label` 或 `adj/features/label`。`data_utils.py` 已兼容常见字段名：

- 特征：`x`, `X`, `features`, `Attributes`, `attr`, `node_feat`, `node_features`
- 标签：`y`, `Y`, `label`, `labels` `Label`, `gnd`, `truth`, `Class`
- 图：`edge_index`, `edges`, `edge`, `adj`, `A`, `network`, `Network`, `graph`

如果某个数据集字段名不同，只需要在 `data_utils.py` 顶部的 key 列表中添加字段名。

## 快速调参

```bash
python tune_optuna.py --dataset ACM --n_trials 50 --eval_trials 3 --device cuda
```

含义：

- `--n_trials 50`：Optuna 搜索 50 组超参数。
- `--eval_trials 3`：每组超参数调用 `run.py --n_trials 3`，减少单次偶然性。
- 最终会输出推荐的 `python run.py ...` 命令。
- 调参记录保存在 `tune_results/<dataset>/`。

## 推荐流程

先粗调：

```bash
python tune_optuna.py \
  --dataset ACM \
  --n_trials 50 \
  --eval_trials 2 \
  --objective auc_auprc \
  --device cuda
```

再用输出的最佳参数做稳定复现实验：

```bash
python run.py \
  --dataset ACM \
  --n_trials 10 \
  --seed 42 \
  --device cuda \
  --lr <best_lr> \
  --weight_decay <best_weight_decay> \
  --epochs <best_epochs> \
  --hidden_dim <best_hidden_dim> \
  --emb_dim <best_emb_dim> \
  --n_layers <best_n_layers> \
  --dropout <best_dropout> \
  --nsreg_weight <best_nsreg_weight> \
  --train_ratio <best_train_ratio> \
  --num_train_anomaly <best_num_train_anomaly>
```

## 大数据集建议

YelpChi、Amazon-all、Flickr 等较大图先降低搜索成本：

```bash
python tune_optuna.py \
  --dataset YelpChi \
  --n_trials 30 \
  --eval_trials 1 \
  --epochs_choices 100,200 \
  --hidden_dim_choices 64,128 \
  --emb_dim_choices 64,128 \
  --device cuda
```

找到较好范围后再提高 `--eval_trials` 和 `run.py --n_trials`。

## 实验
```shell
python run.py --result-csv results.csv --n_trials 10 --dataset book --lr 0.0004328 --weight_decay 0.0006351 --epochs 100 --hidden_dim 64 --emb_dim 64 --n_layers 3 --dropout 0.31485 --nsreg_weight 0.1976 --train_ratio 0.2456 --num_train_anomaly 0 

python run.py --result-csv results.csv --n_trials 10 --dataset Disney --lr 0.001037 --weight_decay 0.00048696 --epochs 400 --hidden_dim 128 --emb_dim 256 --n_layers 2 --dropout 0.1192 --nsreg_weight 0.0103888 --train_ratio 0.5077 --num_train_anomaly 10 --balanced_loss

python run.py --result-csv results.csv --n_trials 10 --dataset elliptic --lr 0.0011454 --weight_decay 2.1066e-06 --epochs 400 --hidden_dim 64 --emb_dim 256 --n_layers 2 --dropout 0.01525 --nsreg_weight 0.02107 --train_ratio 0.11571 --num_train_anomaly 20 --balanced_loss

python run.py --result-csv results.csv --n_trials 10 --dataset questions --lr 0.001037 --weight_decay 0.00048696 --epochs 400 --hidden_dim 128 --emb_dim 256 --n_layers 2 --dropout 0.1192 --nsreg_weight 0.0103888 --train_ratio 0.50773 --num_train_anomaly 10 --balanced_loss

python run.py --result-csv results.csv --n_trials 10 --dataset Reddit --lr 0.001145 --weight_decay 2.1066e-06 --epochs 400 --hidden_dim 64 --emb_dim 256 --n_layers 2 --dropout 0.01525 --nsreg_weight 0.02107 --train_ratio 0.11571 --num_train_anomaly 20 --balanced_loss

python run.py --result-csv results.csv --n_trials 10 --dataset twitter --lr 0.0007476 --weight_decay 2.342e-05 --epochs 200 --hidden_dim 64 --emb_dim 64 --n_layers 3 --dropout 0.15527 --nsreg_weight 0.9717775 --train_ratio 0.115256 --num_train_anomaly 20

python run.py --result-csv results.csv --n_trials 10 --dataset weibo --lr 0.0007476 --weight_decay 2.34238e-05 --epochs 200 --hidden_dim 64 --emb_dim 64 --n_layers 3 --dropout 0.15527 --nsreg_weight 0.9717775 --train_ratio 0.25586 --num_train_anomaly 20

python run.py --result-csv results.csv --n_trials 10 --dataset cs

python run.py --result-csv results.csv --n_trials 10 --dataset photo

python run.py --dataset Facebook \
  --result-csv results.csv --n_trials 10 \
  --lr 0.000747599299995651 --weight_decay 2.3423849847112885e-05 --epochs 200 --hidden_dim 64 --emb_dim 64 --n_layers 3 --dropout 0.15526798896001015   --nsreg_weight 0.9717775305059632 --train_ratio 0.2558555380447055   --num_train_anomaly 20

python run.py --dataset cora \
  --result-csv results.csv --n_trials 10 \
  --lr 0.0011454201562599393 --weight_decay 2.106648601704219e-06 --epochs 400 --hidden_dim 64 --emb_dim 256 --n_layers 2 --dropout 0.015251476046457112 --nsreg_weight 0.021070472806578238 --train_ratio 0.11571459284336713 --num_train_anomaly 20 --balanced_loss

python run.py --dataset citeseer \
  --result-csv results.csv --n_trials 10 \
  --lr 0.0003571447962001007 --weight_decay 9.682012086882464e-05 --epochs 200 --hidden_dim 64 --emb_dim 32 --n_layers 1 --dropout 0.2955106162913183 --nsreg_weight 0.03852378615162033 --train_ratio 0.46122605763075264 --num_train_anomaly 50

python run.py --dataset tolokers \
  --result-csv results.csv --n_trials 10 \
  --lr 0.000747599299995651 --weight_decay 2.3423849847112885e-05 --epochs 200 --hidden_dim 64 --emb_dim 64 --n_layers 3 --dropout 0.15526798896001015 --nsreg_weight 0.9717775305059632 --train_ratio 0.2558555380447055 --num_train_anomaly 20

python run.py --dataset ACM \
  --result-csv results.csv --n_trials 10 \
  --lr 0.000747599299995651 --weight_decay 2.3423849847112885e-05 --epochs 200 --hidden_dim 64 --emb_dim 64 --n_layers 3 --dropout 0.15526798896001015 --nsreg_weight 0.9717775305059632 --train_ratio 0.2558555380447055 --num_train_anomaly 20

python run.py --dataset Flickr \
  --result-csv results.csv --n_trials 10 \
  --lr 0.0019985233361873534 --weight_decay 2.564140900826996e-06 --epochs 300 --hidden_dim 64 --emb_dim 64 --n_layers 3 --dropout 0.24562741040513253 --nsreg_weight 1.5251439600938936 --train_ratio 0.17936967752434835 --num_train_anomaly 50

python run.py --dataset BlogCatalog \
  --result-csv results.csv --n_trials 10 \
  --lr 0.0010158998480836816 --weight_decay 2.1422353438303595e-06 --epochs 300 --hidden_dim 256 --emb_dim 256 --n_layers 3 --dropout 0.21453193837686388 --nsreg_weight 3.9161688872864286 --train_ratio 0.4815496349001899 --num_train_anomaly 50

python run.py --dataset YelpChi \
  --result-csv results.csv --n_trials 10 \
  --lr 0.000747599299995651 --weight_decay 2.3423849847112885e-05 --epochs 200 --hidden_dim 64 --emb_dim 64 --n_layers 3 --dropout 0.15526798896001015 --nsreg_weight 0.9717775305059632 --train_ratio 0.2558555380447055 --num_train_anomaly 20
```