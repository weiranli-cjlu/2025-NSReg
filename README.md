# 2025-NSReg 简化复现版本

## Setup
```shell
uv venv -p 3.12
uv pip install torch==2.11.0 torch_geometric scikit-learn tqdm --torch-backend=cu128
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

## 已移除内容

- 不保存模型 checkpoint。
- 不保存中间结果。
- 不使用 `saved_index`。
- 不使用 shell 脚本启动。
- 不使用 yaml 配置。
- 不使用 tensorboard、plot、TSNE、PCA 可视化等与最终指标无关的代码。
- 训练过程中不打印 epoch 日志，只在所有 trial 结束后打印最终结果。

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
- 标签：`y`, `Y`, `label`, `labels`, `gnd`, `truth`, `Class`
- 图：`edge_index`, `edges`, `edge`, `adj`, `A`, `network`, `Network`, `graph`

如果某个数据集字段名不同，只需要在 `data_utils.py` 顶部的 key 列表中添加字段名。
