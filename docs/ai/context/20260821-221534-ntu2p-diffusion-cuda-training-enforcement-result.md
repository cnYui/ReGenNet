# NTU 双人 diffusion 训练 CUDA 强制策略实现结果

## 已实现

训练入口：

```text
train/train_ntu_2p_forecasting_diffusion.py
```

新增行为：

1. 新增 `--device`，默认值为 `cuda:0`。
2. 训练开始前验证 device 必须是 CUDA、CUDA 必须可用、GPU 索引必须在当前可见范围内，并通过 `torch.cuda.set_device` 固定设备。
3. 禁止静默回退 CPU；传入 `cpu` 或 GPU 不可用时立即抛错。
4. 将 `device`、GPU 名称写入每条训练日志；将 `cuda_runtime` 写入 `args.json` 和 checkpoint。

## 当前环境验证

```text
Python=/home/rpartx3080/.local/micromamba/envs/regennet/bin/python
torch=1.7.1
CUDA=11.0
device=cuda:0
GPU=NVIDIA GeForce RTX 3080
```

完成独立目录的一步完整 CUDA smoke：

```text
save_dir=/tmp/regennet-ntu2p-cuda-smoke-20260821
inter_loss_weight=1.0
train_loss=3.239926
rot_mse=0.665609
inter_loss=2.574317
```

输出验证：

```text
args.json.device=cuda:0
train_log.device=cuda:0
checkpoint.cuda_runtime.device=cuda:0
checkpoint.cuda_runtime.device_name=NVIDIA GeForce RTX 3080
```

## 检查结果

通过：

```text
python -m py_compile train/train_ntu_2p_forecasting_diffusion.py
训练入口 --help 可见 --device
cuda:0 最小 tensor 位于 cuda:0
cpu 请求被明确拒绝
git diff --check
```

## 后续运行

任何后续训练命令必须显式带上：

```text
--device cuda:0
```

运行开始时必须确认日志包含：

```text
device=cuda:0 gpu=NVIDIA GeForce RTX 3080
```

此次修改只修复设备选择和可追溯性，不改变当前 `L_dm-only` 尚未通过 copy-last validation gate 的实验结论。
