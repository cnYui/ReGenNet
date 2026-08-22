# NTU 双人 diffusion 训练 CUDA 强制策略设计

## 背景

最近一次 `L_dm + L_inter` exploratory 训练的 `args.json` 记录为：

```text
device=cpu
```

训练入口 `train/train_ntu_2p_forecasting_diffusion.py` 原先在 `torch.cuda.is_available()` 为 false 时静默回退到 CPU。因此 GPU 未暴露给运行环境时，5000-step 训练仍会继续执行，导致每个 step 的 SMPL-X FK 训练速度很慢，也难以及时发现设备错误。

当前 `regennet` 环境已确认：

```text
torch.cuda.is_available()=true
device_count=1
device0=NVIDIA GeForce RTX 3080
```

## 目标

该训练入口必须运行在 CUDA 上；CUDA 不可用、请求 CPU、或请求不存在的 GPU 编号时，必须在创建数据加载器和模型前明确报错。

## 最小修改方案

1. 新增 `--device` 参数，默认 `cuda:0`。
2. 用单一设备解析函数验证请求为 CUDA、CUDA 可用且设备索引有效，并调用 `torch.cuda.set_device` 固定当前进程设备。
3. 设备验证成功后记录设备名称、CUDA/Torch 版本和可见设备数到 `args.json` 与 checkpoint。
4. 保留现有 `resume_checkpoint`、模型、数据、loss 和训练协议；不修改采样/评估入口。

## 非目标

- 不自动将 CPU 任务迁移或重跑到 GPU。
- 不改变 batch size、loss 权重、训练步数或 `L_inter` 实现。
- 不将这次 exploratory 实验改写为正式 stage 2。

## 验证

1. 编译训练文件。
2. 检查 `--help` 包含 `--device`。
3. 在当前 `regennet` Python 环境解析 `cuda:0`，并确认最小 tensor 位于 RTX 3080。
4. 验证 `cpu` 请求会明确失败，避免静默回退。

## 后续运行约束

恢复或重新启动长训练时必须显式传入：

```text
--device cuda:0
```

启动日志与 `args.json` 必须显示 `device=cuda:0` 和对应 GPU 名称；否则不得开始正式或探索性全量训练。
