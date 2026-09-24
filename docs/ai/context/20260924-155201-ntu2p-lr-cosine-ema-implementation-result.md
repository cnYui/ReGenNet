# NTU2P residual refiner 学习率余弦衰减与权重 EMA：实现与等价性验证结果

计划：`docs/ai/context/20260924-154319-ntu2p-lr-cosine-ema-plan.md`。本文记录实现、启动前的验证，以及 2×2 实验的启动状态；实验结论另写结果文档。

## 实现

| 文件 | 改动 |
|---|---|
| `train/train_ntu2p_residual_refiner_xyz.py` | 新增 `--lr_schedule {constant,cosine_tail}`、`--lr_decay_start_frac`（默认 0.8）、`--lr_min`（默认 3e-5）、`--ema_decay`（默认 0，即关闭）。`constant` 时不改 param_group 的 lr。EMA 是 deepcopy 出来的影子模型，在 `optimizer.step()` 后更新：可训练参数做滑动平均，冻结的 base 参数与 buffer 直接复制。EMA checkpoint 与原始 checkpoint 同格式，存到 `save_dir/ema/model{step}.pt`，不存 optimizer。训练日志新增 `lr` 字段。 |
| `scripts/run_ntu2p_articulation_stage3.py` | `_train` 增加 `extra_args` 参数，默认为空，已有调用不变。 |
| `scripts/run_ntu2p_final_10k.py` | `_split_rows` / `_write_final` 增加 `label` / `summary_dir` 参数，供新驱动复用。顺带修复 PR #9 审查的 minor：参考权重缺失且没有已有 test JSON 时给出明确报错。 |
| `scripts/ntu2p_error_vs_horizon.py` | 顺带修复 PR #9 审查的 minor：docstring 中 copy-last 的描述从"外推"改为"重复观测末帧"。 |
| `scripts/run_ntu2p_lr_ema.py`（新增） | 2×2 驱动：串行训练 `ema`（恒定学习率 + EMA）与 `cosema`（余弦尾部衰减 + EMA）两组 × seed 0/1/2 × 10000 step。每个 run 训完立即评估 raw 与 `ema/` 的全部 val checkpoint；最后读取已有 const-raw（final 10k），生成 `summary.md`（逐 checkpoint 曲线）、`equivalence.md`（逐位等价核对）、`decision.md/json`（J / L / S / gate 与预登记判断）。`--test_variant` 单独调用，只评估选中变体的 test。 |

取舍：

- EMA 放在子目录，而不是改文件名前缀。原因是 Stage 1 驱动用 `model(\d+)\.pt` 匹配 checkpoint，`model_ema*.pt` 会在同一 glob 里匹配失败。子目录可以直接复用 `_evaluate`。
- EMA 用固定 decay，不做 warmup。timm 式 warmup `(1+t)/(10+t)` 要到第 8990 步才达到 0.999，会让终点的 EMA 实际窗口变短、名不副实。固定 decay 下，10000 step 时初始权重的残留占比约 4.5e-5；≤3000 step 的 EMA checkpoint 偏向初始权重，只画曲线用。
- 余弦只衰减尾部 20%，前 8000 step 可与恒定学习率 run 逐位对照，这就是现成的等价性测试。

## 启动前验证（全部通过）

1. **默认参数逐位不变**：新代码用默认参数跑 seed 0 × 1000 step，`model000001000.pt` 与已有 `final_s2_5_dct1_root01_s0_10000/model000001000.pt` 逐张量 `torch.equal`，249/249 相同，最大差 0.0。
2. **开启 EMA 不扰动训练**：`--ema_decay 0.999` 跑 1000 step，原始权重同样 249/249 逐位相同。
3. **EMA 权重结构正确**：EMA checkpoint 中 100 个 `base_model.*` 张量与原始权重完全相同（冻结参数被复制），其余 149 个可训练张量全部不同。`ramp` / `future_sin_pos` 是非持久 buffer，不进 state_dict，加载时按 config 重建。EMA@1000 可被评估脚本直接加载，val mpjpe 0.2173（raw@1000 为 0.2158；1000 步时 EMA 仍偏向初始权重，符合预期）。
4. **学习率调度取值**（10000 step）：第 0 / 7999 / 8000 次更新为 3e-4，第 8500 次 2.60e-4，第 9000 次 1.65e-4，第 9500 次 6.95e-5，第 9999 次 3.0e-5。
5. **驱动 dry run**：6 个训练命令与 final 10k 相比，只多出 `save_dir` 和额外参数（`--ema_decay 0.999`，或 `--lr_schedule cosine_tail --lr_decay_start_frac 0.8 --lr_min 3e-5 --ema_decay 0.999`）。对照组统计与计划一致：J 0.00313，终点 mpjpe 0.18300 ± 0.00164，gate 3/3。
6. `python3 scripts/check_repo_conventions.py --base fork/main`（含 CI 同版本 ruff 0.16.8）通过。

快检产物放在会话 scratchpad，不入库。

## 实验状态

- 2026-09-24 15:50 在 cuda:0 以 `setsid nohup` 后台启动 `scripts/run_ntu2p_lr_ema.py`，日志 `results/forecasting/ntu120_label/ntu2p_lr_ema/driver.log`。6 个 run 约 3 h。
- run 目录：`save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_{ema,cosema}_s2_5_dct1_root01_s{0,1,2}_10000/`，EMA 权重在各目录的 `ema/` 下。
- 驱动可断点续跑：中断后直接重跑同一命令即可。已有终点权重的 run 会跳过训练，已有评估 JSON 的 checkpoint 会跳过评估。
- 完成后按计划的预登记规则判断。只有被采纳的变体才运行 `--test_variant <变体>` 评估 test，结论写入结果文档与 AGENTS.md。
