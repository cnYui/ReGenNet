# NTU2P root 轨迹 DCT 头与 inter 权重重测：实现与启动前验证结果

计划：`docs/ai/context/20260925-110531-ntu2p-root-dct-head-inter-weight-plan.md`（含启动前分析）。本文记录实现、启动前验证与 Stage A 的启动状态；实验结论另写结果文档。

## 实现

| 文件 | 改动 |
|---|---|
| `model/forecasting_ntu2p_residual_xyz.py` | 新增 `build_anchored_dct_basis`，即 φ_k(t) − φ_k(0)，k = 1..K。构造参数新增 `root_head_mode`（`none` / `dct`，默认 `none`）与 `root_dct_k`（默认 5），写入 `config()`，加载时缺省按 `none` / 5。`dct` 时在 `__init__` 末尾创建 root 头：`Linear(60, 256)` 编码观测 root 运动学，与 decoder 输出的时间均值相加 → LayerNorm → Linear → GELU → `Linear(256, K·3)`（零初始化）。输出系数乘锚定基得到每人的整体平移，加到全部 55 个关节。 |
| `train/train_ntu2p_residual_refiner_xyz.py` | 新增 `--root_head_mode`、`--root_dct_k`，传给模型构造。 |
| `scripts/run_ntu2p_root_inter.py`（新增） | Stage A 串行训练 `rootdct`、`inter10`、`inter3`、`inter1`，各 3 seed × 10000 step，均开 EMA。每个 run 训完立即评估 raw 与 `ema/` 的 val。汇总输出 `summary.md`（逐 checkpoint）与 `decision.md/json`（L2、摆动、J、root 与关系指标，以及预登记的采纳判断）。`--stage_b` 在 root 与某个 inter 权重都采纳时训练组合；`--test_config` 只评估选中配置的 test。统计复用 LR/EMA 驱动的 `_stats`，训练与评估复用 Stage 1/3 驱动。 |

取舍：

- **root 头在 `torch.random.fork_rng(devices=[])` 内初始化**：第一版没有这样做，root 头初始化多消耗了全局 CPU 随机数，同 seed 下 DataLoader 的打乱顺序随之改变，step 1 的 loss 就与对照不同（6.13 vs 4.10），无法配对比较。fork 之后：
  - root 头 run 的 step 1 loss 与对照逐位相同（4.103426）；
  - step 2、3 只因 root 头的第一次更新产生小差异（5.952 vs 5.930、7.397 vs 7.377）；
  - 数据顺序与 dropout 保持一致。
- inter 配置不加参数，天然与同 seed 对照配对。实现方式是在公共参数的 `--inter_loss_weight 0.01` 之后追加覆盖值，argparse 取最后一次出现的值。
- 不做"解耦"变体，理由见计划。

## 启动前验证（全部通过）

1. **默认参数逐位不变**：新代码用默认参数（`none`）训练 seed 0 × 1000 step，`model000001000.pt` 与已有 final s0 的同名权重逐张量 `torch.equal`，249/249 相同。
2. **零初始化等价**：`dct` 模式加载无头模型（const-EMA s0 `ema/model000010000.pt`）的全部权重后，缺失的只有 `root_*` 键，前向输出与原模型逐位相同。
3. **首帧与整体平移**：把 `root_out` 随机化后：
   - 首帧偏移恰为 0.0；
   - 55 个关节上的偏移相同（最大差 2.4e-7，来自浮点减法）；
   - 后续帧偏移非零。
4. **参数量**：root 头 85,775，模型可训练参数合计 7,153,845。
5. **保存加载往返**：`root_head_mode` / `root_dct_k` 正确恢复，输出逐位相同；旧 checkpoint 仍按 `none` 加载，不含 `root_*` 键。
6. **训练冒烟**：`--root_head_mode dct --ema_decay 0.999` 训练 300 step，loss 有限，`root_out` 权重离开零点（最大绝对值 0.023）。EMA@300 评估的 first_step_error 为 0.0，指标全部有限。
7. **inter 覆盖生效**：追加 `--inter_loss_weight 10.0` 后，run 目录 `args.json` 中为 10.0。
8. **驱动**：
   - dry run 核对 12 个训练命令，额外参数分别为 `--ema_decay 0.999 --root_head_mode dct --root_dct_k 5` 与 `--ema_decay 0.999 --inter_loss_weight {10.0,3.0,1.0}`；
   - 只有对照完整时，汇总输出对照一行、不采纳任何配置；
   - 用构造数据测试判断逻辑的 8 个分支均正确：都不采纳、只采纳 root、被 mse 阻断、被 gate 阻断、inter 并列取较小权重、待跑 Stage B、组合最优、组合不如单因子。
9. `python3 scripts/check_repo_conventions.py --base fork/main`（含 CI 同版本 ruff）在提交前运行。

快检产物放在会话 scratchpad，不入库。

## 实验状态

- 2026-09-25 11:16 在 cuda:0 以 `setsid nohup` 后台启动 `scripts/run_ntu2p_root_inter.py`（Stage A），日志 `results/forecasting/ntu120_label/ntu2p_root_inter/driver.log`。12 个 run 约 6.4 h。
- run 目录：`save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_{rootdct,inter10,inter3,inter1}_s2_5_dct1_root01_s{0,1,2}_10000/`，EMA 权重在各自的 `ema/` 下。
- 完成后按计划的预登记规则判断。只有当 root 与某个 inter 权重都采纳时，才运行 `--stage_b`。之后只对最终采纳的配置运行 `--test_config <配置>`，并画 test horizon 曲线。
