# NTU2P residual refiner 学习率余弦衰减 × 权重 EMA：2×2 结果

计划：`docs/ai/context/20260924-154319-ntu2p-lr-cosine-ema-plan.md`；实现与启动前验证：`docs/ai/context/20260924-155201-ntu2p-lr-cosine-ema-implementation-result.md`；驱动审查跟进：`docs/ai/context/20260924-155510-ntu2p-lr-ema-driver-review-followup.md`。
6 个新 run 于 2026-09-24 15:50–18:59 在 cuda:0 串行完成，无中断。之后用修复后的驱动重跑汇总：等价性改为直接比较权重，已有权重与评估全部跳过。最后按预登记规则只对选中变体评估 test。

## 结论

1. **采纳 const-EMA**（恒定学习率 + `--ema_decay 0.999`，报告 `ema/` 下的终点权重）。val 上：
   - 终点抖动 J 从 0.00313 降到 **0.00041**（−87%）；
   - 终点 mpjpe 从 0.1830 ± 0.0016 降到 **0.1751 ± 0.0014**（−4.3%）；
   - gate 3/3 通过。
   cos-EMA 也满足采纳条件，但 J（0.00095）和水平（0.1761）都不如 const-EMA。按规则取 J 最小者，const-EMA 同时也是更简单的方案。
2. **test（只评估选中变体）**：mpjpe **0.1882 ± 0.0005**，原始权重对照为 0.1967 ± 0.0015。三项 L2 同时改善：xyz_mse −6.5%、xyz_mae −4.2%、mpjpe −4.3%。相对 base −21.5%，相对 copy-last −36.4%；摆动 gate 3/3。test 上的 seed 标准差从 0.0015 缩到 0.0005，主数字的不确定性下降约 3 倍。
3. **余弦尾部衰减单独使用不达标**：cos-raw 的 J 为 0.00209（−33%，未达到减半的要求），逐 seed 为 0.00246 / 0.00029 / 0.00351，不稳定；水平改善到 0.1782。叠加 EMA 后（cos-EMA）抖动和水平都不如只用 EMA。可能的原因（未验证）：学习率衰减后，EMA 窗口内的权重差异变小，平均能抵消的噪声也随之变少。
4. **EMA 不削弱摆动**：dct_mid 在 val 上从 0.514 升到 0.583、test 上从 0.555 升到 0.623。frozen 略升（val 0.111 → 0.119，test 0.095 → 0.101），仍接近 GT（val 0.095 / test 0.085）。权重平均不等于输出平均，没有把预测压回均值。
5. **等价性全部通过**：
   - const-EMA 训练的原始权重在 3 个 seed 的 1000–10000 共 10 个 checkpoint 上，与 const-raw 逐张量 `torch.equal`；
   - cos 训练的原始权重在 1000–8000 的 8 个 checkpoint 上同样逐张量相同。
   新旧两版驱动（比较指标 / 比较权重）给出的判断完全相同。
6. 参考：Stage 3 单 seed 长训 20000 step 的原始权重 test mpjpe 为 0.1929。const-EMA 只训 10000 step 就更好（0.1882）。这说明平台期上的权重平均优于平台上任何单个点，与"终点抖动"的判断一致。

## val 四变体（终点 10000 step，3 seed）

| 变体 | J = mean\|m10k−m9k\| | J 逐 seed | 8k–10k std | 终点 mpjpe | xyz_mse | xyz_mae | dct_mid | frozen | gate |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| const-raw（对照） | 0.00313 | 0.00213 / 0.00506 / 0.00220 | 0.00245 | 0.18300 ± 0.00164 | 0.02482 | 0.08725 | 0.514 | 0.111 | 3/3 |
| **const-EMA** | **0.00041** | 0.00027 / 0.00061 / 0.00035 | 0.00053 | **0.17512 ± 0.00140** | **0.02279** | **0.08373** | 0.583 | 0.119 | 3/3 |
| cos-raw | 0.00209 | 0.00246 / 0.00029 / 0.00351 | 0.00209 | 0.17818 ± 0.00108 | 0.02358 | 0.08517 | 0.599 | 0.114 | 3/3 |
| cos-EMA | 0.00095 | 0.00066 / 0.00134 / 0.00085 | 0.00068 | 0.17614 ± 0.00165 | 0.02308 | 0.08420 | 0.587 | 0.116 | 3/3 |

采纳条件：J ≤ 0.5 × 0.00313、L ≤ 0.1830 + 0.0016、gate 3/3。满足者为 const-EMA 与 cos-EMA，选中 **const-EMA**。

## test（1253，仅选中变体 + 已有对照）

| 方法 | xyz_mse | xyz_mae | mpjpe | 相对 base | 相对 copy-last | dct_low | dct_mid | dct_high | frozen | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **const-EMA（3 seed）** | **0.02928 ± 0.00016** | **0.09049 ± 0.00024** | **0.18816 ± 0.00046** | −21.5% | −36.4% | 0.585 | 0.623 | 0.0068 | 0.101 | 3/3 |
| const-raw（3 seed） | 0.03131 ± 0.00038 | 0.09449 ± 0.00080 | 0.19670 ± 0.00150 | −18.0% | −33.6% | 0.546 | 0.555 | 0.0081 | 0.095 | 3/3 |
| base（冻结独立单人） | 0.04661 | 0.11340 | 0.23974 | 0 | −19.0% | 0.208 | 0.0016 | 0.0005 | 0.304 | — |
| copy-last | 0.06902 | 0.13713 | 0.29604 | +23.5% | 0 | 0 | 0 | 0 | 1.000 | — |

逐 seed（test mpjpe）：const-EMA 0.18811 / 0.18773 / 0.18864；const-raw 0.19644 / 0.19831 / 0.19534。

## 误差随 horizon（test，单位 m，3 seed 均值）

`results/forecasting/ntu120_label/ntu2p_lr_ema/error_vs_horizon_test.{png,json}`

| 第 N 帧 | 指标 | base | const-raw | **const-EMA** |
|---:|---|---:|---:|---:|
| 10 (0.5 s) | mpjpe / root / local | 0.150 / 0.067 / 0.119 | 0.133 / 0.065 / 0.108 | **0.129 / 0.062 / 0.106** |
| 20 (1.0 s) | mpjpe / root / local | 0.244 / 0.130 / 0.172 | 0.197 / 0.113 / 0.145 | **0.189 / 0.108 / 0.141** |
| 30 (1.5 s) | mpjpe / root / local | 0.277 / 0.166 / 0.184 | 0.225 / 0.137 / 0.163 | **0.215 / 0.129 / 0.158** |
| 50 (2.5 s) | mpjpe / root / local | 0.362 / 0.268 / 0.196 | 0.287 / 0.203 / 0.182 | **0.274 / 0.197 / 0.177** |
| 50 (2.5 s) | 双人相对 root | 0.286 | 0.236 | **0.219** |

EMA 在所有 horizon 上一致改善，seed 间的方差带也更窄。结构性结论不变：root 误差仍近似线性增长（第 50 帧 0.197），局部姿态在 1.2 s 后饱和。

## 对协议与 AGENTS.md 的影响

- **最终数字协议更新**：10000 step × 3 seed，训练时加 `--ema_decay 0.999`，报告 `ema/model000010000.pt` 终点的均值 ± 标准差；原始权重不再作主数字。
- **主数字**：val mpjpe 0.1751 ± 0.0014，test mpjpe 0.1882 ± 0.0005。主线 checkpoint 为 `save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_ema_s2_5_dct1_root01_s{0,1,2}_10000/ema/model000010000.pt`；需要单一权重做可视化时用 s0。
- **5000 step 筛选**：EMA 几乎零成本，建议同样开启，并同时报告 raw 与 EMA 终点。5000 step 时初始权重在 EMA 中的残留约 0.7%，这个步数下 EMA 相对 raw 的表现本轮未单独验证。
- 余弦尾部衰减不采纳；开关保留，默认关闭。
- 汇报数字应使用上面的 EMA 主数字（替换摆动恢复前的 0.21527，以及原始权重的 0.1830 / 0.1967）。

## 下一步候选（未实现）

1. root 轨迹 DCT 头 + 新损失下重测 `inter_loss_weight`：针对线性增长的 root 误差。5000 step 筛选，开 EMA。
2. Track B residual diffusion：以 const-EMA 终点为确定性底座，处理 ≥1.5 s 的姿态多模态。

## 文件

- 新 run：`save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_{ema,cosema}_s2_5_dct1_root01_s{0,1,2}_10000/`（原始权重与 `eval_val_*.json`），EMA 在各自的 `ema/` 子目录；const-EMA 的 test JSON 为 `ema/eval_test_000010000.json`。
- 汇总：`results/forecasting/ntu120_label/ntu2p_lr_ema/{summary.md,summary.json,equivalence.md,decision.md,decision.json,final.md,final.json,driver.log,error_vs_horizon_test.png,error_vs_horizon_test.json}`
