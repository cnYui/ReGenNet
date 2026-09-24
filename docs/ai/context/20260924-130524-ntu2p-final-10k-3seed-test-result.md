# NTU2P residual refiner 最终数字（s2_5 × 3 seed × 10000 step）与首次 test 评估结果

计划：`docs/ai/context/20260924-113322-ntu2p-final-10k-3seed-test-eval-plan.md`（协议在启动前冻结，test 结果不回流任何选择）。
驱动：`scripts/run_ntu2p_final_10k.py`；horizon：`scripts/ntu2p_error_vs_horizon.py`。
2026-09-24 11:35–13:04 在 cuda:0 串行完成，无中断。

## 结论

1. **最终数字（3 seed 终点均值 ± 样本标准差）**

   | split | xyz_mse | xyz_mae | mpjpe | 相对 base | 相对 copy-last | gate |
   |---|---:|---:|---:|---:|---:|---|
   | val（198） | 0.02482 ± 0.00059 | 0.08725 ± 0.00061 | **0.1830 ± 0.0016** | −19.2% | −34.1% | 3/3 |
   | test（1253） | 0.03131 ± 0.00038 | 0.09449 ± 0.00080 | **0.1967 ± 0.0015** | −18.0% | −33.6% | 3/3 |

   val 终点落在 Stage 3 平台带（0.1823 ± 0.0023）内，符合预期。
2. **test 上排序和幅度都保持**：三个 seed 在 test 上都同时低于 base 和 copy-last 的三项 L2，摆动 gate 全过（dct_low 0.546 ± 0.045、dct_mid 0.555 ± 0.028、dct_high 0.008，frozen 0.095，GT 0.085）。val → test 的 mpjpe 上升：模型 +7.5%、base +5.8%、copy-last +6.5%。方向一致，说明主要是 test 本身更难（不同受试者）。模型的上升略大于 base，与 Stage 3 观察到的 train/val gap 一致，但相对 base 的收益只从 −19.2% 收窄到 −18.0%。
3. **摆动恢复在 test 上同样成立**：摆动恢复前的 control（`inter001_s0_5000`）test mpjpe 0.2268（相对 base 仅 −5.4%），dct_mid 0.0017，frozen 0.314，不过 gate。最终模型相对 control 的 mpjpe 为 −13.3%，dct_mid 从 0.0017 升到 0.555。"回归到均值"问题及其修复在未见过的受试者上都复现。
4. **训练完全确定**：新 seed 0 run 的 10 个 val checkpoint 与 Stage 3 `s0_20000` 前 10000 step 逐位一致；seed 1 / seed 2 的 1000–5000 step 与 Stage 3 `s1_5000` / `s2_5000` 逐位一致。同 seed 同配置没有 CUDA 非确定性噪声，方差只来自 seed。
5. **终点抖动与 seed 方差同量级**：seed 0 的 10000 step 终点在平台内偏高（val 0.1849），同一 run 的 20000 step 终点 val 为 0.1789、test 为 0.1929（10000 终点 test 为 0.1964，差 −1.8%）。恒定学习率下相邻 checkpoint 的抖动（约 ±0.003）大于 3 seed 的标准差（0.0015）。这**不改变协议**：不能据 test 或 val 改选 20000。它支持下一步做学习率余弦衰减或权重 EMA，以降低终点抖动。单 seed 20000 的数字不再作为主数字引用。

## 逐 seed 终点

| split | seed | xyz_mse | xyz_mae | mpjpe | 相对 base | dct_low | dct_mid | dct_high | frozen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0 | 0.02513 | 0.08790 | 0.18485 | −18.4% | 0.491 | 0.521 | 0.0074 | 0.122 |
| val | 1 | 0.02519 | 0.08715 | 0.18241 | −19.5% | 0.578 | 0.479 | 0.0075 | 0.107 |
| val | 2 | 0.02414 | 0.08668 | 0.18173 | −19.8% | 0.567 | 0.540 | 0.0080 | 0.106 |
| test | 0 | 0.03118 | 0.09423 | 0.19644 | −18.1% | 0.494 | 0.565 | 0.0074 | 0.104 |
| test | 1 | 0.03174 | 0.09539 | 0.19831 | −17.3% | 0.575 | 0.524 | 0.0082 | 0.090 |
| test | 2 | 0.03102 | 0.09386 | 0.19534 | −18.5% | 0.568 | 0.577 | 0.0087 | 0.090 |

## test 对照行

| 方法 | xyz_mse | xyz_mae | mpjpe | dct_mid | frozen | gate |
|---|---:|---:|---:|---:|---:|---|
| 最终模型（3 seed 均值） | **0.03131** | **0.09449** | **0.19670** | 0.555 | 0.095 | 3/3 |
| base（冻结独立单人，`cuda_retrain_s0_5000@5000`） | 0.04661 | 0.11340 | 0.23974 | 0.0016 | 0.304 | — |
| copy-last | 0.06902 | 0.13713 | 0.29604 | 0 | 1.000 | — |
| 参考：摆动恢复前 control `inter001_s0_5000@5000` | 0.04339 | 0.10773 | 0.22683 | 0.0017 | 0.314 | ✗ |
| 参考：Stage 3 `s0_20000@20000`（单 seed） | 0.03094 | 0.09286 | 0.19291 | 0.519 | 0.106 | ✓ |

注：base 行来自同一评估 JSON，是 residual refiner 实际冻结使用的 `cuda_retrain_s0_5000` 权重。它和独立单人 baseline 结果文档里的 test 0.2442 不是同一个 run（那是 `cuda_full_s0_5000@3000`），同口径对比以本表为准。

## 误差随 horizon（test 1253，单位 m）

`results/forecasting/ntu120_label/ntu2p_final_10k/error_vs_horizon_test.{png,json}`，最终模型为 3 seed 均值 ± 标准差。

| 第 N 帧 | 指标 | copy-last | base | control | **最终模型** |
|---:|---|---:|---:|---:|---:|
| 10 (0.5 s) | mpjpe / root / local | 0.159 / 0.071 / 0.125 | 0.150 / 0.067 / 0.119 | 0.146 / 0.064 / 0.118 | **0.133 / 0.065 / 0.108** |
| 20 (1.0 s) | mpjpe / root / local | 0.284 / 0.159 / 0.188 | 0.244 / 0.130 / 0.172 | 0.235 / 0.122 / 0.170 | **0.197 / 0.113 / 0.145** |
| 30 (1.5 s) | mpjpe / root / local | 0.350 / 0.222 / 0.204 | 0.277 / 0.166 / 0.184 | 0.261 / 0.150 / 0.180 | **0.225 / 0.137 / 0.163** |
| 50 (2.5 s) | mpjpe / root / local | 0.475 / 0.364 / 0.200 | 0.362 / 0.268 / 0.196 | 0.336 / 0.240 / 0.191 | **0.287 / 0.203 / 0.182** |
| 50 (2.5 s) | 双人相对 root | 0.435 | 0.286 | 0.284 | **0.236** |

seed 间标准差在所有关键帧 ≤ 0.005。Stage 3 在 val 上的两条结构性结论在 test 上复现：

- root 误差近似线性增长、不饱和：第 50 帧 0.203，相对 copy-last（0.364）追回约 44%。这是长程误差中可修的部分。
- 局部姿态误差在第 25 帧后饱和于 0.16–0.18。第 50 帧与 base / copy-last 的差距收窄到 0.014 / 0.018，这部分是多模态的，交给 Track B。
- 双人相对 root 误差在第 50 帧比 base 低 17%（0.236 vs 0.286），而 control 与 base 几乎相同（0.284）。这说明相对位置的改善来自摆动恢复后的新损失组合，不能单独归因于跨人 attention。

## 对 AGENTS.md 与汇报的影响

- NTU2P residual refiner 的主数字改为本文的 3 seed × 10000 终点：val mpjpe 0.1830 ± 0.0016，test 0.1967 ± 0.0015。
- 主线 checkpoint 为三个终点 `save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_final_s2_5_dct1_root01_s{0,1,2}_10000/model000010000.pt`；需要单一权重做可视化时用 seed 0（与之前的视频索引同 seed）。
- 汇报计划（`20260827-184636-two-person-motion-prediction-presentation-plan.md`）中的 MPJPE 0.21527 是摆动恢复前的 val 数字，应替换为本表的 val / test 数字。

## 下一步候选（未实现）

1. 学习率余弦衰减或权重 EMA：降低终点抖动（本轮证据：同 run 相邻终点差 ±0.003 ≥ seed 标准差）；需先实现并做关闭时的等价性测试。
2. root 轨迹 DCT 头 + 新损失下重测 `inter_loss_weight`：针对线性增长的 root 误差；5000 step 筛选，胜出者补 10000 × 3 seed。
3. Track B residual diffusion：以本组终点为确定性底座，处理 ≥1.5 s 的姿态多模态。

## 文件

- run：`save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_final_s2_5_dct1_root01_s{0,1,2}_10000/`（`eval_val_*.json` × 10、`eval_test_000010000.json`）
- 参考 test JSON：`.../ntu2p_residual_refiner_xyz_inter001_s0_5000/eval_test_000005000.json`、`.../ntu2p_residual_refiner_xyz_artic_s3_s2_5_dct1_root01_s0_20000/eval_test_000020000.json`
- 汇总：`results/forecasting/ntu120_label/ntu2p_final_10k/{summary.md,summary.json,final.md,final.json,driver.log,error_vs_horizon_test.png,error_vs_horizon_test.json}`
