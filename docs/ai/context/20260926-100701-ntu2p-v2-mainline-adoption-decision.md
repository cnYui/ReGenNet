# NTU2P v2 设为新主线：用户确认记录

日期：2026-09-26。结果文档：`docs/ai/context/20260926-005610-ntu2p-v2-final-result.md`（PR #17 已合并）。

## 决定

用户确认采纳 v2 最终方案 `A6-F-A5f0.05-A4-GH0.5` 作为 NTU2P 双人动作预测的新主线。它取代 rootdct（`ntu2p_residual_refiner_xyz_artic_rootdct_s2_5_dct1_root01_*`）。

- 主线 checkpoint：`save/forecasting/ntu120_label/ntu2p_v2_A6-F-A5f0.05-A4-GH0.5_s{0,1,2}_10000/ema/model000010000.pt`；需要单个权重做可视化时用 s0。
- 汇报数字（test 1253，3 seed，EMA 终点）：mpjpe **0.1810 ± 0.0004**，xyz_mse 0.02773，xyz_mae 0.08777，gate 3/3。
  - 这组数字替换 rootdct 的 0.1845 与 const-EMA 的 0.1882。
- 用户同时知悉并接受结果文档第 6 节记录的偏离：步态规则中"滑行帧比 ≤ 1.10 × 参照"这条护栏未通过，采纳依据是审查图与多数自然度指标。相对 rootdct，test 上的滑行帧比只高 3%。

## 对协议的影响

- **后续实验的对照与底座**：改为 v2 主线。
  - 训练入口：`train/train_ntu2p_v2.py`；
  - 评估：`eval/eval_ntu2p_v2.py`，含自然度与步态相位指标；
  - 驱动：`scripts/run_ntu2p_v2_screen.py`。
- **旧 refiner 系**（A0/const-EMA、rootdct）的 checkpoint 与结论保留，作为历史参照。
- **筛选步数**：root/inter Stage A 与 v2 都改用 10000 step × 3 seed 筛选。原因是 5000 step 的 seed 噪声约为 0.8% 采纳阈值的 3 倍（见 `docs/ai/context/20260925-142304-ntu2p-v2-implementation-and-review-result.md`）。最终数字仍为 10000 step × 3 seed 的 EMA 终点。
- **评估时的自然度检查**：新方案除 L2 与摆动 gate 外，还须报告两类指标：
  - 自然度：骨长误差、步行 GT 站定帧脚速、滑行帧比；
  - 步态：相位相关、步数。

  同时须看审查图，工具为 `sample/render_ntu2p_review_sheet.py` 与 `sample/merge_ntu2p_pred_arrays.py`。平均骨长误差会掩盖个别帧的肢体拉伸，不能单独作为自然度证据。

## 下一步候选（未实现）

1. Track B residual diffusion：以 v2 终点为确定性底座，处理起步者的迈步相位多模态和偏慢的步频。
2. 手指损失去重（α=1/15）：把梯度还给身体，针对 test 上滑行帧比略高的问题。
3. 从 train 中留出受试者作选型用 val：当前 val 的 42 个受试者全部出现在 train 中，val 收益约为 test 的两倍。
