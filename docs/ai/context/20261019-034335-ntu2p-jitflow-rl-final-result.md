# NTU2P JiT 流 + 双人接触图 + GRPO（JiT-RL）：完整结果

设计与预登记：`20261018-201532-ntu2p-jit-flow-rl-design-and-plan.md`；修订：`20261018-205749-ntu2p-jitflow-t-distribution-amendment.md`（t 分布）、`20261018-223934-ntu2p-jitflow-grpo-gradient-balance-amendment.md`（GRPO 梯度平衡）；第 1 阶段：`20261018-220255-ntu2p-jitflow-stage1-result.md`。全部为受试者留出 H-val（train 1709 / val 247），未读取 test。

## 决定

**按预登记规则，本轮没有候选被采纳，主线仍为 v2+FD2，未进入原协议复训与 test。**

- 提分：J3/J4 的确定性端点在 3 seed × 10000 step 上稳定优于 FD2（mpjpe −3.30% / −3.22%，逐 seed 全负，xyz_mse −2.6/−2.9%，xyz_mae −3.6/−3.5%），但 jerk 均值为 FD2 的 1.26/1.27 倍，超过预登记护栏（≤1.2×），按规则不采纳。是否接受这一取舍由用户决定；不擅自放宽护栏。
- 交互（"手没碰到就躲"）：没有任何候选或读出通过预登记交互规则；确定性模型仍在接触前平滑后退。
- JiT-RL（GRPO）：端点的变化完全来自 RL 循环里继续训练的端点损失（PG 置 0 的对照与之逐项相同）；奖励（含接触与接触前后退项）对点预测和样本都没有可测增益。

## 3 seed × 10000 step（H-val，EMA 终点，均值 ± 样本标准差）

| 行 | xyz_mse | xyz_mae | mpjpe | Δmpjpe（逐 seed） | jerk | 穿地 | excess_antic | 漏接触 | 判定 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| FD2 参照 | .01938±.00030 | .08412±.00012 | .17220±.00056 | — | 20.40 | .0669 | .0247 | 49.7 | — |
| J3 端点（STG 流） | .01888±.00021 | .08109±.00057 | .16652±.00116 | −3.30%（−2.81/−4.36/−2.71） | 25.76 | .0528 | .0309 | 48.3 | jerk 护栏未过 |
| J4 端点（STG 流 + 接触图 + 跨人因果） | .01882±.00016 | .08119±.00059 | .16666±.00105 | −3.22%（−3.83/−2.90/−2.92） | 25.94 | .0558 | .0374 | 51.0 | jerk 护栏未过 |
| J4N 端点（同结构，无流损失） | .01996±.00040 | .08306±.00069 | .17098±.00160 | −0.71%（−0.05/−0.64/−1.44） | 27.91 | .0628 | .0311 | 48.0 | jerk 护栏未过 |
| J4 中位数16 | .01975 | .08323 | .17109 | −0.65% | 57.21 | .0616 | .0395 | 48.3 | jerk/dct_high |
| J4 单样本（16 个平均） | .02833 | .10317 | .21170 | +22.94% | 55.71 | .1015 | .0441 | 39.4 | 多项 |

其余自然度（3 seed 均值，FD2 / J3 / J4）：滑行帧比 .180/.180/.187，步数 1.45/1.40/1.39（GT 2.29），f1–10 步态相位相关 .468/.459/.476，dct_mid .64/.61/.61，dct_high .011/.031/.030（护栏 .05 内），接触距离 RMSE 全动作 .186/.180/.180、推人 .208/.200/.199，骨长误差均为 0。完整表：`results/forecasting/ntu120_label/ntu2p_jitflow/subjval/rep10000_summary.{json,md}`。

## 归因

1. **扩散（JiT 流）训练在 10000 step 下是点预测收益的主要来源**：同结构去掉去噪损失的 J4N 在 5000 step 时 −2.50%，10000 step 降到 −0.71%（逐 seed 都明显回退）；带流损失的 J4 从 −3.35% 维持在 −3.22%。三个 seed 上 J4 − J4N 分别为 −3.8/−2.3/−1.5 个百分点。去噪任务对小数据（1709 条）上的长训练起正则作用；5000 step 时两者差异在 seed 噪声内，不能只看单 seed 筛选。
2. **新去噪器结构是另一半**：DCT-Mixer 主干同时承担去噪与回归时端点退化 4–5%（J1/J2），时空图 Transformer（人物 × 身体部位 × 5 帧时间块，adaLN-Zero、qk-norm、SwiGLU、瓶颈 patch 嵌入）不退化。
3. **接触图与跨人因果掩码没有可测增量**：J4 与 J3 的 3 seed 均值几乎相同（−3.22% vs −3.30%），J4 的交互指标反而更差（excess_antic .0374 vs .0309）。
4. **jerk 上升的来源**：J 端点的高频能量约为 FD2 的 3 倍（dct_high .030 vs .011），jerk 26 vs 20（GT 含拟合抖动为 307）。是更多中高频肢体运动，不是骨长或首帧问题；但按预登记属于未过护栏。

## 交互（"A 的手没碰到、B 就躲"）

- 机理（设计 2 节与 4.2 节）：DCT 截断不是原因（GT 投影到同一基后后退起点只偏 ≈0.01 s）；FD2 的问题主要是"A 的手始终没真正到达、B 照样按动作标签后退"，与确定性回归把手臂伸展与接触时刻取平均一致。
- 主指标 primary_excess_antic（全动作三阈值）：FD2 .0247，J3 .0309，J4 .0374，J4N .0311，样本读出更差（.044）。推人/击打小子集上（6/11 个新接触）J4 与 GRPO 后的端点有描述性改善（推人 0.25 m：FD2 .221 → J4 .177 → GRPO 后 .118），但同幅度改善也出现在 PG=0 的对照里，属继续训练效应，且样本太少。
- 全部 12 条推人曲线（`screen5000_tmix/analysis_stage1/all_pushing_contact_retreat.png`）与审查图（`screen5000_tmix/review_stage1/`，预先固定为全部推人窗 + seed0 随机 6 例）：所有确定性模型的 B 远离速度都从约 0.5 s 平滑上升，GT 在接触后才陡升；单样本反应陡峭但时刻随机，并伴随 A 冲前、脚滑。**用户关心的问题本轮没有解决**；生成式样本按本轮实现也没有给出更好的时序。

## JiT-RL（GRPO）

在 J4（5000 step EMA）上各 1500 次迭代，G=16、batch 8、10 步随机 flow-DDIM（ρ=0.7）、β=0.04、PPO-clip 0.2、端点配方损失权重 1、PG+KL ×50（梯度平衡修订）。第一版因 PG 梯度只有端点项的 1/47 而作废。

| 读出 | J4 | RL-A（只精度） | RL-F（完整奖励） | RL-S（PG=0 对照） |
|---|---:|---:|---:|---:|
| 端点 mpjpe | .16646 | .16640 | .16641 | .16638 |
| 端点漏接触 / excess_antic | 60 / .0363 | 53 / .0361 | 53 / .0359 | 53 / .0363 |
| 推人 0.25 m excess（端点） | .177 | .118 | .118 | .118 |
| 单样本 mpjpe | .2137 | .2183 | .2190 | .3558 |
| Energy Score | .208 | .212 | .213 | .346 |

结论：端点变化与 RL 无关（RL-S 相同）；RL 唯一的可测作用是让继续训练时采样器不崩（RL-S 的样本 mpjpe .356、ES .346），但没有把样本改善到初始模型之上；RL-F 的接触/接触前后退/物理奖励相对 RL-A 无增量。样本概率质量（ES .21）劣于上一轮 C4（.156）与 OOF bootstrap（.128）。

## 可复用产物与验证

- 模型：`model/forecasting_ntu2p_jitflow.py`（正交 ramp-DCT 状态编解码、JiT x 预测流、Mixer/STG 去噪器、自条件接触图、跨人因果掩码、条件化腿流、ODE/均值/几何中位数/medoid 读出）；`model/forecasting_ntu2p_v2.py` 增加 `ntu2p_jit_flow` 分派。
- 训练：`train/train_ntu2p_jitflow.py`（复用 v2 损失配方与管线；`--t_uniform_mix`）、`train/train_ntu2p_jitflow_grpo.py`（`--pg_scale`）。
- 评价：`eval/eval_ntu2p_v2.py` 新增 `--flow_readout/--flow_samples/--flow_nfe/--flow_solver/--flow_sample_seed/--interaction_metrics`（旧模型默认行为不变，只多输出字段）；`eval/eval_ntu2p_jitflow_samples.py`；交互时序指标 `utils/ntu2p_interaction_timing.py`。
- 驱动与分析：`scripts/run_ntu2p_jitflow.py`、`scripts/summarize_ntu2p_jitflow_seeds.py`、`scripts/analyze_ntu2p_jitflow_results.py`；诊断 `scripts/diag_ntu2p_dct_onset.py`、`scripts/diag_ntu2p_jitflow_t_coverage.py`、`scripts/diag_ntu2p_jitflow_grpo_grad_balance.py`。
- 结构检查 32 项通过：`scripts/check_ntu2p_jitflow.py` → `results/forecasting/ntu120_label/ntu2p_jitflow/structural_checks.json`。FD2 参照经新评价路径复现上一轮全部数字（mpjpe .172228、推人 RMSE .2027、jerk 20.40）。
- 权重：`save/forecasting/ntu120_label/ntu2p_jitflow_subjval_{J3,J4,J4N}-tmix_s{0,1,2}_10000/ema/model000010000.pt`；筛选与 GRPO 权重见第 1 阶段文档与 `..._grpo1500/`。

## 解释边界与后续

- DCT + 扩散、接触图、RL 微调扩散都有近邻工作，不声称首创；本轮可陈述的是"在本数据与协议下，时空图 Transformer + JiT 流训练使确定性端点 mpjpe 稳定降低约 3.2–3.3%，代价是 jerk +26%"。
- 若用户接受 jerk 取舍，下一步按协议在原划分 train 1758 × 3 seed 复训 J3（或 J4）并只评估一次 test；若不接受，可另立设计专门约束高频（如端点 jerk/高频 DCT 项）再筛，须重新预登记。
- 交互时序问题需要的不是更强的确定性回归：本轮证据显示换结构、加图、加 RL 奖励都不改变"接触前平滑后退"，需要把接触时刻作为显式离散事件来预测/采样，并解决样本质量（ES 劣于 bootstrap）后才可能有效。
