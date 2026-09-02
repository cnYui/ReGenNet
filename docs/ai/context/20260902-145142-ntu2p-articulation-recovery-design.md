# NTU2P residual refiner 关节摆动恢复：调整设计

## 目标

在保持"paired `xyz_mse/xyz_mae/mpjpe` 必须超过 `copy-last`"这一硬约束不变的前提下，让 residual refiner 的输出出现真实的肢体摆动：
- 局部姿态帧间摆动能量比值（`articulation_energy_ratio_to_target`）从当前 **0.55%** 提升至少一个数量级（Stage 1 目标 ≥ 10%，Stage 2 目标 ≥ 30%）；
- frozen 关节-帧占比从 **37.6%** 降到 ≤ 20%（GT 为 9.8%）；
- 视频中人物不再是整体刚性平移。

前置诊断与定量证据：
`docs/ai/context/20260902-143842-ntu2p-residual-refiner-regression-to-mean-diagnosis.md`、
`docs/ai/context/20260902-144108-ntu2p-residual-refiner-regression-to-mean-quantitative-result.md`。

## 补充测量（本次新增，val 198 条）

脚本 `eval/analyze_ntu2p_residual_refiner_articulation.py` 新增了三组分解量，原始结果在
`results/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_inter001_s0_5000_articulation_analysis.json`：

| 量 | copy_last | base | model | 解读 |
|---|---:|---:|---:|---|
| root 位置 MSE | 0.0374 | 0.0229（−39%） | 0.0210（−44%） | 学习型模型的收益主要来自 root |
| local_pose 位置 MSE | 0.0224 | 0.0174（−22%） | 0.0169（−24%） | local 也降了，但降幅约为 root 的一半 |
| model 相对 base 的改进 | — | — | root −8.3% / local −2.7% | residual 分支 3 倍地把收益投在 root 上 |
| 全局速度 MSE（与训练 velocity loss 同口径） | 0.001242 | 0.001222 | 0.001220 | 模型几乎没有改善速度；`0.2 × 0.00122 ≈ 2.4e-4`，仅为位置 MSE（0.0387）的 **0.6%**，velocity/acceleration 项在当前权重下实际上不起作用 |
| local_pose 时间 std 比值 | 0 | 0.328 | 0.338 | 模型输出的是幅度约 1/3 的**缓慢单调形变**，而非振荡 |
| local_pose 帧差能量比值 | 0 | 0.0050 | 0.0055 | 高频/振荡成分几乎为零 |

两个关键推断：

1. **local MSE 下降 24% 但摆动能量只有 GT 的 0.55%**，说明模型学到的是"把静止姿态从最后一帧观测位置慢慢挪到未来的条件均值姿态"——预测均值姿态在 L2 上一定优于任何一个固定姿态，但不需要任何摆动。这就是回归到均值的精确机制，也是"指标好、视频差"的直接原因。
2. **时间 std 比值（0.34）远高于帧差能量比值（0.0055）**，说明现有模型只有低频漂移、没有振荡；因此评估时**必须同时报告帧差能量口径**，单看 std 会严重低估问题。

## 竞争性假设与判别实验

设计不预设唯一原因，Stage 1 用 one-factor 实验分别判别：

| 假设 | 机制 | 判别实验 |
|---|---|---|
| H1 训练不足 | 5000 step × batch 8 ≈ 40k 样本 ≈ 23 个 epoch，零初始化起步，可能尚未走出"线性漂移"解 | 同配置训到 20000 step，每 1000 step 评估 |
| H2 残差被正则压小 | `delta_reg_weight=0.01` 直接惩罚 `delta` 幅度；`alpha` 学到 0.81 也在收缩 | `delta_reg_weight=0` |
| H3 结构性时间抑制 | (a) `ramp = linspace(0,1,50)` 令前 10 帧只允许 ≤20% 的残差，平均只放行 50%；(b) `future_pos` 为零初始化可学习参数，初始时 50 个未来 query 完全相同 → 解码器初始只能输出"随时间线性"的位移，振荡必须靠位置编码先学出来 | (a) 快速饱和 ramp `min(t/5, 1)`；(b) 固定多频正弦位置编码 |
| H4 loss 尺度失衡 | 位置 MSE 被 root 主导；速度/加速度项数值上只占 0.6%，等价于没有动力学约束 | 各项按 GT 尺度归一化后加权；新增局部姿态速度损失 |
| H5 L2 点估计对多模态未来必然回归到均值 | 步态相位/摆臂幅度不可预测时，逐点 L2 最优解就是均值姿态，与架构无关 | 加入相位无关的摆动能量匹配损失；根本方案是生成式（residual diffusion） |

H1–H4 的修复是"让确定性模型至少能表达并被鼓励表达摆动"；H5 决定了确定性路线的上限。预期 Stage 1 后 H3/H4 至少有一项显著有效，H5 决定是否需要 Track B。

## 调整方案（按实现成本排序）

### A. 评估与 gate（必做，无训练成本）

1. 将 `articulation_energy_ratio_to_target`、`local_pose_temporal_std_ratio_to_target`、`frozen_joint_frame_ratio_below_epsilon`、root/local 位置 MSE 分解并入 `eval/eval_ntu2p_residual_refiner_xyz.py` 的标准输出，每个 checkpoint 都报告。
2. 新 gate：
   - 硬约束：三项 L2 指标必须低于 `copy-last`（不变）；
   - 新增硬约束：`articulation_energy_ratio ≥ 0.10`（Stage 1）/ `≥ 0.30`（Stage 2），`frozen_ratio ≤ 0.20`；
   - 在通过 gate 的 checkpoint 中再按 L2 选优。
3. 可视化选例改为"L2 最佳 8 例 + 随机 8 例"。此前 8 例按 `copy_last_xyz_mse − model_xyz_mse` 降序选取，会系统性偏向 root 大位移的样本，放大"平移"观感、掩盖真实分布。
4. 待确认的取舍（需用户决定）：相对 inherited base 允许多大 L2 回退。建议：**允许 mpjpe 相对 base 回退 ≤ 5%，但必须仍显著低于 copy-last**，换取摆动能量比值 ≥ 10 倍的提升。

### B. 损失函数（主攻 H4、H5）

保持现有各项与默认权重不变以便对照，新增以下可选项（默认关闭，旧配置可精确复现）：

1. **按 GT 尺度归一化**（`--loss_scale_normalize`）：每一项除以其在训练集上的 GT 均方常量（一次性统计写入 args.json），使每项在 O(1) 量级，权重才有可解释性。速度/加速度项从 0.6% 恢复到与位置项同量级。
2. **局部姿态速度损失** `L_lv`（`--local_velocity_loss_weight`）：
   `mse(Δ_t local_pose(pred), Δ_t local_pose(target))`，含最后一帧观测到第 1 帧预测的差分。与 root 解耦，直接监督"关节相对 root 怎么动"。
3. **相位无关摆动能量匹配损失** `L_energy`（`--articulation_energy_loss_weight`）：
   对每个 (person, joint) 计算 `E = mean_t ||Δ_t local_pose||²`，取 `mse(E_pred, E_target)`（对 batch 平均）。这是非逐点的矩匹配：即使相位预测错，只要摆动幅度对就不受罚，是对抗均值坍缩的直接手段。风险是模型用高频抖动凑能量；用（归一化后的）acceleration loss 抑制，并在 Stage 2 视需要改为 DCT 分频带幅度匹配（`torch_dct` 已安装）。
4. `delta_reg_weight → 0`（H2）。
5. 可选：action-feature 感知损失。代码路径已存在（`action_feature_loss_weight` / `eval/action_xyz_classifier.py`），但可用的 xyz 分类器权重已于 20260822 清理且为 p40 协议，需先按 o10/p50 重训分类器。放在 Stage 2 之后再评估是否值得。

### C. 模型结构（主攻 H3；用户要求一并给出可尝试项）

按"改动小、可归因"优先：

1. **快速饱和 ramp**（`--ramp_mode saturate --ramp_saturate_frames 5`）：`ramp_t = min(t/5, 1)`，首帧仍为 0 保证连续性，第 5 帧后残差全额放行。平均放行率 0.5 → ≈0.95。
2. **固定多频正弦未来位置编码**（`--future_pos_mode sinusoidal`）：替换零初始化的可学习 `future_pos`，让解码器从第 0 步就能表达周期性输出；可与可学习偏置相加。
3. **root / local 双头输出**：`delta_proj` 拆为 root 位移头（3 维）与局部姿态头（54×3 维），分别接受归一化损失。让两个分量的容量与梯度显式分离，配合 B.1 使权衡可控。中等改动。
4. **局部姿态 DCT 域输出**（SomoFormer 风格）：局部姿态残差在 DCT 低频系数（如前 16 个）上预测再逆变换，天然平滑，且 `L_energy` 可直接在系数幅度上做。中等改动，项目已有 SomoFormer/DCT 经验。
5. **解码器容量**：2 层/256 维偏小，但基于以上分析容量不是首要瓶颈；仅在 C.1–C.3 生效后，作为 Stage 2 的一个附加因子试 4 层。
6. **后期低学习率解冻 base**：会破坏"冻结 base + 可归因 residual"的实验设计，只在 Stage 3 作为可选项，且需单独对照。

### D. 训练预算（用户提问：step 是否可提升）

可以，且成本很低：当前 5000 step ≈ 14 分钟（RTX 3080），20000 step ≈ 56 分钟。但训练集仅 1758 条序列、类别极不平衡（最少类 1–9 条），延长训练需配合每 1000 step 的 val 评估以监控过拟合；`weight_decay=1e-4`、`dropout=0.1` 保持。单独延长 step 只能判别 H1，不预期单独解决问题。

### E. Track B：residual diffusion（根本解决 H5）

AGENTS.md 已将"迁移到 residual diffusion"列为下一阶段。本设计只固定其**评估协议**，实现另开 design：
- L2 三指标用 **K 个样本的均值轨迹**（mean-of-K，K=10）计算，与确定性模型的条件均值口径可比；
- 摆动指标用**单样本**计算并报告 K 个样本的分布；
- 同时报告 best-of-K，作为多模态上界参考。
此前 ntu2p diffusion "未超过 copy-last"是用单样本算 L2 得出的，对生成式模型天然不利；该结论需在新协议下重新解读，但不改写历史记录。

## 不做的事

- 不引入 GAN/判别器：项目无相关基础设施，训练不稳定，且 `L_energy` 已能提供相位无关的"反静止"信号。
- 不改变 o10/p50 协议、manifest、val split、seed 0、base 冻结、`inter_loss_weight=0.01`（与当前最佳 checkpoint 一致，保证可比；AGENTS.md 的"默认 0"留待本轮之后再统一）。
- 不修改已有 checkpoint、结果文件与历史文档。

## 与既有结论的关系

- 不推翻 `20260823-084520` 中"residual refiner 在三项 L2 指标上超过 base 与 copy-last"的结论；本设计是补充"这三项指标对摆动不敏感"并给出修复路径。
- 若 Stage 2 后确定性路线的摆动能量比值仍停在 10–30%，则说明 H5 成立，Track B 成为主线。

## 详细训练计划

见 `docs/ai/context/20260902-145142-ntu2p-articulation-recovery-training-plan.md`。
