# NTU2P residual refiner 视频"平移无摆动"问题诊断与定量验证计划

## 背景

用户反馈：`ntu2p_residual_refiner_xyz_inter001_s0_5000`（当前 `xyz_mse/xyz_mae/mpjpe` 全面超过 inherited baseline 与 `copy-last` 的最佳 checkpoint，见
`docs/ai/context/20260823-084520-ntu2p-residual-refiner-architecture-visualization-result.md`）导出的视频中，人物骨架看起来是整体刚性平移到目标位置，左右手/脚没有随动作摆动。指标更好但视觉质量差，需要定位原因。

## 代码走查结论（定性诊断）

1. **架构是零初始化的确定性回归，不是扩散/生成式采样**：
   - Baseline `NTULabelXYZTransformer.forward`：`pred = obs_xyz[:, -1:] + delta`，`output_proj` 权重与 bias 显式零初始化（`model/forecasting_ntu_xyz.py:143-144`），初始输出严格等于 copy-last。
   - Residual refiner `NTU2PResidualRefinerXYZ.forward`：`pred_xyz = base_xyz + alpha * ramp(delta)`，`delta_proj` 同样零初始化（`model/forecasting_ntu2p_residual_xyz.py:114-116`）。
   - 训练时对 `delta` 额外加 L2 正则：`delta_reg_weight=0.01 * (delta**2).mean()`（`train/train_ntu2p_residual_refiner_xyz.py:123-124`）。
   - 三者叠加：模型从"等价 copy-last"起步，且被正则持续拉回小幅修正，天然偏保守。

2. **损失函数是纯 L2/L1 加权和，没有对抗/感知损失，也没有多模态采样**（`_loss_terms`, `train/train_ntu2p_residual_refiner_xyz.py:99-125`；baseline 的 `training_loss`, `model/forecasting_ntu_xyz.py:217-286`）。对于步态相位、摆臂幅度这类未来多模态问题，逐点 MSE 的最优解是对所有合理未来取平均：
   - `root_positions`（`utils/ntu_smplx_2p_xyz.py:120-121`）在几十帧内相对单峰，容易学、也最"划算"（`root_loss_weight=1.0`）。
   - `local_pose = value - root`（`utils/ntu_smplx_2p_xyz.py:124-125`）因相位不确定被平均后趋向静止姿态附近（`local_pose_loss_weight=1.0`，与 root 同权重但未做量级归一化，root 位移幅度通常远大于关节相对偏移幅度，梯度被 root 主导）。
   - `velocity_loss_weight=0.2` / `acceleration_loss_weight=0.1` 鼓励延续观测速度做平滑外推，进一步强化"匀速滑步"解。

3. **训练预算有限**：5000 step、batch_size=8，从零初始化的"copy-last 等价解"起步，难以在有限步数内学出大幅、有相位细节的摆动模式。

4. `inter_loss_weight=0.01`（当前最佳 checkpoint 使用）只约束双人关节间距离，不约束单人自身肢体摆动幅度，不能缓解此问题。

上述诊断指向经典的**确定性回归对多模态运动的"回归到均值"（regression-to-the-mean）现象**：`xyz_mse/xyz_mae/mpjpe` 等 L2 距离类指标系统性奖励"贴近均值、保守"的解，因此指标提升不代表视觉上有真实的肢体摆动。

## 定量验证计划

在 val split（198 条样本，与既有评估口径一致）上，对比 `model(residual refiner)`、`base(inherited single-person)`、`copy_last`、`target(GT)` 四者的：

1. **articulation energy**（关节相对 root 的帧间摆动能量）：
   `local_pose = value - root_positions(value)`；
   `local_pose_velocity = local_pose[:, 1:] - local_pose[:, :-1]`；
   `articulation_energy = mean(local_pose_velocity ** 2)`（对 batch/frame/joint/coord 取均值）。
   - `copy_last` 按定义恒为 0（每帧都是同一帧的复制），作为"零摆动"下界参照。
   - 预期：若诊断成立，`model`/`base` 的 articulation energy 远小于 `target`，且比值（model/target）显著低于对应的整体位移比值。

2. **root energy**（root 关节帧间位移能量）：
   `root_velocity = root_positions[:, 1:] - root_positions[:, :-1]`；
   `root_energy = mean(root_velocity ** 2)`。
   - 预期：`model` 的 root energy 相对 `target` 的比值明显高于 articulation energy 的比值，证明模型把优化预算主要花在了 root 轨迹上。

3. **frozen-frame 比例**：以 `target` 的 per-joint per-frame local_pose 帧间位移中位数作为尺度参考阈值 `epsilon`（GT 位移中位数的 10%），统计 `model` 预测中 local_pose 帧间位移小于 `epsilon` 的关节-帧比例，直观量化"基本不摆动"的时间占比。

复用现有推理与数据加载路径（`load_ntu2p_residual_refiner_checkpoint`、`NTU2PDiffusionForecastDataset`、`ntu_2p_rot6d_to_xyz`），新增只读分析脚本 `eval/analyze_ntu2p_residual_refiner_articulation.py`，不改动任何已有训练/评估代码，只做前向推理与统计聚合。

- 使用 checkpoint：`save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_inter001_s0_5000/model000005000.pt`
- manifest：`results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json`
- split：`val`，`--device cuda:0`

结果将写入新的时间戳文档，不覆写本文件。
