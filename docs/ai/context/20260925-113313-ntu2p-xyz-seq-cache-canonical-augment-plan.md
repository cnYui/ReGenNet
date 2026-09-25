# NTU2P 快速 xyz 训练数据管线 + 坐标规范化 + 数据增广：设计与计划

目的：为后续多个新架构实验提供可复用的训练数据基础设施。只新增文件，不改变任何现有训练/评估入口的行为。

## 背景

- 现有 residual refiner 训练每步在 GPU 上把 rot6d 窗口经 SMPL-X FK（`Rotation2xyz_x`，含 10475 顶点的 LBS）转成 xyz，`num_workers=0`、batch 8，约 5.7 step/s，GPU 利用率约 35%，推测数据/FK 是瓶颈。
- 已有 `results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_xyz_cache/` 只存固定窗口，不能做 train 随机起点采样。
- xyz 是原始世界系，未做空间规范化，也没有增广。

## 设计

1. `scripts/build_ntu2p_xyz_seq_cache.py`：manifest 中 train/val/test 每条**完整序列**，按与训练完全相同的路径（h5 raw axis-angle → `raw_ntu_2p_to_rot6d` → `ntu_2p_rot6d_to_xyz`）逐帧 FK，存 float32 扁平帧张量 `xyz [总帧数,2,55,3]` + `offsets/lengths/actions/sample_ids` + `manifest_hash` + 生成配置。FK 按帧独立（`num_person>1` 分支的平移不减首帧），所以整序列 FK 后切片与窗口 FK 等价，差异只来自 GPU 批量矩阵乘的浮点顺序。
2. `data_loaders/forecasting/ntu2p_xyz_seq_cache.py`：
   - `NTU2PXYZSeqCache`：加载缓存并校验 manifest hash；
   - `NTU2PXYZTrainSampler`：整份 train 帧放 GPU（约 178MB），独立 `torch.Generator` 驱动；序列按"逐 epoch 随机排列、跨 epoch 拼接"取（每条序列出现频率与 DataLoader shuffle 相同，但始终满 batch），起点在 `[0, length-60]` 均匀；一次 gather 出 `obs_xyz/target_xyz/action`；
   - val/test：按 manifest 顺序、`start = (length-60)//2` 中心窗口，返回全量张量与逐 batch 迭代器（附 meta）。
3. `utils/ntu2p_canonical.py`：
   - 坐标系调查函数（竖直轴、地面高度、yaw 与位置分布），结论写结果文档；
   - 可逆场景规范化：以观测末帧为参考，平移双人 root 水平中点到原点、绕竖直轴旋转使参考方向对齐固定轴，返回 `(R, t)` 供反变换；
   - 训练增广：整场景随机 yaw、A/B 交换、左右镜像（水平轴反射 + SMPL-X 左右关节置换）、可选随机水平平移；均为纯函数，随机数由调用方 generator 提供。
4. 验证：val/test 窗口与原数据集 + on-the-fly FK 逐样本对比；train 抽检同一 `(sample_id, start)`；刚体变换下 L2 指标不变；增广自检（镜像两次恒等、yaw 保骨长与竖直分量）。
5. 测速：当前最佳 residual refiner（s2_5 配置）前向+反向，旧/新管线各约 300 step，batch 8/32/64。GPU 与另一训练共享，报告时注明。

## 约束

- 显存 ≤ 3GB；不写入 Stage A 的 root/inter 目录；临时产物放会话 scratchpad，正式缓存放 `results/forecasting/ntu120_label/ntu2p_xyz_seq_cache/`（gitignored）。
- 采样器用独立 generator，不消耗全局 RNG，使模型初始化与 dropout 的随机流与数据采样解耦。
