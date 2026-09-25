# NTU2P 快速 xyz 训练数据管线 + 坐标规范化 + 数据增广：结果

计划：`docs/ai/context/20260925-113313-ntu2p-xyz-seq-cache-canonical-augment-plan.md`。只新增文件，现有训练/评估入口行为不变。

## 新增文件

| 文件 | 作用 |
|---|---|
| `scripts/build_ntu2p_xyz_seq_cache.py` | 按 manifest 把 train/val/test 每条完整序列经训练同款 FK（h5 axis-angle → `raw_ntu_2p_to_rot6d` → `ntu_2p_rot6d_to_xyz`）存为 float32 扁平帧 `xyz [F,2,55,3]`，附 `offsets/lengths/actions/sample_ids/action_codes/config`（含 `manifest_hash`）。GPU 上约 2 分 44 秒。 |
| `data_loaders/forecasting/ntu2p_xyz_seq_cache.py` | `NTU2PXYZSeqCache`（加载时校验 manifest hash、sample_id 顺序、length、action）；`NTU2PXYZTrainSampler`（整份 train 帧常驻 GPU，独立 CPU `torch.Generator` 抽序列与起点，一次 gather，支持 `state_dict`，可挂增广）；`eval_windows` / `iter_eval_batches`（按 manifest 顺序取中心窗口 `(length-60)//2`）。 |
| `utils/ntu2p_canonical.py` | `estimate_up`、`canonical_frame` / `to_canonical` / `from_canonical` / `canonicalize`、`NTU2PSceneAugment`、`SMPLX_MIRROR_PERMUTATION`。 |
| `scripts/check_ntu2p_xyz_seq_pipeline.py` | 可复跑的自检：缓存与原数据集逐样本对比、采样器、规范化、增广。 |

正式缓存：`results/forecasting/ntu120_label/ntu2p_xyz_seq_cache/{train,val,test}_xyz_seq.pt`（train 1758 条 / 134,630 帧 / 170MB，val 198 / 15,013，test 1253 / 96,440）。

## 取舍

- **采样**：逐 epoch 随机排列、跨 epoch 首尾拼接。每条序列的出现频率与 `DataLoader(shuffle=True)` 相同，但每步都是满 batch；起点在 `[0, length-60]` 上均匀（含端点，同 `random.randint`）。
- **随机数用 CPU generator**：torch 1.7.1 在 CUDA generator 上调用 `randperm` 会段错误。采样器不消耗全局 RNG，所以数据采样与模型初始化、dropout 的随机流互不影响。增广另用一个 generator（seed + 7919），开关增广时抽到的序列与起点保持不变，便于配对比较。代价是随机流与旧 DataLoader 不同，新旧管线之间不能逐步配对。
- **3x3 变换用逐元素乘加实现**：torch 1.7 在 Ampere 上默认 `allow_tf32=True`，3x3 旋转用 matmul 往返的误差有 2.2e-3 m，逐元素实现为 7e-7。顺带发现：现有 SMPL-X FK 与模型训练本来就跑在 TF32 下，缓存与在线 FK 用的是同一路径，两者一致。

## 坐标系调查

train 1758 条 / test 1253 条，逐序列统计；详细数字见会话 scratchpad `r3_infra/coord_survey.json`。

1. **竖直轴**：xyz 在相机系下（y 向下、z 指向场景内）。重力"上"方向约为 −y，但带俯仰。train 全体 unit(neck−pelvis) 均值为 (−0.003, −0.977, −0.213)，逐序列相对 −y 的倾角中位数 12.1°（p95 28.8°），roll 约 0 ± 5°。俯仰随 setup 变化：各 setup 均值在 −0.1° 到 37.2° 之间，组间标准差 9.1°，组内中位标准差 3.8°。所有双人样本都来自 C001 相机。**不能把任何一个坐标轴直接当成竖直轴。**
2. **地面**：沿"上"方向取最低脚关节高度的序列中位数，结果为 −1.60 ± 0.26 m。各 setup 均值在 −2.04 到 −1.07 m 之间，组内标准差约 0.09 m。**地面高度跨样本不一致**，随相机高度变化。用脚点拟合的地面法向与身体"上"方向之间，夹角中位数 9–14°、p90 约 50°，说明单目深度噪声大，逐样本估计地面平面不可靠。
3. **朝向与位置**（观测末帧，中心窗口）：
   - A 相对"正对相机"的 |yaw| 中位数为 57°，p95 为 105°；B 的分布更散，中位数 78°，p95 154°。
   - A→B 连线的 yaw 在 ±90° 处呈双峰，即两人在画面中左右排开。A 在 B 左侧的比例是 50.1%，A/B 顺序不携带左右语义。
   - 两人相对朝向集中在 ±180°（面对面）附近。A 朝向 B（60° 以内）占 85%，B 朝向 A 占 63–69%。
   - 双人水平距离中位数 0.89 m，p1 0.25 m，最小 0.01 m。
   - 两人中点的水平位置：沿相机 x 为 0.14 ± 0.49 m，沿视线方向为 −0.65 ± 0.16 m。
4. **离群**：
   - 坐标绝对值超过 5 m 的序列：train 1 条、test 2 条，最严重的是 S013C001P028R001A010，出现 30 m 跳变。
   - 187 条 train 序列至少一人出现单帧 pelvis 跳变 > 0.3 m；只统计 A 为 80 条。其中真正的 A/B 身份互换很少（A 口径 train 1 条、test 5 条），其余是拟合毛刺。

## 规范化与增广的定义

**规范化** `canonical_frame(obs)`，只用观测，变换为 x_can = R(x − t)，可精确反变换：

- 原点 t：观测末帧双人 pelvis 中点。默认连竖直分量一起减去，因为地面和相机高度跨样本不一致；`vertical_origin="keep"` 可保留竖直分量。
- +Y："上"方向。默认 `up_mode="obs"`：观测 10 帧、双人的 unit(neck−pelvis) 均值；与数据集常量夹角超过 30° 时回退常量。回退样本数：train 4 条、val 2 条、test 11 条。观测估计相对所在 setup 均值的偏差中位数 3.2°、p99 16°；直接用常量时偏差中位数 4.1°、p99 24.8°。
- +X：A→B pelvis 连线的水平分量。水平距离小于 0.1 m 时回退为 A 的朝向（train 3 条、val 1 条、test 2 条）。也可选 `yaw_ref="a_facing"`。
- +Z = X × Y。
- 选 A→B 连线的理由：pelvis 位置最可靠；对齐后双人布局只剩距离一个自由度；A 的朝向依赖髋和肩，扭身时有歧义。

**增广** `NTU2PSceneAugment`，仅用于训练，必须在相机系下、规范化之前做：

- A/B 交换。
- 镜像：绕过末帧中点、法向为水平化相机 x 轴的竖直面反射，再用 SMPL-X `left_*`/`right_*` 名称置换关节。置换表：`0,2,1,3,5,4,6,8,7,9,11,10,12,14,13,15,17,16,19,18,21,20,22,24,23,40..54,25..39`。
- 绕"上"轴随机 yaw：默认 ±180°，转轴过末帧中点。
- 可选随机水平平移：`translate_std`，默认 0。
- yaw 与平移会被规范化精确消去，只在不做规范化的训练中有用。交换与镜像在两种训练里都有效：规范化后分别等价于绕 Y 转 180° 加人物交换、z 轴反射加关节置换。

## 验证

- **缓存与原数据集对比**：原路径为 `NTU2PDiffusionForecastDataset` + 在线 FK（GPU，batch 16）。样本顺序、起点、action 全部一致。
  - val 198 条：obs 最大绝对误差 0，target 4.8e-7；
  - test 1253 条：obs 0，target 1.9e-6；
  - train 按采样器给出的 64 个 (sample_id, start) 抽检：0。
- **采样器**：同 seed 前 20 步逐位相同；`state_dict` 恢复后续步相同；一个 epoch 内每条序列至多出现一次；起点全部落在合法范围内。
- **规范化**：往返误差 ≤ 9.5e-7，旋转矩阵正交误差 ≤ 3e-7，行列式 ≥ 0.9999997；规范系下末帧中点 ≤ 7e-8，非回退样本的 A→B z 分量 ≤ 6e-8。
- **指标不变性**：在 val 198 条 + train 256 条上，分别用 copy-last 与当前主线 const-EMA s0 模型的预测，比较规范系与原系算出的指标。
  - `xyz_mse/mpjpe/root_mse/local_mse` 的相对差 ≤ 2.1e-7，属于数值误差；
  - 其余旋转不变指标的最大相对差为 1.3e-5，来自 `frozen_ratio` 阈值附近的少数翻转，或 copy-last 中本应为 0 的约 1e-13 量级 DCT 能量；
  - **`xyz_mae` 与 `local_pose_temporal_std` 按坐标分量计算，不具旋转不变性**：`ab_line` 下相对差 0.04–0.4%，`a_facing` 下 3–7%。因此指标必须在 `from_canonical` 回到原系后计算。
  - 同理，loss 中按分量计算的项（mae、`dct_*_amplitude`、`temporal_std`）在规范系下数值会变。
- **增广**：
  - 交换两次误差 2.4e-7，镜像两次 1.7e-6；
  - yaw 与平移后骨长误差 7e-7，竖直分量误差 7e-7，末帧两两距离误差 9.5e-7，up 估计不变；
  - 镜像后骨长与对侧原骨长一致（7e-7）；与同名原骨长最多差 1.75 cm，这是 SMPL-X 中性模板本身的左右不对称（左右 hip→knee 骨长差 1.7 cm，模板关节位置的左右差最大 3.4 cm），镜像人体因此不是严格的零 beta SMPL-X；
  - 规范化消去 yaw 与平移的误差 1.1e-6；镜像等价于 z 反射（1.4e-6）；交换等价于转 180°（7e-7，距离 ≥ 0.1 m 的 452/454 个样本）。

规范化与增广部分在最后一次代码整理后又用 CPU 复跑过一遍，结论相同。

## 测速

当前最佳 residual refiner，配置同 s2_5 const-EMA：前向、归一化 loss、反向、clip、AdamW、EMA，每步 `.item()`。warmup 20 步后计时 300 步。GPU 同时被另一会话的训练（约 4.6GB）占用，期间还短暂跑过第三个评估进程，因此绝对值偏低，比值更可信。

| 配置 | step/s | samples/s | 仅数据 batch/s | 峰值显存 MB |
|---|---|---|---|---|
| 旧 b8 | 2.73 | 21.8 | 4.75 | 1769 |
| 新 b8 | 9.85 | 78.8 | 4701 | 500 |
| 旧 b32（FK 按 8 窗分块，100 步） | 1.54 | 49.2 | 1.90 | 1777 |
| 新 b32 | 10.64 | 340.5 | 4534 | 761 |
| 新 b64 | 10.56 | 676.1 | 4628 | 1104 |
| 新 b32 + 增广 + 规范化 | 10.45 | 334.3 | 253 | 764 |
| 新 b64 + 增广 + 规范化 | 8.21 | 525.4 | 182 | 1109 |

- 同 batch 8 时新管线快 3.6 倍；batch 32 时快 6.9 倍（samples/s）。新 b64 的样本吞吐是旧 b8 的 31 倍，仅看数据供给约快 1000 倍。
- 旧管线的 FK 很占显存：batch 8 时 FK 约 1.3GB，不分块跑 b32/b64 会超过 3GB 显存预算。
- 新管线在 b8–b64 下 step/s 基本不变（约 10.5），瓶颈是每步的 Python/kernel 启动与同步开销，不是算力，所以加大 batch 几乎不增加耗时。
- 增广与规范化在 b64 上每步约 5.5 ms，使 step/s 降低 22%。如需优化，可改用分量显式公式，或在规范化训练中跳过 yaw/平移。

## 待 GPU 空闲后执行

- `PYTHONPATH=. python scripts/check_ntu2p_xyz_seq_pipeline.py --device cuda:0`：在最终代码上完整复跑，含缓存对比。上文的缓存对比数字来自代码整理前的同一逻辑。
- 可选：`PYTHONPATH=. python scripts/build_ntu2p_xyz_seq_cache.py --device cuda:0` 重建缓存。现有缓存与最终代码的差别只在 config 里的两条描述性字符串（`xyz_layout`、`fk_path`），数值无差别，不重建也能正常使用。
- 可选：在独占 GPU 时重跑测速，得到干净的绝对值。测速脚本在会话 scratchpad `r3_infra/bench.py`。
