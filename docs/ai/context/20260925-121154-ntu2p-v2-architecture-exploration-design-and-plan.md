# NTU2P 第二代架构探索（v2）：设计与计划

日期：2026-09-25。用户要求：用子 agent 深度调研并大胆设计新模型架构，实跑测试比较优劣，适当复用 ReGenNet 原架构的优点，目标是提高分数；同时必须亲自看渲染结果，防止出现"分数高但动作不自然、两条腿不前后迈步而是整体漂移"的情况。

对照：当前主线 const-EMA（`ntu2p_residual_refiner_xyz_artic_ema_s2_5_dct1_root01_s{0,1,2}_10000/ema/model000010000.pt`），val mpjpe 0.1751 ± 0.0014，test 0.1882 ± 0.0005。另一会话正在跑 root DCT 头与 inter 权重的 Stage A（`docs/ai/context/20260925-110531-ntu2p-root-dct-head-inter-weight-plan.md`），本计划不依赖其结果。

## 1. 前期调研（三个子 agent，只用 train/val，未看 test）

原始报告在会话 scratchpad：`r1_diagnosis/`（自然度诊断）、`r2_design/design.md`（文献与候选设计，含 a1–a9 分析脚本）、`r3_infra/`（数据管线）。基础设施结果另见 `docs/ai/context/20260925-120854-ntu2p-xyz-seq-cache-canonical-augment-result.md`。

### 1.1 自然度诊断：当前最佳模型确实在"滑步"

新工具：`utils/ntu2p_naturalness.py`（脚滑、骨长、步态交替、分部位误差）、`eval/analyze_ntu2p_naturalness.py`（checkpoint 或 pred 数组两种输入，按 26 类分解）、`sample/render_ntu2p_review_sheet.py`（静态审查图：GT/各方法 × 6 个时间点、俯视踝关节足迹、踝高度与 root 速度曲线；可选 mp4）。

val 上 const-EMA s0（步行 = GT 未来 root 水平净位移 ≥ 0.5 m 的人，共 98 人次）：

| 指标 | 模型 | base | copy-last | GT |
|---|---:|---:|---:|---:|
| GT 站定帧上的预测脚速（步行，m/s） | 0.36 | 0.27 | 0.00 | 0.10 |
| 滑行帧比（步行） | 0.23 | 0.63 | – | 0.03 |
| 与 GT 迈步相位的相关 | 0.10 | 0.05 | – | 1 |
| 腿摆幅 / GT | 52% | 8% | – | 100% |
| 身体骨长相对误差（均值 / 最坏） | 5.2% / 51% | 3.9% | 0 | 0 |

- 看图结论（亲自核对了 `walk_096`、`rand_122` 等审查图）：步行时腿基本冻结在分腿姿态、整体向前拖，地面足迹是连续拖痕；只有小幅、相位与 GT 无关的剪刀摆动。非步行动作姿态合理，但出拳、拍背等手臂动作幅度偏小，小碎步被慢速滑脚替代。
- 推断（未验证）：DCT 幅度损失只约束能量不约束相位，腿"有摆动但相位随机"，与 root 前进不耦合，于是表现为滑步。

### 1.2 误差结构与可用杠杆（R2，val）

- **mpjpe 由手臂末端主导**：30 个手指关节占误差总和 66%，但手指相对手腕的误差只有 0.05（copy-last 0.058），手指误差主要是手腕位置误差（0.187）。把手指固定为观测末帧只损失 2.7 mm，手可以当刚体。
- **坐标规范化是最强的归纳偏置**：岭回归在相机系 0.2280，按双人连线做 yaw 规范化后 0.1963（−14%），已优于冻结 base（0.2265）。同动作 kNN（K=40）相机系 0.1938，规范系 + 镜像库 0.1810。
- **检索与学习模型互补**：0.5·模型 + 0.5·kNN 为 0.1667（−4.8%）；只混合 root 为 0.1698（−3.1%），dct_mid 不变。
- **骨架合法化不损失 L2**：把输出 IK 投影到固定骨架，mpjpe 0.17515 → 0.17516。
- **拟合状态**：refiner train 0.133 / val 0.175，过拟合；base 在 train 上只有 0.215，欠拟合。
- **人物顺序是施动 / 受动**：A 的动作幅度与误差都大于 B；ReGenNet（CVPR 2024，Table 8）也显示显式施动-受动顺序优于随机顺序。因此**不做 A/B 交换增广**。镜像会破坏惯用手（岭回归 +1.5%），只能带"是否镜像"的指示 embedding 使用。

### 1.3 数据管线（R3）

完整序列 xyz 缓存 + GPU 常驻随机截窗采样器 + 可逆场景规范化 + 增广。缓存与原数据集 + 在线 FK 的逐样本误差：val 4.8e-7、test 1.9e-6。batch 8 下 2.73 → 9.85 step/s。xyz 处在相机系（y 向下，俯仰 0–37° 随 setup 变化，地面高度 −1.60 ± 0.26 m），不能把任何坐标轴当竖直轴。

### 1.4 一个会影响模型选择的数据事实

**val 的 42 个受试者（sample_id 的 P 字段）全部出现在 train 中；test 的受试者与 train 完全不重叠。** 同一受试者会以 R001/R002 重复表演同一动作，检索很可能直接命中"同一人演同一动作"的窗口，val 因此会系统性高估检索类方法。R2 的 kNN 数字没有做受试者排除。

规定：检索库只由 train 构建；训练和 val 查询时**排除同受试者**（训练时同时排除同序列），模拟 test 的"受试者不重叠"条件；test 查询不需要排除（天然不重叠）。报告 val 时同时给出排除与不排除两种口径的 kNN-only 数字，作为这项偏差的量化。

## 2. ReGenNet 原架构中复用的优点

| ReGenNet 设计 | 在 v2 中的形式 |
|---|---|
| SMPL-X 运动学表示，输出永远是合法人体 | 可微骨架投影层（固定 SMPL-X 骨长 + 刚体手），不作为扩散目标（已失败） |
| 施动-受动顺序（NTU120-AS 标注） | 人物 role embedding，不做 A/B 交换增广 |
| 显式交互几何（相对平移、相对朝向、FK 相对位置） | 双人共享的场景规范系（原点 = 双人中点，+X = A→B 连线），相对几何直接成为坐标 |
| Transformer 解码 + 跨人 attention | 保留现有 refiner 的人物内 self-attn、双向 cross-attn 与 memory 解码 |
| 直接预测 x0 并在 x0 上加几何损失 | 保留确定性 xyz 输出与全部几何损失 |

## 3. 候选架构

全部新开关默认关闭；在新数据管线下，关闭全部开关时应等价于旧 refiner（数值上等价，随机流不同）。模型在规范系中工作时，loss 仍在规范系计算（L2 类损失对刚体变换不变），但 **`xyz_mae` 不具旋转不变性，评估一律回到原相机系用现有指标函数计算**。

### A0 ctrl：旧 refiner，新管线

与主线同配置（s2_5 损失、EMA 0.999、batch 8），只换成缓存管线。作为所有候选的同管线、同 seed 配对对照。

### A1 Canon：规范系 refiner（稳健）

- 冻结 base 仍在相机系运行（它就是在相机系训练的），得到 base_xyz；obs 与 base_xyz 一起变换到场景规范系，作为 refiner 的 token 输入；delta 在规范系输出，再旋回相机系。
- obs token 增加位移通道：`Linear(165→d)` 作用于 `x_t − x_last`，与位置投影相加。
- 人物 role embedding：`nn.Embedding(2, d)`，加到 A/B 的 obs 与 future token 上。
- 假设：绝对相机坐标迫使模型用数据去学平移与 yaw 不变性；规范化直接去掉这部分负担，并能减少原地动作上的虚假 root 漂移。

### A2 Canon + 解冻 base（检验"冻结 base 是否瓶颈"）

A1 基础上解冻 base，base 学习率为 refiner 的 0.1 倍，loss 不变。

### A3 RAR：检索锚点 refiner（大胆）

- 检索库：train 全部 L≥60 序列按步长 1 切窗（约 3.1 万个），在各自的场景规范系中存储。键为观测 10 帧 × 22 个身体关节；值为未来相对观测末帧的位移（运行时从 GPU 常驻缓存 gather，不单独存大数组）。
- 查询：同动作、排除同受试者（训练时再排除同序列），取 top-K（K=16）。
- 锚点融合：`anchor = base + ramp · β ⊙ (Δ_knn − Δ_base)`。β 分 root / 局部两个通道、逐帧，root 初始化 0.5、局部初始化 0（局部完全取 base，保护摆动），邻居权重由距离 softmax 加零初始化打分 MLP 得到。
- exemplar 记忆：每个邻居每人一个 token（`Linear` 编码其 DCT 压缩的未来位移 + 距离 embedding），拼进 refiner decoder 的 memory。
- 输出：`pred = anchor + ramp · delta`，delta 零初始化。
- 风险：检索的训练 / 测试分布不一致；局部通道被平均后摆动下降；val 的受试者重叠（已用排除规则处理）。

### A4 KinProj：可微骨架投影 + 刚体手（自然度）

- 固定 SMPL-X 静息骨架（betas=0，数据骨长标准差 5e-8 m）。自上而下投影：单子节点关节沿预测方向取固定骨长；多子节点关节（pelvis、spine3、head）用加权 Procrustes 求旋转；手作为刚体，以观测末帧的手部构型为模板，用 Procrustes 对齐预测的手指点云。
- 首帧：投影前首帧等于观测末帧（本身合法），投影为恒等，首帧误差仍为 0。
- 训练时对投影后的输出算全部损失，另对投影前的自由输出加 0.1 × mse 的辅助项，防止 SVD 退化处梯度不稳。
- 预期：骨长误差 5.2% → 0，L2 在 ±0.5% 内。

### A5 FootLock：脚接触一致性损失（自然度）

- GT 接触掩码：规范系中脚（踝、脚掌）高度低于该人观测期最低脚高 + 阈值，且 GT 水平速度低于阈值（阈值沿用 R1 用 GT 校准的值）。
- `L_foot = mean(m_gt · ‖v_h,pred‖²)`，按 copy-last 同口径归一化，初始权重使其占总 loss 约 5%。
- 与 L2 目标一致（GT 在这些帧上本来就不动），但把"何时踩实"的相位信息显式交给模型。
- 判据：步行子集的脚速、滑行帧比、迈步相位相关改善，L2 不劣化超过 0.5%。

### A6 InterMixer：端到端规范系双人 DCT-Mixer（大胆）

- 取代冻结 base + refiner。输入为规范系下的位移历史（padding 到 60 帧后做 DCT），加末帧姿态、相对几何、动作 FiLM、role embedding。
- 骨干：时间混合（系数轴 FC）+ 通道 MLP + 人物混合（零初始化）的 Mixer 块。
- 两个输出头：root（低阶 DCT）与局部姿态（中阶 DCT），IDCT 后乘 saturate ramp 加观测末帧；零初始化使初始输出等于 copy-last。可选接 A4 的投影。
- 参数约 1.5–2.7M。
- 假设：两级冻结结构本身是瓶颈；在规范系里，小而强归纳偏置的模型在 1758 条序列上更合适（siMLPe、EMPMP 的经验）。

### A7 Mirror：镜像增广 + 指示 embedding（低优先级）

在最佳候选上开镜像增广（p=0.5），并加 `nn.Embedding(2, d)` 的镜像指示（测试时恒为 0）。

## 4. 实验计划

### 4.1 公共设置

- 数据：新缓存管线，manifest `results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json`。
- 训练：batch 8，lr 3e-4，s2_5 损失配方（`--loss_scale_normalize --local_velocity_loss_weight 1 --ramp_mode saturate --dct_low/mid_amplitude_loss_weight 1 --root_loss_weight 0.1 --inter_loss_weight 0.01`），EMA 0.999，`--device cuda:0`。
- 筛选：5000 step × seed 0/1/2，报告 `ema/` 终点在 val 上的结果，与 A0 同 seed 配对。
- 评估：`xyz_mse / xyz_mae / mpjpe` 与摆动 gate 在原相机系计算（复用 `compute_ntu_xyz_metrics`、`compute_ntu_articulation_metrics`）；自然度指标用 `utils/ntu2p_naturalness.py`。

### 4.2 预登记的采纳规则

- **L2**：配对 Δmpjpe 均值 ≤ −0.8%，且 3 个 seed 都为负；xyz_mse 与 xyz_mae 均值不升。
- **摆动 gate**：3/3（`dct_low ≥ 0.40`、`dct_mid ≥ 0.10`，mpjpe 相对 base 回退 ≤ 5%）。
- **自然度（新增，作为否决项）**：步行子集的"GT 站定帧脚速"与"滑行帧比"不得比 A0 差超过 10%；身体骨长相对误差不得上升。任何候选都要看审查图（随机 12 例 + 位移最大的 6 例）后才能采纳。
- A4、A5 以自然度为主要目标：L2 允许在 ±0.5% 内，自然度指标须明显改善。

### 4.3 阶段

1. **Stage 0（GPU 空闲前，CPU）**：实现全部代码；CPU 冒烟测试（前向/反向、首帧误差为 0、默认开关与旧 refiner 输出逐位一致、投影后骨长恒定）；在 CPU 上建检索库并报告 kNN-only 的 val 数字（排除 / 不排除同受试者）。
2. **Stage 1**：A0、A1、A2、A3、A6 各 3 seed × 5000 step（15 个 run）。
3. **Stage 2**：在 Stage 1 最佳结构上叠加 A4、A5、A7（9 个 run）。
4. **Stage 3**：采纳的组合 10000 step × 3 seed，EMA 终点；只对最终方案评估一次 test；导出 test/val 视频与审查图并亲自看图。

GPU 纪律：另一会话的训练结束前（预计 17:30–18:00）不使用 GPU。之后单 run 显存约 1–2 GB，可两个 run 并行。

## 5. 代码布局（全部新增，不改现有入口的行为）

| 文件 | 内容 |
|---|---|
| `model/forecasting_ntu2p_v2.py` | 规范系 refiner（A1/A2），并挂接检索锚点（A3）与骨架投影（A4） |
| `model/forecasting_ntu2p_intermixer.py` | A6 端到端 Mixer |
| `utils/ntu2p_kinematic_projection.py` | A4 投影层；A5 的脚接触掩码与损失 |
| `data_loaders/forecasting/ntu2p_retrieval_bank.py`、`scripts/build_ntu2p_retrieval_bank.py` | A3 检索库 |
| `train/train_ntu2p_v2.py` | 缓存管线上的统一训练入口（`--arch`），复用 `train_ntu2p_residual_refiner_xyz` 的损失函数 |
| `eval/eval_ntu2p_v2.py` | 统一评估：L2、摆动 gate、自然度，可导出 pred 数组给审查图 |
| `scripts/run_ntu2p_v2_screen.py` | Stage 1–3 串行驱动与汇总 |
