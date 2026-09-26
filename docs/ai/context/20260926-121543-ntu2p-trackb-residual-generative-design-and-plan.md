# NTU2P Track B：v2 底座上的交叉拟合残差扩散（ResDiff-OOF+）设计与计划

本文是"下一步候选"的第 3 批。第 1 批（受试者留出 val）和第 2 批（手指去重）正在 cuda:0 上运行。本文只含设计与计划。实现可以先在 CPU 上完成，训练要等第 1、2 批释放 cuda:0，并且必须显式指定 `--device cuda:0`。

背景文档：
- `docs/ai/context/20260926-005610-ntu2p-v2-final-result.md`
- `docs/ai/context/20260926-100701-ntu2p-v2-mainline-adoption-decision.md`
- `docs/ai/context/20260926-105152-ntu2p-subject-holdout-val-plan.md`
- `docs/ai/context/20260822-115553-ntu2p-diffusion-vs-independent-baseline-deep-audit.md`
- `docs/ai/context/20260902-145142-ntu2p-articulation-recovery-design.md`（E 节）

## 0. 目标与边界

**要解决的 v2 局限**（test 数字）：
- 起步者先迈哪只脚、步频是多峰的：步数 1.34，GT 为 2.19；起步者相位相关的上界只有约 0.4。
- 出拳等击打动作的手臂幅度偏小，这是回归到均值的结果。
- 滑行帧比略高于 A0。

**Track B 的定位**：附加的随机模式，每个观测给出 K 个合法、彼此可区分的未来样本。
- **确定性主数字仍是 v2（test mpjpe 0.1810 ± 0.0004）**，Track B 单独成表，不替换它。
- 不指望降低单样本 L2。若生成器校准，单样本 MSE ≈ v2 的 MSE + 条件方差；HumanMAC 等方法的单样本 L2 也只和 zero-velocity 相当。

**协议**：
- 选型在受试者留出协议 `ntu2p_v2subj` 上进行：
  - manifest：`results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json`；
  - 缓存：`results/forecasting/ntu120_label/ntu2p_xyz_seq_cache_subjval/`；
  - H-val 共 247 条。
- 底座是该协议下训练的 v2 主线：`save/forecasting/ntu120_label/ntu2p_v2subj_A6-F-A5f0.05-A4-GH0.5_s{0,1,2}_10000/ema/model000010000.pt`，生成器 seed i 与底座 s_i 配对。
- 采纳后在原协议（原 train 1758 条）上重做交叉拟合并重训生成器，底座换成原 v2 主线 `save/forecasting/ntu120_label/ntu2p_v2_A6-F-A5f0.05-A4-GH0.5_s{0,1,2}_10000/ema/model000010000.pt`，test（1253 条）只评估一次。
- 第 1 批的有效性检验即使未通过，Track B 仍用受试者留出 val。原因：该检验针对确定性模型的 L2 Δ 能否预测 test；而 Track B 的离散度校准要求 val 受试者未在 train 中出现，这正是 test 的情形。受试者重叠的旧 val 会让校准偏乐观。

## 1. 三份设计的比较与裁决

| 维度 | A：ResDiff-OOF | B：ResFlow-Mixer | C：ResFlow-DCT |
|---|---|---|---|
| 能否表达多峰步态 | 能。root + 21 关节的 k0–11 双人联合生成，root 可变，因此起步时机和步频都能变 | 不能。root 与 DC 固定为 v2；起步者 v2 root 到 2.5 s 只走了 GT 的 72%，腿若按 GT 步频迈步就会与慢 root 冲突而滑行，否则步数仍偏少（作者自列风险 2） | 能。root/腿/臂/躯干分组 PCA，含 DC |
| 能否表达手臂幅度 | 能，含 DC | 弱。去掉 DC，且手臂再乘 λ≈0.5，作者自述只能部分恢复 | 能，含 DC |
| 与 v2 一致 | ramp + 骨架投影 + 刚体手 ✓；用朝向对方系；root 锚定靠替换式 inpainting，只是近似条件 | 个人朝向系 ✓；最小二乘编码 ✓ | 个人系 ✓；但用 hook 读 v2 隐藏层 |
| 训练稳定与成本 | 复用 ReGenNet 扩散 ✓；OOF ✓；但 ddim10 从 t=900 起步（ᾱ=0.024），与训练分布不对齐 | 条件流匹配最简单 ✓；OOF ✓ | 无交叉拟合：in-sample 残差按部位收缩 0.61–0.81，单个 τ 校准不了；隐藏层条件容易记忆，且折模型与部署模型的潜空间不对齐 |
| 评估是否诚实 | 以高斯-OOF 为 ES 参照 ✓；但有三处问题：先迈脚用 best-of-5 覆盖，而 bootstrap 就能到 0.96–1.0，没有区分力；mean-of-20 ≤ 1.005×v2 会否决完美校准的生成器；A4 单样本护栏与 A7 校准互斥 | ClimRes 参照 ✓；Brier/CRPS ✓；但对样本做 λ 线性缩放，既破坏校准又混合相位 | 在相同单样本 L2 下与 bootstrap 包络比较 ✓，最严格；Brier ✓；静止人与分配比 ✓；但在 val 上从 8 档温度中挑，研究者自由度大 |

**裁决**：以 A 为骨架，理由如下。
- 表示范围最完整（含 root 与 DC），直接复用 ReGenNet 的 x0 预测、余弦调度与 DDIM。
- 零初始化使初始输出即 v2。
- OOF 交叉拟合从原理上修正 in-sample 残差的收缩。

**从 B 嫁接**：
- 个人朝向系（与 GH 腿部流同一个系，"前/后""左/右脚"对所有人含义一致）；
- 最小二乘斜坡 DCT 编码，使编码与解码互逆；
- bootstrap 基线按"v2 预测步行状态"分层；
- 在原协议上重做同结构的 3 折交叉拟合。

**从 C 嫁接**：
- 在相同单样本 L2 下与 bootstrap 包络比较（主判据）；
- 三类先迈脚 Brier，取代 any-of-K 覆盖；
- 静止人样本的步数与分配比，检查"不该走的人被采样出迈步"；
- 穿插帧比、手臂幅度对数偏差、逐窗 bootstrap 置信区间。

**本裁决新增的修正**：
1. root 锚定改为**训练式条件化**：以 p=0.25 的概率给出干净的 root 系数，并加 root-known 嵌入。这样模型学到的是 p(局部 | root)，替换式 inpainting 只能近似它，而这样做不增加成本。
2. 采样步距改为 `space_timesteps(1000,[10])` = {0,111,…,999}，从 ᾱ=2.4e-9 的纯噪声起步，消除历史失败中"训练/采样不对齐"那一类问题。
3. 点估计判据改用**单步条件均值** x̂0(x_T, t=999)。mean-of-K 带有 var/K 项：完美校准的生成器在 K=20 时 mse 理论上 +5%，A、B、C 的 1.005–1.02 阈值都会误否。
4. **两级温度**：
   - 概率模式 P（mode F，τ_P）要求校准；
   - 展示模式（mode R，τ_D）受单样本 L2 护栏约束，τ_D 按预登记规则取值。
   - 校准的采样器在只采样局部时，单样本 mpjpe 约 +15%（旧 val 的 bootstrap 数据），不可能同时满足校准与 ≤1.05×v2，所以两者必须分开判定。

**不采纳的部分**：
- B 的确定性 root 与去 DC、λ 线性缩放；
- C 的隐藏层条件、in-sample 目标与 8 档温度挑选；
- A 的朝向对方系残差、替换式 inpainting、best-of-5 覆盖判据、从 t=900 起步的 ddim10；
- 主臂不用 L_inter，不用 CFG（会压缩多样性）。

## 2. 设计依据（CPU 实测，未看 test）

### 2.1 设计阶段三份分析（旧 val 198 条，v2 GH s0；脚本在 scratchpad/trackb/）

**in-sample 收缩**：
- v2 在 train 上 mpjpe 0.130 / mse 0.0120，旧 val 0.162 / 0.0189，test 0.181 / 0.0277。
- train/val 的残差能量比：root 0.61、手臂 0.69、手指 0.65、腿 0.81。收缩按部位不均匀。

**残差分布**：
- 残差主要在低频：局部 ≤1 Hz 约占 90%，1.2–2 Hz 约占 5–6%。
- oracle 只修正某一部位时的 mpjpe（v2 为 0.1617）：只修 root 0.1210（−25%），只修手臂 0.1126（−30%，因为手指跟随手腕），只修腿 0.1592（−1.6%）。**腿部多样性在整窗 mpjpe 中几乎看不出来，必须用腿专属指标判定。**
- 局部残差能量：手臂 DC 3.72、k1–5 2.59；腿 DC 1.62、k1–5 2.40。body21 修正 k0–10 的 oracle mpjpe 为 0.1088，只修 k1–10 为 0.1450。**DC 对手臂幅度必不可少。**

**平凡随机基线**（bootstrap，同动作类）：

| 设置 | 单样本 mpjpe | ES | minADE@10 |
|---|---|---|---|
| s=0.5 | 0.1729（+6.9%） | 0.1328（−17.9%） | 0.1530（−5.4%） |
| s=1.0 | 0.2012（+24%） | 0.1220（−24.6%） | — |
| root 锚定 s=0.5 | +4.1% | — | — |
| root 锚定 s=1.0 | +15.5% | — | — |

- 先迈脚 any-of-10 覆盖：bootstrap 为 0.96–1.0，没有区分力。Brier：v2 0.217，bootstrap 0.168–0.177。
- 高斯残差能把步数刷到 1.80（GT 1.85），但 GT 站定帧脚速从 0.18 升到 0.38；滑行帧比反而降到 0.11（v2 0.21），可见噪声能让滑行帧比变好看。因此必须同时看自洽的脚滑指标与审查图。
- 双人残差耦合：终帧 root 残差 A/B 相关 0.23–0.28，局部能量的 log 相关 0.34–0.40。**双人要联合生成。**

### 2.2 裁决阶段补测

**本方案编解码的原型**（scratchpad/trackb/codec_proto.py，旧 val 198，v2 s0）：
- 表示能力：个人系、最小二乘斜坡 DCT、root + 21 关节、K=12、手指/下颌/眼睛随动，oracle mpjpe **0.0367**，v2 为 0.1617（K=10 时 0.0380）。只修局部、root = v2 的 oracle mpjpe 0.1071。
- 零系数：零系数时输出与 v2 的最大差 **4.3e-6**，这是骨架投影重复作用的浮点误差。
- 首帧误差严格为 **0**。
- root 系数为 0 时，输出 pelvis 与 v2 **torch.equal**。

**跳变窗**（任一人 pelvis 帧间位移 > 0.25 m/帧，即 5 m/s）：
- subjval train 的 stride-1 窗口：30,214 个中占 **12.8%**；
- H-val 中心窗：**24/247**；
- test 中心窗：**125/1253**。

**采样步距**：`space_timesteps(1000,"ddim10")` = {0,100,…,900}，ᾱ(900)=0.0236，起点 x_900 = 0.154·x0 + 0.988·ε，从纯噪声起步不对齐；`space_timesteps(1000,[10])` = {0,111,…,999}，ᾱ(999)=2.4e-9。

**交叉拟合折**：受试者按序列数降序、贪心放入当前最轻的折。
- subjval train：38 名受试者，三折 569/572/568，P008 在第 0 折；
- 原 train：53 名受试者，三折 587/586/585。

**成本参照**：第 1 批 3 个 v2subj run 并行训练 10000 step，墙钟约 51 分钟。

## 3. 模型结构

### 3.1 底座与配对

冻结 v2（A6-F-A5f0.05-A4-GH0.5，EMA 终点），用 `model/forecasting_ntu2p_v2.py:load_ntu2p_model_checkpoint` 加载，得到骨架投影后的相机系输出 P [B,50,2,55,3]。

| 阶段 | 所用底座 |
|---|---|
| 训练 | 交叉拟合的折模型 F_{-f}：同配方，seed 0，10000 step，EMA 终点 |
| 选型评估 | v2subj s_i |
| 最终评估 | 原主线 s_i |

生成器只读 v2 的**输出**，不读隐藏层。原因：折模型与部署模型的潜空间不对齐，读隐藏层还会让生成器在 train 上认出序列、记住未来。

### 3.2 坐标系

先用 `canonical_frame(obs, ab_fallback="camera_x")` 得到场景规范系（+Y 为上），再用 `individual_frames(to_canonical(obs, frame))` 得到每人的个人朝向系：由观测末 3 帧的朝向之和定前向，三行依次为前、左、上，与 GH 腿部流是同一个系。残差是向量，只做旋转，不加平移。

### 3.3 残差参数化

- **通道**，每人 66 维：
  - pelvis 残差 3 维；
  - 关节 1..21 的局部残差（该关节残差 − pelvis 残差）63 维。
- **随动关节**：22–24（下颌/眼）随头（15）的残差；手指 25–39 随左腕（20）、40–54 随右腕（21）的残差刚性平移。手部朝向沿用 v2，由投影中的 Procrustes 保持刚体模板。
- **时间基**：B[k,t] = ramp(t)·φ_k(t)，k = 0..11（φ_k 为 50 帧正交 DCT-II，频率 0.2k Hz，最高 2.2 Hz，覆盖 1.2–2 Hz 步频），ramp 为 `build_residual_ramp(50,"saturate",5)`，与 v2 相同。
- **编码与解码**：
  - 编码用最小二乘：c = (BBᵀ)⁻¹B r；
  - 解码：r̂ = Bᵀc；
  - 编码与解码严格互逆。
- **目标与草稿**：
  - 目标 x0：GT − P 在个人系中的系数；
  - 草稿 d：P − obs_last 在同一组基上的系数，作为条件；
  - 两者都是 [2, 12, 66]。
- **归一化**：每个 (k, 通道) 分别除以 OOF 干净窗口上的标准差 σ_target、σ_draft，两人合并统计，下限为中位数的 5%，不减均值。

### 3.4 解码与结构保证

y = SkeletonProjector()(P + R_sceneᵀ R_personᵀ r̂, obs_last)

- **首帧**：B[:,0] = 0，所以 r̂ 的首帧严格为 0；y 首帧 = Proj(v2 首帧) = 观测末帧，误差 ≤ 1e-6。
- **骨长**：等于观测末帧的骨长，由 v2 A4 的同一个投影器保证。
- **刚体手**：由投影器保证。
- **初始即 v2**：x̂0 ≡ 0 时 y 与 v2 的差 ≤ 1e-5。
- **mode R**：root 系数恒为 0，输出 pelvis 与 v2 torch.equal。
- **不引入高频**：带宽 ≤ 2.2 Hz，不给 v2 增加高频抖动。

### 3.5 条件（只来自观测、动作与 v2 输出）

- `draft`：d / σ_draft，[2,12,66]。
- `obs_feats`：观测 10 帧在个人系中的 [pelvis(t) − pelvis(last)，关节 1..21 − pelvis(t)]，66 维；按 OOF 干净窗口上的逐通道标准差标准化。
- `rel_geom`：`relative_geometry(规范系末帧)`，6 维旋转不变量。
- `action`：26 类；角色（施动/受动）用 role embedding 表示，不交换 A/B。

### 3.6 去噪网络 NTU2PResDiffDenoiser

参照 CMDM 的 trans_enc，约 4.9M 参数。

**token 共 47 个，d=256**：

| token | 数量 | 构成 |
|---|---|---|
| 全局 | 1 | TimestepEmbedder(t) + 动作嵌入（CMDM 的 add 模式，复用 `model/cmdm.py`） |
| 几何 | 2 | 每人一个：Linear(6) + 角色嵌入 |
| 观测 | 20 | 每人 10 个：Linear(66) + 帧位置嵌入 + 角色嵌入 |
| 系数 | 24 | 每人 12 个：Linear(132)([x_t; d])，同阶频率对齐，+ 系数位置嵌入(k) + 角色嵌入 + root-known 嵌入（仅在给定 root 时加） |

**主干与输出**：
- 主干：`nn.TransformerEncoder`，6 层、4 头、ff1024、GELU、dropout 0.1。两人的全部 token 做全连接注意，以此表达 A/B 残差的耦合。
- 输出：系数 token 经 LayerNorm → Linear(256→66)，**权重与偏置零初始化**，输出 x̂0（归一化空间）。
- 给定 root（root_known）时：
  - 输入 x_t 的 root 通道先替换为 root_value；
  - 输出 x̂0 的 root 通道也替换为 root_value。

### 3.7 扩散与采样

**训练扩散**：
- 调度：`gd.get_named_beta_schedule("cosine", 1000)`；
- 模型：`GaussianDiffusion(START_X, FIXED_SMALL, MSE)`；
- t 由 `UniformSampler` 均匀采样，加噪用 `q_sample`；
- 损失自写，不用 `training_losses`：它绑定了 rot2xyz/SMPL。

**采样扩散**：
- `SpacedDiffusion(space_timesteps(1000,[N]))`，N=10 为主，含 t=999；
- 调用 `ddim_sample_loop(model, shape, noise=τ·ε, clip_denoised=False, model_kwargs={"y": cond}, device=…, eta=0.0)`；
- clip_denoised 必须为 False，否则单位方差的 x0 会被截到 ±1；
- 噪声 ε 在 CPU 上按 manifest 顺序一次性生成，形状 [N_win, K, 2, 12, 66]，所有变体共用，保证配对比较。

**诊断**（只作诊断，不参与选择）：
- 单步条件均值：对 K 个噪声取 x̂0(x_T, t=999) 的均值；
- teacher-forced 下的分 t 误差；
- NFE 取 5/20/50 时的对比。

### 3.8 采样模式与温度

同一个模型支持两种模式：
- **mode F（联合）**：root 与局部都采样。它是**概率模式 P**，温度 τ_P ∈ {1.0, 0.8}，按 5.4 的规则确定。
- **mode R（root 锚定）**：root_known=1、root_value=0，即 root 逐位等于 v2，只采样局部（先迈哪只脚、步频与步幅的组合、手臂幅度）。它是**展示模式**，温度 τ_D ∈ {1.0, 0.8, 0.6, 0.4}，按 5.4 的规则确定。
- 温度只缩放初始噪声（DDIM 是确定性 ODE，缩放初始噪声等于降温采样）。不对样本做 λ 线性缩放，也不用 CFG。

## 4. 训练

### 4.1 交叉拟合折（新脚本 `scripts/build_ntu2p_crossfit_folds.py`）

- **分折**：把协议 train 按受试者分 3 个不相交的折，规则见 2.2（序列数降序、贪心、同数量按受试者 id）。
- **折 manifest**：第 f 折的 train = 其余两折，val = 第 f 折，test 与源 manifest 相同。
  - 保持源 manifest 中的顺序；
  - hash 用 `manifest_payload_hash` 重算；
  - 断言三折受试者两两不重叠、并集等于源 train。
- **缓存**：从协议 train 缓存按序列切片，不重算 FK。只生成 train 与 val：折模型不评估 test，这一点写入 split_config。

### 4.2 折模型

- 用现有驱动训练：`scripts/run_ntu2p_v2_screen.py --stage 1 --configs A6-F-A5f0.05-A4-GH0.5 --seeds 0 --steps 10000 --workers 1 --manifest_path <折 manifest> --cache_dir <折缓存> --baseline_checkpoint <协议独立 base> --run_prefix ntu2p_v2fold{f}_{protocol} --summary_dir …`。
  - 3 折同时起 3 个进程，墙钟约 50 分钟；
  - 训练入口与驱动一律不改。
- 驱动会顺带评估每折的 val，结果只作 OOF 误差诊断。其中 base 指标用的协议 base 见过该折受试者，只作参考。

### 4.3 OOF 残差库（新脚本 `scripts/build_ntu2p_oof_residual_bank.py`）

- **遍历范围**：协议 train 每条序列的全部 stride-1 窗口，subjval 为 30,214 个，原协议为 30,908 个。
- **推理**：每个窗口用"没见过该受试者"的折模型前向。断言该窗口的受试者不在该折模型 checkpoint 所记录 manifest 的 train 中，且该 manifest 的 hash 等于折 manifest 的 hash。
- **存储字段**：
  - 索引：seq_index、start、fold、performer、action；
  - 标记：clean（非跳变）、glitch；
  - 目标：target 系数（米）；
  - 条件：draft 系数、obs_feats、rel_geom；
  - v2_walk：v2 预测 f50 的 pelvis 水平位移 ≥ 0.5 m；
  - base_mpjpe（诊断）；
  - 统计量：σ_target、σ_draft、feat_std，只在 clean 窗口上计算。
- **显存**：fp32 约 0.6 GB，可常驻 GPU。

### 4.4 跳变过滤（按物理阈值预登记，不看 val）

- **规则**：窗口 60 帧内任一人 pelvis 帧间位移 > 0.25 m（即 5 m/s）的窗口，不进入生成器训练，也不参与统计量估计。subjval train 中占 12.8%。
- **理由**：这类窗口约 1/3 是 A/B 身份交换，不过滤的话生成器会学到"瞬移"。
- **不受影响的部分**：v2、折模型与全部评估都不过滤；评估另报干净子集。

### 4.5 损失与 root 条件化

- **主损失**：L = mean((x̂0 − x0)² ⊙ mask)，在归一化空间计算，对全部 t 均匀施加。
- **root 条件化**：每个样本以 p=0.25 标记为 root_known。这类样本的输入 root 通道等于干净的 x0 root，损失 mask 掉 root 通道。
- **主臂不加的损失**：几何、脚接触、L_inter 一律不加。
  - 首帧、骨长、刚体手已由结构保证；
  - 基于 GT 接触帧的脚损失会把样本拉回 GT 的相位；
  - 历史上 L_inter 约为 L_dm 的 10 倍，压制了个体拟合（20260822 深度排查）。

### 4.6 超参

- 优化：AdamW，lr 2e-4，wd 1e-4；前 500 step 线性预热，之后恒定；梯度裁剪 1.0；dropout 0.1。
- EMA 0.999，**只报告 EMA 终点**，不在 val 上挑 checkpoint。
- batch 64，20000 step，seed 0/1/2。
- 每 5000 step 存一次 checkpoint 与 EMA，仅供诊断。
- 必须显式 `--device cuda:0`，禁止静默回退 CPU；冒烟时显式加 `--allow_cpu_for_smoke_test`。
- 残差库常驻 GPU，预计 30–60 step/s，单个 seed 约 6–12 分钟，3 个 seed 并行约 15 分钟。

### 4.7 训练诊断（不参与选择）

- 训练日志：每 100 step 记录总损失，按通道组（root/腿/臂/其余）与按 t 四分位的损失，梯度范数，学习率；应急臂另记录各项的 `*_share`。
- seed 0 的 4 个 EMA checkpoint：在 H-val 上计算 teacher-forced 的归一化 MSE（t ∈ {50, 250, 500, 750, 999}），以及 APD 的 train/val 比。

### 4.8 应急臂（预登记；只在 E 类自然度护栏失败时启用，每类只跑一轮）

- **C1 foot**：对应滑行、脚滑、穿地/悬空护栏失败。
  - 对 t < 300 的样本，把 x̂0 解码并投影；
  - 加 `foot_skate_loss`（GT 接触帧），按"纯滑行参考"（`slide_reference`）归一化，w=0.05；
  - t 很小时 x_t 已带有 GT 的相位，这个损失只要求解码侧踩实，不改变高噪声阶段对相位的采样。
- **C2 inter**：对应穿插护栏失败。
  - 损失为 ReGenNet 式 L_inter 的 xyz 版本：投影前 body22 的相对关节向量 (x_A − x_B) 的 MSE，加 pelvis 相对平移；
  - 除以 x̂0=0（即纯 v2）在残差库上的值归一化，w=0.1；
  - 日志的 share 若持续超过 15%，视为尺度失控。
- **实现方式**：两者都需要解码到 xyz，因此在训练时在线运行对应的折模型（no_grad）得到 P，需要读 cache。
- **不设应急臂的情况**：jerk 或高频护栏失败时，先看 NFE 规则；仍失败则判为未通过，不追加迭代。

### 4.9 in-sample 对照（诊断，只跑 seed 0）

- 做法：用 v2subj s0 在它自己的 train 窗口上的残差建 in-sample 库，其余与主臂相同。
- 预登记假设 H_cf：in-sample 生成器在 H-val 上的 SSR（root/腿/臂的均值）比 OOF 主臂低至少 0.15，即交叉拟合是必要的。
- 它不是采纳判据。

## 5. 预登记评估协议

### 5.1 数据、配对与噪声

- **选型**：H-val 的 247 个中心窗；生成器 seed i 与 v2subj s_i 配对，报告 3 个 seed 的均值 ± 标准差，以及逐 seed 值。
- **采样**：每窗 K=20；噪声 torch.Generator(0) 在 CPU 上一次性生成；bootstrap 的抽样用 Generator(1)。所有变体共用同一份噪声。
- **子集**：全体窗口为主口径，干净窗口为次口径，两者方向须一致。关键指标另报逐窗 bootstrap 的 95% 置信区间（1000 次重采样）。
- **人物分组**：沿用 `gait_phase_stats` 的定义。
  - 步行人：GT f50 的 pelvis 水平位移 ≥ 0.5 m。
  - 已在走：观测末 5 个帧差的平均速度 > 0.2 m/s。
  - 起步者：步行人中未在走的。
  - 静止人：GT 位移 < 0.1 m，且观测速度 < 0.1 m/s。
  - 手臂动作人：动作 ∈ ARM_ACTIONS {0,2,3,4,6,8,11,12,13,14,15,17,18,23,24,25}，且 GT 手臂幅度 ≥ 0.1 m。
  - 击打类：动作 ∈ {0 出拳, 2 推, 11 持物击打, 12 持刀, 13 撞倒}，且 GT 手臂幅度 ≥ 0.1 m。

### 5.2 参照

- **确定性**：v2（部署底座 s_i）、copy-last、独立单人 base（subjval base）、A0（`ntu2p_v2subj_A0_s{i}_10000` 的 EMA 终点，存在时报告）、v2 的 3 seed 输出均值（诊断）。
- **平凡随机基线 bootstrap**（免训练，最关键的参照）：
  - 做法：v2 加 s × 从 OOF 干净库中抽出的整场景 target 系数；解码与投影与生成器完全相同。
  - 分层：先按 (动作, v2_walk_A, v2_walk_B)；该层少于 30 个窗口时回退到只按动作，再回退到全体。
  - s ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}（s=0 即 v2），分 mode F 与 mode R（root 系数置 0）两种。
- **v2 3 seed 样本集**（K=3）：报告 ES 与 minADE@3，仅作诊断。

### 5.3 指标定义

全部在相机系上计算；默认 55 关节；"局部"指关节减去同帧 pelvis。

1. **单样本 L2**：K 个样本分别算 xyz_mse / xyz_mae / mpjpe / first_step_error，再对 K 取平均。另按 root 与局部分解，并报与 A0 的比值。
2. **mean-of-20 L2**：先对 K 个样本平均再算 L2。完美校准时理论上 mse 约 +5%、mpjpe 约 +2.5%，与数字一起写明。
3. **单步条件均值 L2**：x̂0(x_T, 999) 对 K 个噪声取均值后解码，是生成器对条件均值的估计。
4. **best-of-K**：
   - 场景级 minADE@K，K = 1/5/10/20 画成曲线，A/B 必须取同一个样本；
   - minFDE@10；
   - 人物级 body22 minADE@10。
   - **这些是用 GT 挑样本得到的上界，不是预测。**
5. **能量分数 ES**：
   - 逐窗计算 mean_{t,p,j}[(1/K)Σ_k d(x_k,y) − (1/(2K(K−1)))Σ_{k≠l} d(x_k,x_l)]，d 为欧氏距离。K=1 时 ES 等于 mpjpe。
   - ES_legs_walk：步行人的腿局部；ES_arms_act：手臂动作人的臂局部；ES_joint_clean：干净子集上的 ES。
6. **先迈脚 Brier**（三类）：
   - 定义：每条轨迹取自身第一个 |s(t) − s_last| > 0.1 m 的帧，其中 s 为左右踝沿 GT 位移方向的前后分离，按该帧的符号记为 L 或 R；没有这样的帧记为"无"。
   - 计算：对起步且 GT 有事件的人，Brier = Σ_c (p̂_c − 1[c=GT])²；同时报 p̂ 的熵（bit）。v2 的 p̂ 是 one-hot。
   - 另报 naturalness 口径的 `gait_lead_foot_acc_starting`。
7. **步数**：
   - 步行人：沿 GT 位移方向计数，复用 `gait_stats` 的平滑与迟滞 0.03 m；静止人：沿观测朝向计数。
   - 报告：样本 CRPS（E|X−y| − ½E|X−X'|）、样本池与 GT 的 W1、均值，以及静止人样本的平均步数。
8. **多样性**：
   - APD，即样本两两之间的窗 mpjpe 均值，拆成全部、root、腿局部、臂局部；
   - 分配比 = 步行人腿局部 APD / 静止人腿局部 APD。
9. **校准 SSR**：只在干净子集上算，分 root、腿局部、臂局部三组；SSR = sqrt((K+1)/K) × 集合离散度的 RMS / 集合均值的 RMSE。
10. **单样本自然度**：
    - 计算方式：对前 5 个样本槽分别调用 `eval/eval_ntu2p_v2.naturalness_block`，同时带入 GT、copy_last、v2 三行，再取平均。
    - 覆盖的指标：bone_rel_err_body(_max)、skate_gt_contact_speed_walk（GT 站定帧脚速）、skate_self_ratio_walk、slide_frame_ratio、foot_penetration_ratio / foot_float_ratio、jerk_body、步数、gait_sep_corr f01_10 / f11_20 / f21_30、gait_lead_foot_acc(_starting)、leg_swing_amp_ratio_to_gt、root_distance_abs_err、min_interperson_dist_abs_err。
    - 另外：dct_low / mid / high 的比值，由 `compute_ntu_articulation_metrics` 与 `articulation_ratios` 对每个样本槽计算；穿插帧比，即双人 body22 的最小距离 < 0.05 m 而 GT ≥ 0.15 m 的帧比例。
11. **手臂幅度**：A = 未来帧中左右腕相对 pelvis 的位移（相对观测末帧）的最大值。报告击打类与手臂动作类的 |E log(A_s/A_GT)| 与比值的中位数。
12. **扩散诊断**：
    - NFE 取 5/20/50 时 mode F τ=1 的单样本 L2、ES 与 minADE@10；
    - teacher-forced 下的分 t 误差；
    - first_step_error 的最大值、mode R 下 pelvis 与 v2 的 torch.equal 结果。

### 5.4 预登记流程

按顺序执行，全部只看 H-val：

1. **结构自检**：必须在任何判据之前全部通过，失败即判为实现错误并停止。
   - first_step_error ≤ 1e-6；
   - bone_rel_err_body ≤ 1e-5；
   - mode R 下 pelvis 与 v2 相同；
   - 样本全部有限；
   - 复跑 seed 0 的前 16 窗，结果逐位相同。
2. **NFE 规则**：若 mode F τ=1 下 NFE20 的 ES_joint ≤ 0.99 × NFE10，且 3/3 seed 成立，主采样改用 20 步（重跑完整评估）；否则用 10 步。
3. **τ_P 规则**：τ=1 时 root/腿/臂 SSR 的中位数（3 seed 均值）若 > 1.4，判为过散，τ_P = 0.8；否则 τ_P = 1.0。
4. **τ_D 规则**：在 {1.0, 0.8, 0.6, 0.4} 中取满足以下条件（3/3 seed）的最大值：mode R 单样本 mpjpe ≤ 1.05×v2，且 xyz_mse ≤ 1.10×v2。都不满足则展示模式不可用。
5. 在选定的设置下逐项判定 A–H。
6. 按审查图规则渲染并作出审查判断，结论写入文档之后才允许动 test。

### 5.5 采纳判据

"3/3"表示三个配对 seed 各自满足，其余看 3 seed 均值；比较对象 v2 是同 seed 的部署底座。

**概率模式 P（mode F，τ_P）**

- **A. 胜过平凡基线**：在生成器的单样本 mpjpe 处，对 bootstrap-F 的 (单样本 mpjpe, 指标) 曲线做分段线性插值，要求：
  - A1：ES_joint ≤ 0.98 × 包络，3/3；
  - A2：minADE@10 ≤ 0.97 × 包络，3/3；
  - A3：ES_legs_walk ≤ 0.95 × 包络；
  - A4：先迈脚 Brier ≤ 包络 − 0.02，且低于 v2；
  - A5：ES_arms_act ≤ 0.97 × 包络。
  - 超出 s 网格时按外推值判定，并标注"超出网格"。
- **B. 点估计不劣**：
  - 单步条件均值：mpjpe ≤ 1.01×v2，xyz_mse ≤ 1.02×v2；
  - mean-of-20 的 mpjpe ≤ 1.04×v2，只作健全性检查。
- **C. 覆盖**：minADE@10 ≤ 0.92 × v2 mpjpe，3/3。
- **单样本健全性**：单样本三项 L2 都低于 copy-last 与独立 base，3/3。
- **E. 单样本自然度护栏**（前 5 个样本）：
  - E1：bone_rel_err_body ≤ 1e-5，first_step_error ≤ 1e-6；
  - E2：skate_self_ratio_walk ≤ 1.15×v2；
  - E3：slide_frame_ratio ≤ 1.10×v2；
  - E4：穿地、悬空比例各 ≤ v2 + 0.01；
  - E5：jerk_body ≤ 1.5×v2，且 dct_high ≤ max(1.2×v2, 0.05)；
  - E6：穿插帧比 ≤ v2 + 0.01；
  - GT 站定帧脚速、min_interperson_dist_abs_err、root_distance_abs_err 在 P 模式下只报告：它们与精度耦合，root 被采样时必然升高。
- **F. 定向收益**（只在 E 通过时计为收益；原因：噪声也能刷出步数）：
  - F1：步行人 |样本步数均值 − GT 步数均值| ≤ 0.7 × v2 的差距，且步数 W1 ≤ 0.8 × v2；
  - F2：起步者先迈脚的熵 ≥ 0.5 bit，且单样本先迈脚正确率 ≥ v2 − 0.10；
  - F3：gait_sep_corr_f01_10 ≥ v2 − 0.05，即前 0.5 s 可预测的相位不被破坏；
  - F4：击打类 |E log(A_s/A_GT)| ≤ 0.7 × v2，且幅度比中位数 ≤ 1.3；
  - F5：静止人样本的平均步数 ≤ v2 + 0.2，且分配比 ≥ 3。
- **G. 校准**：干净子集上 root/腿/臂 SSR 各在 [0.6, 1.4] 内。
- **H. 审查图**：
  - 选例：只按 GT 预先选 18 个窗口：
    - 起步者 6 个：GT 位移最大，且 GT 有先迈脚事件；
    - 已在走 2 个；
    - 击打类 4 个：GT 手臂幅度最大；
    - 其余手臂动作 2 个；
    - `random.Random(0)` 随机 4 个。
  - 渲染：GT、v2、mode F 的样本 0/1/2、mode R（τ_D）的样本 0/1/2。用 `sample/render_ntu2p_review_sheet.py --indices … --video`，另画起步者 20 个样本的左右踝分离叠图。
  - 出现下列任一情况即否决：
    - 肢体变形、抖动、瞬移，或双人穿插；
    - 起步者各样本之间，先迈脚与起步时机看不出区别；
    - 静止人的双脚没有踩实；
    - 步行样本不交替迈步。

**展示模式（mode R，τ_D）**

- D：τ_D 存在；单样本 L2 低于 copy-last；前 5 个样本 dct_low ≥ 0.40、dct_mid ≥ 0.10 的均值过线；腿局部 APD(τ_D) ≥ 0.5 × APD(τ=1)，防止温度过低退化成 v2 的复制品。
- E'：E1–E6 全部满足，并且 skate_gt_contact_speed_walk ≤ 1.5×v2、min_interperson_dist_abs_err ≤ 1.10×v2（root 锚定时这两项与精度解耦得较好）。
- F'：F3 与 F5 必须满足；F1、F2、F4 只报告。

### 5.6 判定

| 情况 | 结论 |
|---|---|
| P 的 A、B、C、E、F、G、H 与单样本健全性全部通过，展示模式 D、E'、F' 也通过 | **完全采纳**：概率模式 + 展示模式；进入原协议复训与一次 test |
| P 全部通过，展示模式未通过 | **只作多假设生成器采纳**，不用于单样本展示；原协议复训与 test 须经用户确认后再做 |
| A 未通过 | **不采纳**，结论为"扩散不优于残差 bootstrap"；不再追加应急臂 |
| E 或 E' 未通过 | 按 4.8 预登记的应急臂只跑一轮，重新按本表判定；仍失败则不采纳 |
| B、C、F、G、H 中任一项未通过（A、E 通过） | 本轮不采纳；记录诊断，不做 test |

### 5.7 汇报约束

- Track B 单独成表，分栏为"单样本 / mean-of-20 / 单步均值 / best-of-K / ES / 自然度"。
- 任何 best-of-K 数字都标注"GT 选样上界"，不得与 v2 的单样本数字并列写成"超过 v2"。
- ES 只与相同单样本 L2 下的 bootstrap 比较，不以"ES 低于 v2 的 mpjpe"作为证据。
- 若单步均值或 mean-of-K 低于 v2，只记为"残差堆叠的附带观察"。要作为确定性结果，须另立协议，不替换主数字。

### 5.8 最终协议与 test

- **流程**：原协议 train 1758 条 → 3 折（587/586/585）的折模型 → OOF 库 → 生成器 3 seed（主臂，以及触发过的应急臂）。
- **底座与冻结设置**：底座为原 v2 主线 s_i；NFE、τ_P、τ_D 冻结为 H-val 上确定的值。
- **评估内容**：test 1253 条，K=20，指标与 H-val 相同，bootstrap 基线来自原协议 OOF 库。
- **只评估一次**：driver 在 test 结果已存在时拒绝重跑。

## 6. 实验矩阵与分批

| 批次 | 内容 | 前置条件 | 资源与时间 |
|---|---|---|---|
| T0 | 实现 11 个新文件；`scripts/check_ntu2p_resdiff.py` 的 CPU 测试全部通过；在 CPU 上生成两个协议的折 manifest 与切片缓存；各阶段 CPU 冒烟 | 无 | 仅 CPU（2 线程、nice 10），约 1 个工作日 |
| T1-a | subjval 的 3 个折模型并行训练 10000 step | 第 1、2 批结束，cuda:0 空闲 | 约 50 分钟 |
| T1-b | OOF 库（约 2 分钟）+ in-sample 库（v2subj s0，约 2 分钟） | T1-a；v2subj s0–2 的 EMA 终点 | 约 5 分钟 |
| T1-c | 生成器主臂 3 seed × 20000 step 并行；in-sample 对照 seed 0 | T1-b | 约 20–30 分钟 |
| T1-d | H-val 评估：6 种 (mode, τ) × K=20 × 3 seed，bootstrap 网格，参照，NFE 扫描，诊断；必要时按 NFE 规则重跑 | T1-c | GPU 约 15 分钟 + 自然度 CPU 约 10 分钟/seed |
| T1-e | 汇总：按 5.4 流程判定；渲染审查图（review_verdict.json）；写结果文档 | T1-d | 约 20 分钟 |
| T2（条件） | 应急臂 C1/C2 各 3 seed，重新评估 | T1 中 E 或 E' 未通过 | 约 45 分钟 |
| T3（条件） | 原协议：3 折模型（约 50 分钟）+ OOF 库 + 生成器 3 seed（约 25 分钟）+ test 评估一次（约 20 分钟） | 5.6 判为完全采纳，或用户确认只作多假设采纳 | 约 1.7 小时 |

T1 合计约 2–2.5 小时墙钟，其中 GPU 约 1.5 小时。所有 GPU 阶段由 driver 检查是否有 `train_ntu2p_v2.py` 进程在运行，有则拒绝启动，除非显式传 `--allow_shared_gpu`。

## 7. 代码布局（全部新建；正在被 GPU 使用的文件一律不改）

| 文件 | 内容 |
|---|---|
| `utils/ntu2p_residual_codec.py` | 坐标系、最小二乘斜坡 DCT 编解码、随动规则、跳变判定、条件特征 |
| `data_loaders/forecasting/ntu2p_residual_bank.py` | 残差库的保存与加载、统计量、干净窗口采样器、bootstrap 分层 |
| `model/forecasting_ntu2p_resdiff.py` | 去噪网络、扩散构建、采样、单步均值、checkpoint 读写 |
| `utils/ntu2p_probabilistic_metrics.py` | ES、minADE/FDE、APD、SSR、三类 Brier、步数 CRPS/W1、手臂幅度、穿插、包络插值、逐窗置信区间、审查图选例 |
| `scripts/build_ntu2p_crossfit_folds.py` | 受试者折 manifest 与切片缓存 |
| `scripts/build_ntu2p_oof_residual_bank.py` | OOF 库与 in-sample 库 |
| `train/train_ntu2p_resdiff.py` | 生成器训练，含应急臂开关（默认关闭） |
| `eval/eval_ntu2p_resdiff.py` | 预登记指标、参照与 bootstrap、审查数组导出 |
| `sample/plot_ntu2p_gait_fan.py` | 起步者多样本的踝分离叠图 |
| `scripts/run_ntu2p_trackb.py` | 分阶段驱动、预登记判定、test 拦截 |
| `scripts/check_ntu2p_resdiff.py` | CPU 自检 |

复用的现有模块（只 import，不修改）：
- 模型与几何：`load_ntu2p_model_checkpoint`、`independent_base_forward`、`load_base_model_from_checkpoint`、`canonical_frame`、`to_canonical`、`apply_linear`、`estimate_up`、`horizontal`、`individual_frames`、`relative_geometry`、`rotate_rows(_transposed)`、`build_residual_ramp`、`dct_matrix`、`SkeletonProjector`、`foot_skate_loss`、`slide_reference`；
- 评估与指标：`compute_ntu_xyz_metrics`、`compute_ntu_articulation_metrics`、`articulation_ratios`、`copy_last_xyz`、`naturalness_block`、`gait_phase_stats` 的口径常量、`_add`、`_finalize`；
- 数据：`NTU2PXYZSeqCache`、`eval_windows`、`manifest_payload_hash`、`assert_manifest_no_sample_id_leak`、`_split_summary`；
- 训练工具：`_ema_model`、`_update_ema`、`_append_log`、`_write_json`、`train_ntu2p_v2._device`；
- 扩散：`diffusion/gaussian_diffusion.py`、`respace.py`、`resample.py`，以及 `model/cmdm.py` 的 `TimestepEmbedder` 与 `PositionalEncoding`；
- 审查图：`sample/render_ntu2p_review_sheet.py`；
- 折模型训练与评估：`scripts/run_ntu2p_v2_screen.py`（以子进程调用）。

产物路径：
- 折：`results/forecasting/ntu120_label/ntu2p_trackb/{subjval|original}/folds/`
- 残差库：`results/forecasting/ntu120_label/ntu2p_trackb/{protocol}/{oof_bank,insample_bank_s0}.pt`
- 折模型：`save/forecasting/ntu120_label/ntu2p_v2fold{f}_{protocol}_A6-F-A5f0.05-A4-GH0.5_s0_10000/`
- 生成器：`save/forecasting/ntu120_label/ntu2p_trackb_{protocol}_{arm}_s{i}_20000/`
- 评估与汇总：`results/forecasting/ntu120_label/ntu2p_trackb/{protocol}/`

## 8. 风险与失败判据

1. **训练底座与部署底座不一致**。折模型只用 2/3 的数据训练，OOF 残差偏大，生成器可能过散；部署底座更准，生成器还可能过度修正。
   - H-val 与部署时是同一结构，B（单步均值）与 G（SSR）能直接检出；
   - 过散时的预登记旋钮是 τ_P=0.8；
   - 若仍失败，下一批考虑 5 折，训练成本约 ×1.7。本批不临时改动。
2. **单样本 L2 的代价是结构性的**。校准的采样器单样本 MSE 约为 v2 + 条件方差；root 锚定、只采局部的 bootstrap 在 s=1 时 mpjpe +15.5%。
   - 展示模式靠 τ_D 规则控制这个代价；
   - 概率模式如实报告，不设 1.05× 护栏。
   - 这是有意的取舍，需要用户知悉。
3. **指标可被刷分**。ES 与 best-of-K 奖励离散，步数和滑行帧比也能被噪声刷好看（高斯基线的滑行帧比甚至低于 v2）。
   - 对策：主判据 A 在相同单样本 L2 下与 bootstrap 比较；F 只在 E 通过后计为收益；同时看自洽脚滑、GT 站定帧脚速与审查图。
4. **mode R 的条件是否在分布内**。"root 残差 = 0"位于条件分布中心，root 条件化训练使 p(局部 | root) 可学，腿会按慢 root 迈较少、较短的步。
   - 风险在于 root 锚定时步数收益有限，所以 F1 只在 P 模式下判。
   - 腿与 root 不同步导致的滑行由 E' 护栏与 C1 应急臂兜底。
5. **跳变**。train 中 12.8% 的窗口有跳变，按物理阈值过滤、不在 val 上调；评估在全体与干净两个子集上都报告。
6. **过拟合或欠训练**。1709 条序列、30k 个重叠窗口，20000 step × 64 约为 42 个窗口 epoch。
   - 用 teacher-forced 分 t 损失与 APD 的 train/val 比诊断；
   - 本批不挑 checkpoint，也不改步数；需要调整时另立下一批。
7. **样本量小**。H-val 247 条，起步者约几十人，Brier 与熵的误差条大。
   - 要求 3 seed 并报逐窗置信区间；
   - 最终结论以唯一一次 test 为准，但 test 不参与选择。
8. **少样本动作**（A012–A019 在 train 中各 1–14 条）的条件分布学不好。按动作组分别报告，不设判据。
9. **依赖与资源**。T1 需要第 1 批的 v2subj s0–2 EMA 终点与 subjval base，且第 1、2 批已释放 GPU；T3 需要原主线 checkpoint（已存在）。若第 2 批的手指去重改变了主线，底座与 OOF 库须按新主线重建，旧库作废。
10. **受试者字段只记录一名表演者**。"受试者不重叠"只针对这一字段，与 xsub 口径一致。
11. **手部旋转未建模**。手指相对手腕的残差占 4.6% 的能量；只修正这一项的 oracle 让 mpjpe 降 7.6%。留作后续，可加 hands 组。
12. **replacement 以外的 inpainting 近似**：不使用。

**立即停止的条件**：
- 结构自检失败：属实现错误，修复后才能继续；
- 训练 loss 非有限；
- 20000 step 时 teacher-forced 在 t=50 的归一化 MSE ≥ 1.0，即不如预测 0，属训练失败；
- A 未通过：Track B 以本形式不采纳；
- 应急臂跑完一轮仍未通过：不采纳。

## 9. 不做的事与旁支

**不做**：
- rot6d 表示与 FK 扩散；
- 从纯噪声生成整段未来；
- 主臂中的 L_inter、CFG；
- 读隐藏层作条件、在 val 上挑 checkpoint；
- 采纳之前动 test；
- 用 Track B 的数字替换 v2 的确定性主数字。

**旁支**（交给主会话决定，不属于本方案）：设计 A 的分析发现，v2 的 3 seed 输出平均在旧 val 上 mpjpe −3.7%（0.1617 → 0.1557）。这是免训练的确定性增益，但旧 val 与 train 共享受试者，可能偏乐观；应先在 subjval 上核实，再考虑是否单独立项。

## 10. 实现规格

各新文件的接口、CPU 自检（`scripts/check_ntu2p_resdiff.py` 共 10 节）与完成标准由实现工程师按本设计第 7 节的文件清单落实；实现结果、偏离与验证记录写入 `docs/ai/context/<时间戳>-ntu2p-trackb-implementation-result.md`。
