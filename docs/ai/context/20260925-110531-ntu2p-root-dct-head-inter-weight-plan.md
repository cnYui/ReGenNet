# NTU2P residual refiner：root 轨迹 DCT 头与 inter 权重重测——设计与计划

前置：
- 当前主线：const-EMA，即 s2_5 配置、恒定学习率、`--ema_decay 0.999`，报告 `ema/` 终点权重；val mpjpe 0.1751 ± 0.0014，test 0.1882 ± 0.0005。见 `docs/ai/context/20260924-190149-ntu2p-lr-cosine-ema-result.md`。
- 两项候选来自 Stage 3：root 误差随 horizon 线性增长、不饱和；inter 权重的旧消融是在鞍点时代做的。见 `docs/ai/context/20260902-205844-ntu2p-articulation-recovery-stage3-result.md`。

## 启动前分析（2026-09-25，不改代码，脚本在会话 scratchpad）

对象为 const-EMA seed 0 的 `ema/model000010000.pt`。

### 1. 归一化后各 loss 项的占比（train 全集）

| 项 | 相对 copy-last 比值 | 权重 | 占总 loss |
|---|---:|---:|---:|
| local_velocity | 1.005 | 1 | 42.4% |
| local | 0.373 | 1 | 15.8% |
| mse | 0.208 | 1 | 8.8% |
| velocity | 0.985 | 0.2 | 8.3% |
| dct_mid / dct_low | 0.185 / 0.175 | 1 / 1 | 7.8% / 7.4% |
| root | 0.172 | 0.1 | 0.7% |
| **inter** | **0.119** | **0.01** | **0.1%** |

- 当前 `inter_loss_weight=0.01` 在归一化后只占 0.1%，实际等价于没有 inter 损失。
- 旧消融里"0.05/0.1 中后期退化"的结论出自未归一化时代，那时 inter 原始量级约是 mse 的 6 倍，不能直接搬到现在。
- 按现在的量级，权重 1 / 3 / 10 分别对应约 4.8% / 13% / 33% 的占比，这才是有意义的测试区间。

### 2. root 残差在锚定 DCT 基上的能量分布（train）

锚定基为 φ_k(t) − φ_k(0)，k = 1..K。它在首帧恰为 0，且平滑。

| 残差 | K=1 | K=3 | K=5 | K=15 |
|---|---:|---:|---:|---:|
| GT − base（refiner 应学的 root 修正） | 78.1% | 88.7% | 90.3% | 91.9% |
| 模型 − base（当前实际输出的 root 修正） | 92.8% | 98.4% | 99.0% | 100% |
| GT − 模型（剩余 root 误差） | 46.5% | 66.8% | 71.3% | 77.2% |

- K=5（≤1 Hz，与摆动指标的 low 频带同口径）已能覆盖 GT root 残差可表达部分的约 98%（90.3 / 91.9）。
- 当前模型输出的 root 修正本来就很平滑，所以"输出平滑"不是瓶颈。

### 3. root 误差随 horizon（val 198，单位 m）

| 方法 | f10 | f20 | f30 | f50 |
|---|---:|---:|---:|---:|
| copy-last | 0.066 | 0.134 | 0.208 | 0.326 |
| 匀速外推（最后 5 帧速度） | 0.055 | 0.116 | 0.190 | 0.315 |
| base | 0.067 | 0.112 | 0.160 | 0.248 |
| 当前模型（const-EMA s0） | 0.062 | 0.095 | 0.121 | **0.181** |

- 0.5 s 观测估计的速度几乎不能预测 2.5 s 的位移：匀速外推在 f50 只比 copy-last 好 3%。模型已经比匀速外推好 43%。
- 模型 root 修正在 GT 残差方向上的回归系数为 0.49。这和一个约解释一半方差的条件均值估计一致，本身不构成缺陷。

### 对假设的修正

Stage 3 把 root 误差定为"可修"，依据只是线性增长。上面的证据表明：输出平滑性不是瓶颈，朴素运动学外推也很弱，剩余误差可能大部分是内在不确定性。因此 root 头只作为一个待检验假设，预期收益保守。

root 头能提供、而当前结构没有的是：
1. 显式的观测 root 运动学输入，包括本人 root 轨迹和双人相对 root 轨迹；当前模型只能从 165 维逐关节输入中隐式提取。
2. 整体平移的参数化：一个输出同时平移全部 55 个关节，不必靠逐关节 delta 协同产生，也不会在平移时扰动局部姿态。

如果它没有收益，结论就是：root 误差主要是 10 帧观测下的内在不确定性，应交给 Track B（多样本）处理，而不是继续改确定性结构。

## 设计

### A. root 轨迹 DCT 头（`--root_head_mode dct --root_dct_k 5`，默认 `none`）

对每个人 p（A/B 共享参数）：

- 输入：
  - (a) 本人观测 root 相对本人观测末帧的轨迹，10×3；
  - (b) 对方 root 相对本人 root 的观测轨迹，10×3；
  - (c) 本人经过跨人 decoder 后的 future token 时间均值，256 维。
  - (a)(b) 拼成 60 维，经 `Linear(60, 256)` 投影后与 (c) 相加 → LayerNorm → Linear → GELU → Linear(256, K·3)。最后一层零初始化。
- 输出：K×3 个系数 c，得到 `root_offset(t) = Σ_k c_k (φ_k(t) − φ_k(0))`，k = 1..K。它作为整体平移加到全部 55 个关节上：`pred = base + α·ramp·delta + root_offset`。
- 性质：
  - 首帧恰为 0，首帧连续性（first_step_error = 0）不变；
  - 零初始化时输出与无 root 头的模型完全相同；
  - 锚定基作为非持久 buffer，不进 state_dict；
  - `none` 时不创建任何新模块，训练逐位不变；
  - 旧 checkpoint 按默认 `none` 加载。
- 参数量 +85,775（现有 707 万）。
- 配对：root 头在 `torch.random.fork_rng(devices=[])` 内初始化，不消耗全局 CPU 随机数。因此同 seed 下，数据打乱顺序、dropout 与对照完全一致，step 1 的 loss 与对照逐位相同，差异只来自 root 头本身。inter 配置不加参数，天然与同 seed 对照配对。
- 不做"解耦"变体（让 delta 头不能动 root）：分析表明 delta 头的 root 修正已很平滑，解耦只是更强的约束。只有当 A 显示部分收益时才作为后续跟进。

### B. inter 权重重测（`--inter_loss_weight` ∈ {1, 3, 10}）

- 不改代码，覆盖公共参数中的 0.01（argparse 取最后一次出现的值，以 run 目录 `args.json` 为准）。
- inter 项的定义不变：跨人 4 对手腕距离的 MSE，按 copy-last 训练集误差归一化。

## 实验

公共条件：s2_5 配置 + `--ema_decay 0.999`，10000 step，seed 0/1/2，cuda:0，报告 EMA 终点。

**对照**：已有 const-EMA 的 3 个 run，不重训：`..._artic_ema_s2_5_dct1_root01_s{0,1,2}_10000/ema/`。

**Stage A（4 个配置 × 3 seed = 12 个 run，约 6.4 h，串行）**：按信息量排序

| 顺序 | 配置 | 额外参数 |
|---:|---|---|
| 1 | `rootdct` | `--root_head_mode dct --root_dct_k 5` |
| 2 | `inter10` | `--inter_loss_weight 10` |
| 3 | `inter3` | `--inter_loss_weight 3` |
| 4 | `inter1` | `--inter_loss_weight 1` |

**用 10000 而不是 5000 step 筛选**：这里偏离了步数预算文档"筛选用 5000"的约定。新数据显示，const-EMA 在 5000 step 时 3 seed 的 mpjpe 标准差为 0.0043，10000 时为 0.0014。本轮预期的改进在 1–3%（0.002–0.005），5000 step 分辨不出来。筛选直接用最终协议，胜出者的数字也就是最终数字。

**不并行**：RTX 3080 显存 10 GB，单 run 约占 5 GB，两个并行会逼近上限，一旦 OOM 会丢失整个 run（训练不支持中途续跑）。

**Stage B（条件触发，3 个 run，约 1.6 h）**：只有当 `rootdct` 与某个 inter 权重都被采纳时，才跑两者的组合。

## 预先登记的判断规则（只看 val 的 EMA 终点 @10000，3 seed）

对照 const-EMA：mpjpe 0.17512 ± 0.00140，xyz_mse 0.02279，xyz_mae 0.08373，gate 3/3。

- **采纳条件**（同时满足）：
  1. mpjpe 均值比对照低至少 0.00140（1 个对照 seed 标准差，约 0.8%）；
  2. xyz_mse 与 xyz_mae 均值都不高于对照；
  3. gate 3/3。
- **inter 多个权重都满足时**：取 mpjpe 均值最低者；与最低者相差不到 0.0005 的，优先较小的权重（对训练目标改动更小）。
- **Stage B**：组合满足采纳条件，且 mpjpe 均值不高于两个单因子中较好者，则采纳组合；否则采纳单因子中较好者。
- **机制核对**（报告，不参与判断）：
  - rootdct 看 root_translation_error、relative_root_distance_error、root 误差随 horizon 的斜率；
  - inter 看 key_joint_relation_error、contact_error、relative_root_distance_error。
- **不因关系指标单独采纳**：L2 三项是主判据，关系指标只作辅助证据。这与 AGENTS.md 的解释边界一致。
- 没有配置满足条件时，保持 const-EMA，记录负结果。其中 rootdct 的负结果按上面的"假设修正"解读。
- **test 纪律**：只对最终采纳的配置评估 test（3 个 EMA 终点），并画 test horizon 曲线与 const-EMA 对比；其余配置不看 test。

## 实现

1. `model/forecasting_ntu2p_residual_xyz.py`：
   - 新增 `root_head_mode` / `root_dct_k` 构造参数，写入 `config()`，加载时默认 `none` / 5；
   - 锚定 DCT 基作为非持久 buffer；
   - root 头只在 `dct` 时创建，且在 `__init__` 末尾、fork 出的 CPU 随机数流中初始化，保证已有层的初始化与数据顺序不变；
   - `forward` 中计算整体平移。`return_details` 返回的 delta 不含 root_offset，delta_reg 只约束逐关节残差，root 输出靠锚定低频基本身约束平滑。
2. `train/train_ntu2p_residual_refiner_xyz.py`：新增 `--root_head_mode`、`--root_dct_k`，传给模型构造。
3. 新增驱动 `scripts/run_ntu2p_root_inter.py`：
   - 复用 Stage 1/3 的训练与评估函数（`_train(prefix, extra_args)`、`_evaluate`），以及 LR/EMA 驱动的统计函数；
   - 每个 run 训完立即评估 raw 与 `ema/` 的 val；
   - 汇总写入 `results/forecasting/ntu120_label/ntu2p_root_inter/`，包括 `summary.md`、`decision.md/json`，其中列出 L2、摆动、root 与关系指标；
   - `--stage_b`（无参数）按 Stage A 的判断自动训练 root 与选中 inter 权重的组合，未触发时只记录原因；`--test_config <配置>` 只评估选中配置的 test。
4. 启动前测试：
   - 默认参数（`none`）1000 step 的原始权重与 const-EMA s0 的同名权重逐张量相同；
   - `dct` 模式零初始化时，前向输出与加载同一套非 root 头权重的无头模型完全相同；
   - `root_offset` 在 t=0 处恰为 0；
   - checkpoint 保存加载往返后输出一致；
   - 驱动 dry run 核对命令，inter 配置的 `args.json` 中 `inter_loss_weight` 为覆盖后的值。

## 运行与交付

- `setsid nohup` 后台启动，日志 `results/forecasting/ntu120_label/ntu2p_root_inter/driver.log`。驱动可断点续跑。
- 分批同步：
  - 第 1 批：本计划、实现、启动前测试记录，AGENTS.md 登记为进行中；
  - 第 2 批：Stage A（以及触发时的 Stage B）结果、test（仅采纳配置）、AGENTS.md 更新。
