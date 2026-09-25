# NTU2P v2 候选架构：实现、代码审查与筛选协议调整

设计与计划：`docs/ai/context/20260925-121154-ntu2p-v2-architecture-exploration-design-and-plan.md`。数据管线：`docs/ai/context/20260925-120854-ntu2p-xyz-seq-cache-canonical-augment-result.md`。

全部代码由四个并行实现子 agent 在 CPU 上完成（GPU 被另一会话的 root/inter Stage A 独占）。之后用一个对抗式审查 workflow 复核：4 个维度审查、每条发现 2 个独立怀疑者、确认后统一修复、最终把关。只新增文件，现有训练与评估入口的行为不变。

## 新增文件

| 文件 | 内容 |
|---|---|
| `model/forecasting_ntu2p_v2.py` | `NTU2PCanonRefiner`（继承旧 refiner）。开关：`canonical`、`disp_channel`、`role_embed`、`mirror_embed`、`retrieval_k`、`kin_proj`。另有 `forward_details`（统一返回 dict）与按 model_type 分派的 `load_ntu2p_model_checkpoint` |
| `model/forecasting_ntu2p_intermixer.py` | A6 端到端规范系双人 DCT-Mixer，默认 2.71M 参数；刚体手（每手每帧只预测一个旋转）；B 使用"朝向对方"坐标系 |
| `model/ntu2p_retrieval_anchor.py` | A3 检索锚点：距离 softmax + 零初始化打分；root / 局部分通道 β（root 0.5、局部 0，线性参数化）；exemplar tokens |
| `data_loaders/forecasting/ntu2p_retrieval_bank.py`、`scripts/build_ntu2p_retrieval_bank.py`、`scripts/check_ntu2p_retrieval_bank.py` | 规范系检索库：30,908 个 train 窗口，每条序列最多 1 个窗口去重；同动作检索，排除同受试者（训练时再排除同序列）；回退层级 `tier` |
| `utils/ntu2p_kinematic_projection.py` | A4 `SkeletonProjector`：骨长取观测末帧；四肢用两骨 IK；多子节点关节用 Procrustes（不用 `torch.svd`：GPU 上逐矩阵计算，且奇异值相等时梯度为 nan）；刚体手。A5 `foot_contact_mask`、`foot_skate_loss`（速度先平滑） |
| `train/train_ntu2p_v2.py` | 缓存管线上的统一训练入口，`--arch refiner/canon_refiner/intermixer`，直接 import 旧的 loss、EMA、学习率函数；训练日志逐步记录额外项的占比 |
| `eval/eval_ntu2p_v2.py` | 旧评估同结构的 JSON；base 一律用冻结独立单人 base；加自然度块、`--export_arrays`、`--posthoc_root_blend` |
| `scripts/run_ntu2p_v2_screen.py` | Stage 1–3 驱动：`--workers` 并行、断点续跑、预登记判断、test 前置拦截 |

## 验证（CPU）

- **等价性**：全部开关关闭时，把主线 EMA 权重加载进 v2 模型，输出与旧类 `torch.equal`。在已训练权重上开启零初始化的 disp/role/mirror，输出仍与旧类 `torch.equal`。
- **新旧评估一致**：新评估在全部 198 条 val 上与旧 GPU 评估 JSON 相比，mpjpe 差 1.3e-6，gate 相同。
- **首帧误差为 0**：canonical、A3、A6 都是 0；A4 为 0 与 4e-8。A4 投影后骨长误差 1e-15（float64）/ 2e-6（float32）。
- **InterMixer**：零初始化时输出逐位等于 copy-last；刚体变换等变误差 3.4e-6。CPU 上训练 1000 step，val 全量 mpjpe 为 0.1988（copy-last 0.2779，冻结 base 0.2265）。
- **骨架投影作用于当前最佳模型的 val 输出**：mpjpe +0.04%，xyz_mse +0.46%，骨长误差 5.2% → 0，dct_mid 0.598 → 0.581。
- **冒烟**：A0/A1/A2/A3/A6 以及 A1+A4、A1+A5、A1+A7 都走真实训练 CLI 各训 5 step，保存后用评估的加载函数加载并前向，结果全部有限、首帧误差为 0。驱动在临时目录端到端跑通，断点续跑只重训失败的 run。

## 检索 kNN-only 分析：val 受试者重叠的影响（CPU，val 198）

| 方法 | 排除同受试者 | 不排除 |
|---|---:|---:|
| 只用检索，K=8 / 16 / 40（距离软平均） | 0.1830 / 0.1842 / 0.1881 | 0.1753 / 0.1784 / 0.1827 |
| 当前最佳 + 事后只混合 root（0.5，带 ramp） | **0.1709（−2.4%）** | 0.1690 |
| 当前最佳 + 整体 0.5 混合 | 0.1680（−4.1%），但 dct_low 0.399，不过 gate | 0.1646 |

- 受试者重叠让单用检索的结果虚高约 3–4%。所有检索评估都用"排除同受试者"口径。
- 事后只混合 root 是免训练参考线：A3 必须超过它才有意义。驱动在 A0 的 EMA 评估中同时报告这条线。

## 代码审查确认并修复的问题

1. **A5 权重标定过期**：原默认 1.0 实际占总 loss 11–13%。按现行训练路径重标（seed 0/1/2 各 300 步，每步占比中位数），默认改为 **0.35**，初始占比约 5%。I-D 建议的 0.06 是按"收敛后占 5%"标定的，与设计的"初始约 5%"口径不同，不采用。训练日志的 `foot_share` 可以核对实际占比。
2. **自然度否决项的参照**：原来叠加配置与上一级比较，每层都可以再差 10% 而不被否决。已改为固定与 A0 比较。
3. **`--test_config` 可能在 5000 step 的 checkpoint 上用掉唯一一次 test**：已加 `check_test_allowed()`。只允许 10000 step、非 A0、在 `summary_10000.json` 中判为候选采纳、且 3 seed 的 EMA 终点齐全的配置评估 test。
4. **5000 step 的噪声约为采纳阈值的 3 倍**：见下面的协议调整。另加"不确定"档：其它判据都通过、只有 L2 落在噪声内时标"不确定"，不直接判无效。

被否决的发现（11 条）包括 A2 解冻 base 后 dropout 生效、A3+A7 镜像样本查未镜像库、Ctrl-C 停不下排队任务等。怀疑者判断它们不影响本轮结论，或已有其它处理。

## 筛选协议调整（相对设计 4.1 / 4.3）

- **Stage 1/2 改用 10000 step 筛选**（原计划 5000）。
  - 依据：主线 const-EMA 的 val mpjpe seed 标准差在 5000 step 为 0.0043（2.3%），10000 step 为 0.0014（0.8%）。同 seed 配对并不降噪：rootdct 对 const-EMA 的配对 Δ 标准差在 5000 step 为 2.35%。所以 5000 step 无法分辨预登记阈值 0.8% 量级的差异。
  - 先例：root/inter Stage A 也用 10000 step 筛选。
  - 成本：新缓存管线 batch 8 约 10 step/s，比旧管线快 3.6 倍，10000 step 一个 run 约 20 分钟，与原计划中 5000 step 的旧管线相当。
  - Stage 1 胜出的单一配置可以直接复用这组 3 seed × 10000 step 结果作为最终数字，Stage 3 只需训练新的组合。
- **骨长否决项加 10% 容差**：与脚滑否决项一致，避免被 3 seed 的噪声误否。
- **并行**：3 个 worker。新管线不做在线 FK，估计每个进程显存 ≤ 1.5 GB。

## 启动方式

另一会话的 GPU 训练结束前不使用 GPU。本会话的 CPU 任务降为 nice 19，并限制在 CPU 5–7（应对方请求）。会话 scratchpad 中的 `gpu_watch/watch_and_launch.sh` 在两个条件都满足后才启动：另一会话的驱动进程已退出，且 GPU 连续 5 分钟没有其它计算进程。启动前先在 GPU 上核对主线 checkpoint：新评估与旧评估 JSON 的 L2 指标差必须 < 1e-5，通过后才运行：

```bash
PYTHONPATH=. /home/rpartx3080/.local/micromamba/envs/regennet/bin/python scripts/run_ntu2p_v2_screen.py --stage 1 --steps 10000 --workers 3
```

汇总输出到 `results/forecasting/ntu120_label/ntu2p_v2_screen/summary_10000.{md,json}`，run 目录为 `save/forecasting/ntu120_label/ntu2p_v2_{A0,A1,A2,A3,A6}_s{0,1,2}_10000/`。
