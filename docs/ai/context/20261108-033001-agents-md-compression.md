# AGENTS.md 每日压缩归档（20261108）

- 时间：2026-11-08 03:30:01 +09:00
- 压缩前：59 行、25683 字节；压缩后：59 行、25597 字节（`wc -lc AGENTS.md`）。
- 本次精简 3 条（均为原位精简，没有整条移除，所引用的 `docs/ai/context/` 路径全部保留在入口中）。

## 被精简的条目

### 1. 「当前推进中的 NTU 双人 ReGenNet 风格 diffusion」条（`## 当前研究入口`）

命中判据：被后续条目明确取代的中间状态。"当前推进中"的说法已过时：该 diffusion 已训练完成，结果记录在其后各条（均未超过 `copy-last`，主线已迁到 residual refiner / v2 / NTU2PForecast）。协议参数与两份文档路径全部保留。

原文：

```
- 当前推进中的 NTU 双人 ReGenNet 风格 diffusion：`window_len=60, obs_len=10, pred_len=50`，双人 raw axis-angle 转 canonical rot6d，条件为 obs10 + 26 类动作标签，目标为 future50，训练目标为 `L_dm + L_inter`。协议和实现见 `docs/ai/context/20260723-214702-ntu-two-person-forecasting-diffusion-regennet-design.md` 与 `docs/ai/context/20260723-215149-ntu-two-person-forecasting-diffusion-regennet-implementation-plan.md`。
```

精简后：

```
- NTU 双人 ReGenNet 风格 diffusion 协议（训练与结果见下方各条）：`window_len=60, obs_len=10, pred_len=50`，双人 raw axis-angle 转 canonical rot6d，条件为 obs10 + 26 类动作标签，目标为 future50，训练目标为 `L_dm + L_inter`。协议和实现见 `docs/ai/context/20260723-214702-ntu-two-person-forecasting-diffusion-regennet-design.md` 与 `docs/ai/context/20260723-215149-ntu-two-person-forecasting-diffusion-regennet-implementation-plan.md`。
```

移除的片段：`当前推进中的`（改为"协议（训练与结果见下方各条）"，该指向本身即下方各条已有的内容，不新增事实）。

### 2. 「摆动恢复 Stage 0 已完成」条（`## 当前研究入口`）

命中判据：引用文档中已完整保存的过程细节（实现等价性验证）；gate 定义、新增训练选项、驱动脚本、全部文档与汇总表路径均保留。

原文：

```
- 摆动恢复 Stage 0 已完成：新增摆动指标 `compute_ntu_articulation_metrics` 与评估 gate（`passes_full_gate` = L2 三项低于 copy-last + 能量比值 ≥ 0.10 + frozen ≤ 0.20 + mpjpe 相对 base 回退 ≤ 5%）、训练新选项（`--loss_scale_normalize`、`--local_velocity_loss_weight`、`--articulation_energy_loss_weight`、`--ramp_mode saturate`、`--future_pos_mode sinusoidal`，默认全部关闭）、随机选例导出与 Stage 1 串行驱动 `scripts/run_ntu2p_articulation_stage1.py`。默认参数下与 inter001 checkpoint 逐位等价（差异 ≤ 1.2e-7）。结果见 `docs/ai/context/20260902-150402-ntu2p-articulation-recovery-stage0-result.md`；Stage 1 汇总表在 `results/forecasting/ntu120_label/ntu2p_articulation_stage1/summary.md`。
```

精简后：

```
- 摆动恢复 Stage 0 已完成：新增摆动指标 `compute_ntu_articulation_metrics` 与评估 gate（`passes_full_gate` = L2 三项低于 copy-last + 能量比值 ≥ 0.10 + frozen ≤ 0.20 + mpjpe 相对 base 回退 ≤ 5%）、训练新选项（`--loss_scale_normalize`、`--local_velocity_loss_weight`、`--articulation_energy_loss_weight`、`--ramp_mode saturate`、`--future_pos_mode sinusoidal`，默认全部关闭）、随机选例导出与 Stage 1 串行驱动 `scripts/run_ntu2p_articulation_stage1.py`。结果见 `docs/ai/context/20260902-150402-ntu2p-articulation-recovery-stage0-result.md`；Stage 1 汇总表在 `results/forecasting/ntu120_label/ntu2p_articulation_stage1/summary.md`。
```

移除的片段：`默认参数下与 inter001 checkpoint 逐位等价（差异 ≤ 1.2e-7）。`（"新选项默认全部关闭"已在同条保留，等价性验证细节见 Stage 0 结果文档。）

### 3. 「摆动恢复 Stage 2 已解决…」条（`## 当前研究入口`）

命中判据：已被后续条目记录完成的"下一步"计划描述。Stage 3 的 seed 复现与 20000 step 已在「摆动恢复 Stage 3 已完成」条中记录；驱动脚本与汇总路径仍保留。

原文：

```
- **摆动恢复 Stage 2 已解决"整体平移、四肢不动"问题**：底座（`--loss_scale_normalize --local_velocity_loss_weight 1.0 --ramp_mode saturate`）+ 分频带幅度损失 `--dct_low_amplitude_loss_weight 1.0 --dct_mid_amplitude_loss_weight 1.0` + `--root_loss_weight 0.1`（run `s2_5_dct1_root01`）：mpjpe 0.1901（相对 base −16.1%、copy-last −31.6%），1.2–2 Hz 频带能量从 GT 的 0.15% → 57.8%，frozen 0.358 → 0.096（GT 0.098），高频带 < 0.01 无抖动，短/中/长 horizon 与 root/local 同时改善。机制修正：control 的残差在时间上严格线性（零初始化 future_pos 导致的鞍点，非条件均值），幅度型损失在鞍点处梯度不消失，推出后 L2 随之下降；`temporal_std` 只放大慢速漂移，`dct_mid_amplitude` 是关键项。5000 step 未收敛。结果见 `docs/ai/context/20260902-192249-ntu2p-articulation-recovery-stage2-result.md`；轨迹对比图 `results/forecasting/ntu120_label/ntu2p_articulation_stage2/joint_traj_random_cases_control_vs_s2_5.png`；Stage 3（seed 1/2 复现 + 20000 step）驱动 `scripts/run_ntu2p_articulation_stage3.py`，汇总 `results/forecasting/ntu120_label/ntu2p_articulation_stage3/summary.md`。
```

精简后：

```
- **摆动恢复 Stage 2 已解决"整体平移、四肢不动"问题**：（中间内容不变）……结果见 `docs/ai/context/20260902-192249-ntu2p-articulation-recovery-stage2-result.md`；轨迹对比图 `results/forecasting/ntu120_label/ntu2p_articulation_stage2/joint_traj_random_cases_control_vs_s2_5.png`；Stage 3 驱动 `scripts/run_ntu2p_articulation_stage3.py`，汇总 `results/forecasting/ntu120_label/ntu2p_articulation_stage3/summary.md`。
```

（"中间内容不变"表示从"底座"到"5000 step 未收敛。"之间的原文一字未改，见上方原文。）

移除的片段：`（seed 1/2 复现 + 20000 step）`。

## 事实并入

无。没有事实并入其它条目。

## 本次刻意保留的内容

- `## 稳定约束`、`## 解释边界`、标题与引用块：按范围要求原样保留。
- 「NTU2P 首次 test 评估」条：虽标注"主数字已被下方 EMA 取代"，但含 test 集规模（1253）、"摆动恢复前 control 在 test 上不过 gate（dct_mid 0.0017）"的负结果与"训练逐位确定，方差只来自 seed"，保留。
- 「NTU2P 权重 EMA 结论」「rootdct」「v2」条：主数字已被取代，但仍是后续各条的对照基线，并含余弦尾部衰减未采纳、inter 权重未采纳且"不能据此声称交互建模收益"、手指占 loss 81% 的根因、滑行帧比护栏未过的偏离等，同时承载 checkpoint 路径与文档入口保护，保留。
- 早期 diffusion / attention / 独立单人 baseline / `L_dm-only` gate / 深度排查等条：属历史阶段，但含未超过 `copy-last` 的负结果与"不能声称 attention 带来性能提升"的边界，保留。
- 「当前新的主要比较 gate」条：虽有后续 gate 取代其使用场景，但没有条目明确宣布废止，有疑问，保留。
- 「当前最佳 residual refiner 的架构图…」条：措辞已过时，但含可复用 xyz 导出脚本入口，且整条移除会使其引用文档失去入口保护，保留。
- `inter_loss_weight` 消融、回归到均值诊断、Stage 1 / Stage 3、步数预算、FD2 主线与受试者留出 val 协议、Track B 负结果、接触图/双腿/DCT 扩散、JiT-RL 本轮结论及其审查入口、NTU2PForecast 迁移指针：均为现行协议、负结果或坑，保留（JiT-RL 条内的"两处实现修订"属于易复发的坑，保留）。
- `## 文档入口` 全部条目：每条均引用仍需入口保护的文档（含维护脚本设计与首次运行记录），保留。
