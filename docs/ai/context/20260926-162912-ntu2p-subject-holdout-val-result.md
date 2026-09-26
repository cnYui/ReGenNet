# NTU2P 受试者留出选型 val：结果（下一步候选第 1 批）

计划：`docs/ai/context/20260926-105152-ntu2p-subject-holdout-val-plan.md`。

## 1. 新划分

`scripts/build_ntu2p_subject_holdout_manifest.py` 按预登记过程生成（50000 次随机子集搜索，seed 0，评分 0.0666），输出 `results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json`。

- **留出受试者（15 个）**：P004、P027、P031、P038、P046、P049、P055、P056、P074、P078、P081、P082、P089、P091、P097。
- **规模**：train 1709 条，val 247 条，test 1253 条（不变）。代码中已断言三者的受试者两两不重叠。
- **NTU60 占比**：train 0.824，val 0.547，test 0.555。val 覆盖了 26 类动作中的 25 类。

配套产物：
- xyz 缓存：`results/forecasting/ntu120_label/ntu2p_xyz_seq_cache_subjval/`；
- 在新 train 上重训的冻结单人 base：`save/forecasting/ntu120_label/ntu2p_independent_single_person_o10_p50_subjval_s0_5000/model000005000.pt`，供新协议下的 A0 与评估中的 base 指标使用，避免泄漏；
- 驱动新增参数 `--manifest_path/--cache_dir/--baseline_checkpoint/--run_prefix`。新协议的 run 目录为 `save/forecasting/ntu120_label/ntu2p_v2subj_*`，汇总在 `results/forecasting/ntu120_label/ntu2p_v2subj_screen/summary_10000.md`。
- 新协议下关闭"事后 root 混合"参考线：它用的检索库由原 train 建成，含留出受试者，会造成泄漏。

## 2. 协议有效性检验

在新 train 上训练 A0 与 v2 主线（`A6-F-A5f0.05-A4-GH0.5`），各 3 seed × 10000 step，取 EMA 终点。同一批模型分别在新 val 和 test 上评估。test 评估只用于检验协议，不参与任何选型；两个配置都早已采纳或作为对照。

| 口径 | Δmpjpe（v2 vs A0，逐 seed 配对） | Δxyz_mse | Δxyz_mae |
|---|---|---:|---:|
| 旧 val（按序列随机，受试者与 train 重叠；原 train 训练的模型） | −7.8% | −18.1% | −6.7% |
| **新 val**（受试者留出） | **−2.2 / −3.0 / −3.8%，均值 −3.0%** | −7.0% | −2.2% |
| test（新 train 训练的同一批模型） | −4.7 / −4.1 / −4.5%，均值 −4.4% | −7.6% | −3.5% |
| test（原 train 训练的模型，参考） | −3.85% | −5.5% | −2.9% |

绝对数字：
- 新 val：A0 0.17700 ± 0.00105，v2 0.17170 ± 0.00040；
- test（新 train）：A0 0.19031，v2 0.18187。v2 只用 1709 条序列训练，test 仍达到 0.1819，与原主线 0.1810 相差不到 0.5%。

**结论：通过。**
- 新 val 对 test 收益的估计误差，从旧 val 的约 4 个百分点（高估约 2 倍）降到约 1.4 个百分点，而且偏向保守（略微低估）；xyz_mse 的估计几乎一致。
- 新 val 的 seed 标准差：v2 为 0.0004，A0 为 0.0011，与旧 val 相当。
- 今后的选型一律用新协议：在新 train 上训练，用新 val 判断；采纳后再在原协议上训练 3 seed，评估一次 test，得到与主数字可比的数字。主数字仍为原协议下的 v2（test 0.1810）。

## 3. 事故记录

第 1/2 批串联脚本用 `pgrep -f train_ntu2p_independent_single_person` 判断 base 是否训完。这个模式也匹配到了会话中其它等待命令的命令行，形成互相等待，驱动 #2 从 12:33 一直卡到 16:02，GPU 空闲了约 3.5 小时。修复方法：以后等待进程一律用 PID（`kill -0 <pid>`）或完成标记文件，不再用 `pgrep -f` 模式匹配。
