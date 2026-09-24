# NTU2P residual refiner 最终数字：s2_5 × 3 seed × 10000 step + test 评估 计划

前置：`docs/ai/context/20260902-205844-ntu2p-articulation-recovery-stage3-result.md`（s2_5 配置 3 seed 复现、20000 step 长训）、`docs/ai/context/20260902-212338-ntu2p-training-step-budget-design.md`（最终数字 = 10000 step × 3 seed、取终点）。

## 背景与目标

- 摆动恢复系列的所有数字都是 val（198 条）；residual refiner 从未在 test（xsub.test 1253 条）上评估过。
- AGENTS.md 标注的"当前最佳 checkpoint"是单 seed 20000 step，与后来定下的预算约定（10000 step × 3 seed、取终点、不在 val 上挑）不一致。
- 本轮目标：按预算约定产出可汇报的最终数字（val 与 test，3 seed 终点均值 ± 标准差），并附 test 上的误差-horizon 曲线。不改模型、不改损失。

## 协议（启动前冻结，test 结果不回流任何选择）

| 项 | 取值 |
|---|---|
| 配置 | `s2_5_dct1_root01`，与 Stage 3 完全相同：`--loss_scale_normalize --local_velocity_loss_weight 1.0 --ramp_mode saturate --ramp_saturate_frames 5 --dct_low_amplitude_loss_weight 1.0 --dct_mid_amplitude_loss_weight 1.0 --root_loss_weight 0.1`，公共条件 `--inter_loss_weight 0.01`、manifest seed0、冻结 base `ntu2p_independent_single_person_o10_p50_cuda_retrain_s0_5000/model000005000.pt`、`--device cuda:0` |
| seed | 0 / 1 / 2 |
| 步数 | `--num_steps 10000`，`--save_interval 1000` |
| checkpoint 选择 | 每个 seed 取终点 `model000010000.pt`，不在 val 上挑最好 |
| val | 每个 1000-step checkpoint 都评估，用于收敛曲线与 gate 稳定性；终点值进入最终表 |
| test | 只评估三个终点 checkpoint；gate 阈值与 val 相同（`dct_low ≥ 0.40`、`dct_mid ≥ 0.10`、mpjpe 相对 base 回退 ≤ 5%、L2 三项低于 copy-last） |
| 统计 | 3 seed 均值 ± 样本标准差（ddof=1，与 Stage 3 口径一致） |

参考行（同样在 test 上评估，只作对照，不参与任何选择）：

- base（冻结独立单人）与 copy-last：每个评估 JSON 自带，三个 seed 共享同一 base，数值一致；
- 摆动恢复前的 control：`ntu2p_residual_refiner_xyz_inter001_s0_5000/model000005000.pt`；
- Stage 3 长训终点：`ntu2p_residual_refiner_xyz_artic_s3_s2_5_dct1_root01_s0_20000/model000020000.pt`（AGENTS.md 当前标注的最佳，补 test 以便对照新口径）。

报告指标：`xyz_mse / xyz_mae / mpjpe`、相对 base 与 copy-last 的变化、`dct_low / dct_mid / dct_high_energy_ratio`、frozen、`passes_full_gate`；另在 test 上给出逐帧 `mpjpe / root / local / A-B 相对 root` 曲线（3 seed 均值 ± 标准差，对照 base 与 copy-last），关键帧 10 / 20 / 30 / 50。

附带诊断（不额外花算力）：训练过程中没有依赖 `num_steps` 的调度（学习率恒定），因此新 seed 0 run 在 step 10000 与 Stage 3 `s0_20000` 在 step 10000 的差值，就是同 seed、同配置下 CUDA 非确定性带来的噪声量级。

## 实现

1. `scripts/run_ntu2p_articulation_stage3.py`：`_save_dir` / `_train` 增加 `prefix` 参数，默认 `"s3"`，现有路径不变；新驱动复用它，避免复制训练命令拼装逻辑。
2. 新增 `scripts/run_ntu2p_final_10k.py`：串行训练 3 seed → 每个 checkpoint 的 val 评估（复用 Stage 1 的 `_evaluate`）→ 终点 test 评估 → 参考 checkpoint 的 test 评估 → 汇总。可断点续跑：已有终点权重则跳过训练，已有评估 JSON 则跳过评估。输出目录 `results/forecasting/ntu120_label/ntu2p_final_10k/`：
   - `summary.md/json`：逐 checkpoint val 表（与 Stage 1–3 的 summary 同格式）；
   - `final.md/json`：终点 val / test 逐 seed 行、均值 ± 标准差、base / copy-last / 参考行、GT frozen。
   - run 目录：`save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_final_s2_5_dct1_root01_s{0,1,2}_10000/`，test JSON 与 val JSON 同目录，命名 `eval_test_000010000.json`。
3. 新增 `scripts/ntu2p_error_vs_horizon.py`：把 Stage 3 scratchpad 里的 horizon 脚本正式化，参数化 `--split` 和 `--checkpoint name=path`，按 run 名前缀把多 seed 聚合成均值 ± 标准差，输出 json + png。
4. 本工作树没有数据和权重：把 `dataset`、`save`、`results` 软链接到主仓库对应目录（均不入库），产物与历史实验落在同一处。

## 运行

- `setsid nohup` 后台启动驱动（Stage 3 曾因 Claude Code 进程退出被中断），日志写 `results/forecasting/ntu120_label/ntu2p_final_10k/driver.log`。
- 预计耗时：训练 3 × 约 28 min，加 30 次 val 评估、5 次 test 评估，约 1.5–1.7 h。训练结束后再跑 horizon 脚本。
- 启动前先 `--dry_run` 核对命令，确认与 Stage 3 的 `args.json` 除 `seed / num_steps / save_dir` 外一致。

## 验收

- 三个终点权重存在；每个 seed 有 10 个 val JSON 和 1 个 test JSON；两个参考 checkpoint 各有 1 个 test JSON。
- summary 中的均值 ± 标准差能由各 JSON 复算。
- 预期（不作为 gate，仅用于发现异常）：val 终点 mpjpe 落在 Stage 3 平台带（0.1823 ± 0.0023）附近；test 是不同受试者，数值不强求接近 val，重点看相对 base / copy-last 的排序是否保持，以及 test 上摆动 gate 是否仍通过。

## 交付

- 结果文档 `docs/ai/context/<时间戳>-ntu2p-final-10k-3seed-test-result.md`；
- 更新 AGENTS.md 当前研究入口：最终数字改为 3 seed × 10000 终点的 val / test 均值 ± 标准差，"当前最佳 checkpoint"表述改为这一组终点；
- feature 分支提交后运行 `python3 scripts/ship_pr.py`。
