# NTU2P Track B（ResDiff-OOF+）实现与 CPU 验证结果（T0）

设计：`docs/ai/context/20260926-121543-ntu2p-trackb-residual-generative-design-and-plan.md`。本文记录 T0 批（仅 CPU）的实现、偏离与验证；未使用 GPU，未训练任何模型，未看 test。

## 1. 文件清单（全部新建；现有文件一律未改）

| 文件 | 内容 |
|---|---|
| `utils/ntu2p_residual_codec.py` | 场景规范系 → 个人朝向系、斜坡 DCT（K=12）最小二乘编解码、随动规则、跳变判定、条件特征 `condition_features` |
| `data_loaders/forecasting/ntu2p_residual_bank.py` | 残差库读写、统计量（干净窗口 RMS，5% 中位数下限）、按序列均匀的干净窗口采样器、bootstrap 分层池 |
| `model/forecasting_ntu2p_resdiff.py` | `NTU2PResDiffDenoiser`（47 token，4.95M 参数，零初始化输出）、训练/采样扩散、`sample_x0`、`one_step_mean`、checkpoint 读写 |
| `utils/ntu2p_probabilistic_metrics.py` | 人物分组、ES、minADE/FDE、APD/分配比、SSR、三类先迈脚 Brier、步数 CRPS/W1、手臂幅度、穿插、包络插值、逐窗 bootstrap CI、审查选例 |
| `scripts/build_ntu2p_crossfit_folds.py` | 受试者折 manifest 与切片缓存 |
| `scripts/build_ntu2p_oof_residual_bank.py` | OOF 库与 in-sample 库（含 OOF 断言） |
| `train/train_ntu2p_resdiff.py` | 生成器训练；应急臂 C1 foot / C2 inter 开关默认关闭 |
| `eval/eval_ntu2p_resdiff.py` | 预登记全部指标、确定性参照、bootstrap 网格、NFE 扫描、teacher-forced 诊断、审查数组导出 |
| `sample/plot_ntu2p_gait_fan.py` | 起步者 K 个样本的左右踝分离叠图 |
| `scripts/run_ntu2p_trackb.py` | 分阶段驱动、预登记判定（5.4）、原协议与 test 拦截、GPU 占用检查 |
| `scripts/check_ntu2p_resdiff.py` | CPU 自检（10 节） |

已在 CPU 上生成两个协议的交叉拟合折（T0 的一部分，确定性、可重跑）：`results/forecasting/ntu120_label/ntu2p_trackb/{subjval,original}/folds/`（`folds.json`、`manifest_fold{0,1,2}.json`、`cache_fold{0,1,2}/{train,val}_xyz_seq.pt`）。subjval 三折 val 569/572/568 条序列、10271/10132/9811 个窗口（合计 30214），P008 在第 0 折；原协议 587/586/585、10612/9995/10301（合计 30908）。

## 2. 关键接口

- 编解码：`ResidualCodec().target_coeffs(obs, P, gt, frames)`、`.draft_coeffs(obs, P, frames)`、`.apply(obs, P, coeffs_m, frames, projector)`；`codec_frames(obs)`、`repeat_frames(frames, K)`、`condition_features(codec, obs, P)`（库构建与评估共用）。
- 残差库：`NTU2PResidualBank.load/save`、`.batch(ids)`、`.make_sampler(B, seed).sample()`、`.bootstrap_pool(30).draw(action, walk, uniform) -> (ids[B,K], levels[B])`。
- 模型：`NTU2PResDiffDenoiser(stats)(x_t, t, y)`，`y` 键为 draft/obs_feats/rel_geom/action/root_known/root_value；`make_condition(model, feats, action, "F"|"R")`、`build_sampling_diffusion(N)`（`space_timesteps(1000,[N])`，含 0 与 999）、`sample_x0(model, sampler, y, noise, tau)`、`one_step_mean(model, y, noise[B,K',...], tau)`、`to_meters`。
- 评估输出 JSON：`meta / subsets / references{v2,copy_last,independent_base,a0,v2_ensemble,v2_ensemble_samples} / generator[mode][τ] / bootstrap[mode][s] / nfe_scan / diagnostics / review`；每个 generator/bootstrap 块含 single_l2、mean_of_k_l2、one_step_mean_l2、min_ade（@1/5/10/20）、min_fde@10、person_min_ade_body22@10、es_joint(_clean)、es_legs_walk、es_arms_act、apd、allocation_ratio、ssr_clean、lead_foot、steps、arm_amplitude、interpenetration_ratio、articulation_single、naturalness_single/per_slot、first_step_error_max、bone_rel_err_body_mean/max、pelvis_equal_v2、sample_digest、ci（bootstrap 另有 strata_levels）。
- 审查导出（`--export_review DIR`）：`review_arrays.pt`（methods = v2、F_k0..2、R_k0..2）、`review_arrays_{F|R}{τ}.pt`（methods = v2、mean、sample0..sample{K−1}），均可被 `sample/render_ntu2p_review_sheet.py --arrays` 读取；`review_samples.pt`（审查窗口的全部 K 个样本）给叠图脚本。

## 3. 验证（CPU，2 线程，nice 10）

`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 PYTHONPATH=. nice -n 10 python scripts/check_ntu2p_resdiff.py --scratch_dir <scratchpad>/trackb/check`：**76/76 通过**，墙钟 88 s。要点：

- codec：基首帧严格 0；编解码往返 6e-8；随动逐位相等；个人系往返 4.8e-7（残差量级 |v|≤1.5 m）；整场景 yaw + 平移后 target/draft 系数变化 3e-6。
- apply_equivalence（原主线 s0、旧 val 前 16 窗）：零系数与 v2 最大差 2.6e-6；oracle mpjpe 0.0370（v2 0.1501）；首帧误差 0；root 系数为 0 时 pelvis torch.equal；3σ 随机系数骨长相对误差均值 9.8e-8（最大 1.8e-6）、指尖到手腕距离差 1.7e-7；零初始化生成器经完整采样-解码-投影后 mode F/R 与 v2 差 2.6e-6，mode R pelvis torch.equal。
- denoiser / sampling：零初始化输出严格 0、root_known 替换逐位；4,953,666 参数；梯度有限非零；timestep_map = [0,111,…,999]；DDIM 同噪声逐位复现、不同噪声样本不同；零初始化采样严格 0；mode R root 通道严格 0。
- metrics（合成与真实 GT）：相同样本 ES = mpjpe；Brier 正确 0 / 错误 2；CRPS、W1、包络插值；校准高斯集合 SSR 0.998；copy-last 手臂幅度 0；GT 自身 Brier 0（286 人）；审查选例 18 例（起步 6、已在走 2、击打 4、其余手臂 2、随机 4）。
- folds / bank_smoke / train_smoke：折大小与断言如上；冒烟库 6 条序列 163 窗 = Σ(len−59)；冒充折模型时 OOF 断言生效；CPU 训练 20 步 loss 有限，EMA 加载后与内存模型 torch.equal；应急臂 foot+inter 各 2 步可运行并记录 *_share。
- eval_smoke（subjval val 前 8 窗、K=4、NFE2，底座 v2subj s0）：两次运行 JSON 完全相同；第 8 节全部键齐全；样本 APD 1.7 cm；首帧 0、骨长均值 6.9e-8；mode R pelvis 全部 τ 逐位相等；bootstrap s=0 与 v2 单样本 mpjpe 一致；导出数组可被渲染脚本读取并渲染；叠图脚本运行。
- driver：两个协议 `--stage all --dry_run` 打印完整计划；无 adopt 结论时 `--test` 返回 2；模拟 GPU 占用时返回 3；summary 在冒烟数据上完整走过 A–H/D/E'/F' 判据，并能触发训练失败停止条件。
- ruff（`--select E9,F63,F7,F82 --target-version py37`，另加全部 F 类）对 11 个新文件无问题。

## 4. 与规格的偏离与补充（均为实现口径，未改动预登记阈值）

1. 结构自检的骨长用**均值** `bone_rel_err_body_mean ≤ 1e-5`（与 naturalness 的 `bone_rel_err_body` 同义）；最大值受 float32 坐标舍入影响（短骨约 1e-6–1e-5），只报告。
2. 在规格的结论集合外增加三个**中途停止状态**：`structural_failure`、`training_failure`（终点 teacher-forced t=50 归一化 MSE ≥ 1.0，设计第 8 节的立即停止条件）、`pending_nfe_rerun`（NFE 规则触发但尚无 NFE20 评估）。
3. GPU 占用检查识别 `train_ntu2p_v2.py`、`eval_ntu2p_v2.py`、`run_ntu2p_v2_screen.py` 三类 python 进程（规格只要求第一类；第 1/2 批驱动在训练间隙也会跑 GPU 评估）。
4. 原协议的 GPU 阶段（折模型、库、训练、评估）也要求 subjval 结论为 adopt_full，或 adopt_probabilistic_only 加 `--confirm_probabilistic_only`（与 5.6"原协议复训须经用户确认"一致）；`--dry_run` 时只打印警告。test 在驱动与评估脚本两层都拒绝覆盖已有结果。
5. F2 的"单样本先迈脚正确率"用与 Brier 同口径的自身首事件分类（`lead_foot.acc_single`）；naturalness 口径的 `gait_lead_foot_acc_starting` 同时在 `naturalness_single` 中报告。
6. B 把 mean-of-20 mpjpe ≤ 1.04×v2 并入判定（健全性检查失败即视为 B 未通过）。
7. 训练诊断 4.7：seed 0 的 5000/10000/15000 步 EMA checkpoint 各跑一次 teacher-forced 诊断（`diag_main_s0_step*.json`）；**APD 的 train/val 比未实现**。
8. 评估冒烟用生成器原始权重而非 EMA：20 步的 EMA（decay 0.999）几乎仍是零初始化，检验不出"K 个样本可区分"。
9. P 模式只报告项（GT 站定帧脚速、min_interperson_dist_abs_err、root_distance_abs_err）写在 summary 的 `E_report_only` 行。

## 5. GPU 运行命令（第 1、2 批释放 cuda:0 之后；受试者留出协议，生成器 seed i 配底座 v2subj s_i）

```bash
cd /home/rpartx3080/CodeSpace/ReGenNet/.claude/worktrees/two-person-action-prediction-model-0c948c
PY=/home/rpartx3080/.local/micromamba/envs/regennet/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=2
# 折已在 CPU 上生成；以下依次为 T1-a..T1-e（也可用 --stage all 一次跑完 folds→…→summary→review）
$PY scripts/run_ntu2p_trackb.py --protocol subjval --stage fold_models --device cuda:0            # 3 折并行，约 50 分钟
$PY scripts/run_ntu2p_trackb.py --protocol subjval --stage bank  --arms main insample --device cuda:0
$PY scripts/run_ntu2p_trackb.py --protocol subjval --stage train --arms main insample --device cuda:0   # main s0/s1/s2 + insample s0，20000 步
$PY scripts/run_ntu2p_trackb.py --protocol subjval --stage eval  --arms main insample --device cuda:0
$PY scripts/run_ntu2p_trackb.py --protocol subjval --stage summary --arms main insample
$PY scripts/run_ntu2p_trackb.py --protocol subjval --stage review --device cuda:0
# 看图后写 results/forecasting/ntu120_label/ntu2p_trackb/subjval/review_verdict.json（{"pass": bool, "notes": str}），再：
$PY scripts/run_ntu2p_trackb.py --protocol subjval --stage summary --arms main insample
```

驱动实际执行的训练命令（i = 0,1,2，三进程并行）：

```bash
$PY train/train_ntu2p_resdiff.py --bank results/forecasting/ntu120_label/ntu2p_trackb/subjval/oof_bank.pt \
  --save_dir save/forecasting/ntu120_label/ntu2p_trackb_subjval_main_s${i}_20000 --protocol subjval --arm main \
  --seed ${i} --num_steps 20000 --device cuda:0
```

评估命令（i = 0,1,2；s0 另加 `--diagnostics --write_review_cases results/forecasting/ntu120_label/ntu2p_trackb/subjval/review_cases.json`；subjval A0 存在时驱动自动加 `--a0_checkpoint`）：

```bash
$PY eval/eval_ntu2p_resdiff.py \
  --checkpoint save/forecasting/ntu120_label/ntu2p_trackb_subjval_main_s${i}_20000/ema/model000020000.pt \
  --base_checkpoint save/forecasting/ntu120_label/ntu2p_v2subj_A6-F-A5f0.05-A4-GH0.5_s${i}_10000/ema/model000010000.pt \
  --bank results/forecasting/ntu120_label/ntu2p_trackb/subjval/oof_bank.pt \
  --manifest_path results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json \
  --cache_dir results/forecasting/ntu120_label/ntu2p_xyz_seq_cache_subjval --split val \
  --baseline_checkpoint save/forecasting/ntu120_label/ntu2p_independent_single_person_o10_p50_subjval_s0_5000/model000005000.pt \
  --ensemble_checkpoints save/forecasting/ntu120_label/ntu2p_v2subj_A6-F-A5f0.05-A4-GH0.5_s{0,1,2}_10000/ema/model000010000.pt \
  --nfe 10 --batch_windows 32 --device cuda:0 \
  --output results/forecasting/ntu120_label/ntu2p_trackb/subjval/eval_main_s${i}_val_nfe10.json
```

条件分支：
- summary 为 `pending_nfe_rerun`：`--stage eval --nfe 20 --arms main` 后重跑 summary。
- summary 为 `contingency_required`（T2）：`--stage train --arms foot inter`、`--stage eval --arms foot inter`、`--stage summary --arms main foot inter`。
- T3（adopt_full，或用户确认只作多假设采纳时加 `--confirm_probabilistic_only`）：`--protocol original --stage fold_models|bank|train`，再 `--protocol original --stage eval --test`（唯一一次，设置从 subjval summary 冻结读取），最后 `--protocol original --stage summary`。

v2subj 底座 s0–s2 的 EMA 终点已存在；subjval 独立 base 已存在；subjval A0 尚未训练（只影响可选参照）。

## 6. 已知限制

- 未在 GPU 上运行：B×K=640 个样本的骨架投影（float64）显存未实测，不足时调小 `--eval_batch_windows`；去噪网络在 Ampere 上走 TF32，只影响数值、不影响结构保证（零初始化与 root 替换都是精确运算）。
- 冒烟数据极小（库 6 条序列、评估 8 窗、K=4），校准、ES、包络判据等统计性质只能在 T1 实测。
- H-val 起步者只有几十人，Brier/熵的误差条大；包络插值在生成器单样本 mpjpe 落到网格外时按端点外推并标注。
- APD 的 train/val 比（4.7 的诊断）未实现。
- GPU 占用检查只认 python 进程；`run_batch12.sh` 在两个驱动之间的 sleep 等待不会被识别，启动前需确认第 1/2 批已全部结束。当前观察：`run_batch12.sh` 的 driver2（A0 等）在等待 `! pgrep -f train_ntu2p_independent_single_person`，而该字符串出现在两个仍在运行的等待 shell 的命令行里（pid 1645313、1646072），driver2 可能一直不会启动，需要主会话处理。
- 若第 2 批（手指去重）改变了主线，底座与 OOF 库须按新主线重建（设计风险 9）。
- 未更新 `AGENTS.md`（本批约束为不修改现有文件），入口条目由主会话补充。
