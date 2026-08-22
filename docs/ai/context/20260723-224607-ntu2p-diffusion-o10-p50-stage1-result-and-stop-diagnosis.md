# NTU 双人 10->50 diffusion 阶段 1 结果与停止诊断

## 分支

```text
feature/ntu2p-diffusion-o10-p50
```

## 依据

本记录继续执行 `20260723-215149-ntu-two-person-forecasting-diffusion-regennet-implementation-plan.md`。

关键停止条件：

```text
若 L_dm-only diffusion 未超过 copy-last，先诊断表示、采样或训练对齐，不进入阶段 2。
```

本轮未进入阶段 2，未训练正式 `L_dm + L_inter` 模型。

## 阶段 1A：copy-last full baseline

结果沿用已保存 full val/test 指标。

### val

```text
path=results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_copy_last_full_val/metrics_val.json
num_samples=198
xyz_mse=0.061991818437371594
xyz_mae=0.1267309117347303
mpjpe=0.2778617330271788
final_frame_error=0.43888695252062093
relative_joint_vector_error=0.4287607961832875
relative_root_translation_error=0.2660575563257391
relative_orientation_error=0.5582025403326208
contact_error=0.3646956068096739
```

### test

```text
path=results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_copy_last_full_test/metrics_test.json
num_samples=1253
xyz_mse=0.06901601683838979
xyz_mae=0.13713413976424232
mpjpe=0.29603687946737434
final_frame_error=0.47460214395287126
relative_joint_vector_error=0.3977084966392776
relative_root_translation_error=0.24145499782165525
relative_orientation_error=0.5762486095820439
contact_error=0.3782830087546815
```

## 阶段 1B：direct xyz Transformer formal seed0

训练输出：

```text
save_dir=save/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_direct_xyz_formal_s0
best_by_val=model000001000.pt
```

val checkpoint 指标：

```text
step=500  xyz_mse=0.04885
step=1000 xyz_mse=0.03430
step=1500 xyz_mse=0.03834
```

冻结 `model000001000.pt` 后执行 full test：

```text
path=results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_direct_xyz_1000_full_test/metrics_test.json
num_samples=1253
xyz_mse=0.04564785470194514
xyz_mae=0.11381470438179167
mpjpe=0.23874186631069122
final_frame_error=0.36113784549622563
relative_joint_vector_error=0.3175386051273498
relative_root_translation_error=0.1850170786035413
relative_orientation_error=null
contact_error=0.2900014914994419
```

判断：

```text
direct xyz 在 full test 的 xyz_mse、xyz_mae、mpjpe 均超过 copy-last。
direct xyz 没有 rot6d root orientation，因此 relative_orientation_error 为 null。
```

## 阶段 1C：diffusion L_dm-only formal seed0

训练从 `model000001000.pt` 恢复到 5000 step，未使用 `--overwrite`。

训练输出：

```text
save_dir=save/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_ldm_only_s0_5000
final_checkpoint=model000005000.pt
inter_loss_weight=0.0
loss=L_dm
```

训练日志末端：

```text
step=5000
train_loss=0.008212
rot_mse=0.008212
inter_loss=0.000000
```

所有 loss 有限，checkpoint 正常保存。

## L_dm-only val checkpoint 扫描

固定 val、sample_seed=0、DDIM50，扫描 saved checkpoints：

| checkpoint | xyz_mse | xyz_mae | mpjpe | final_frame_error | relative_joint_vector_error | relative_root_translation_error | relative_orientation_error | contact_error | 主 gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| copy-last | 0.061991818 | 0.126730912 | 0.277861733 | 0.438886953 | 0.428760796 | 0.266057556 | 0.558202540 | 0.364695607 | baseline |
| 1000 | 0.122547762 | 0.244432991 | 0.511297428 | 0.571843097 | 0.590741230 | 0.465921584 | 0.834658034 | 0.275464581 | fail |
| 2000 | 0.081518341 | 0.203388623 | 0.419235539 | 0.499084934 | 0.515415011 | 0.402291340 | 0.709896845 | 0.249832712 | fail |
| 3000 | 0.061356412 | 0.176605728 | 0.360241135 | 0.443898543 | 0.463839303 | 0.335528923 | 0.708811779 | 0.348029458 | fail |
| 4000 | 0.055710323 | 0.164753369 | 0.339501942 | 0.425887358 | 0.447837008 | 0.323726543 | 0.667100722 | 0.349999458 | fail |
| 5000 | 0.064220135 | 0.172346908 | 0.360640772 | 0.434417746 | 0.494704685 | 0.393958583 | 0.639198663 | 0.355744592 | fail |

最佳 `xyz_mse` 是 step 4000，但 `xyz_mae` 和 `mpjpe` 均明显差于 copy-last，因此不能判定为超过 copy-last。

## L_dm-only DDIM 步数补查

对 val 最佳候选 `model000004000.pt` 补查 DDIM5/10/50：

| sampling | xyz_mse | xyz_mae | mpjpe | final_frame_error | relative_joint_vector_error | relative_root_translation_error | relative_orientation_error | contact_error | 主 gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| copy-last | 0.061991818 | 0.126730912 | 0.277861733 | 0.438886953 | 0.428760796 | 0.266057556 | 0.558202540 | 0.364695607 | baseline |
| DDIM5 | 0.054057383 | 0.162132262 | 0.333937530 | 0.418275628 | 0.435937101 | 0.318877890 | 0.656015745 | 0.328782248 | fail |
| DDIM10 | 0.054810996 | 0.163113407 | 0.336274255 | 0.421939285 | 0.442794082 | 0.320987633 | 0.660641913 | 0.350814194 | fail |
| DDIM50 | 0.055710323 | 0.164753369 | 0.339501942 | 0.425887358 | 0.447837008 | 0.323726543 | 0.667100722 | 0.349999458 | fail |

补查结论：

```text
DDIM5/10 能进一步改善 MSE 和 final/contact，但 MAE、MPJPE、relative root/orientation 仍差于 copy-last。
采样步数不是当前 gate 失败的充分解释。
```

## Gate 判断

阶段 1C 未通过进入阶段 2 的硬门槛。

具体证据：

1. diffusion `L_dm-only` 可以从噪声生成 finite 双人 future50。
2. `L_dm-only` 训练 loss 可下降，最终 rot MSE 有限。
3. val 上最佳候选仅在 `xyz_mse` 上优于 copy-last。
4. val 上所有已查 checkpoint / DDIM 配置都没有同时超过 copy-last 的 `xyz_mse`、`xyz_mae`、`mpjpe`。

因此：

```text
不得进入阶段 2 diffusion L_dm + L_inter formal training。
不得把当前 L_dm-only diffusion 写成通过 baseline 的正式论文结果。
不得在 xsub.test 上评估 L_dm-only diffusion 作为模型选择依据。
```

## 初步诊断

当前现象与历史 obs20/pred40 free sampling 问题一致：训练端 `START_X` denoising loss 能下降，但多步从高斯噪声反推的样本在 xyz/MPJPE 上无法稳定超过 copy-last。

更具体地：

```text
MSE 可在 step 4000 + DDIM5 达到 0.054057383，说明生成不是完全无效。
MAE 和 MPJPE 明显高于 copy-last，说明错误分布更宽，几何上仍有较多关节偏差。
relative root/orientation 指标差于 copy-last，说明双人相对位姿和朝向在采样链中不稳定。
contact_error 有时优于 copy-last，但这是辅助指标，不能覆盖主 paired gate 失败。
```

## 下一步候选

后续必须先开诊断/修复实验，而不是进入阶段 2 主实验。

候选方向：

1. 对齐训练和采样目标：只在 val 上比较 one-step / high-noise / DDIM 小步数，确认 free sampling 误差来源。
2. 增加采样端约束：检查 first step 与 obs 最后一帧连续性，避免从第 1 个预测帧开始大偏移。
3. 表示诊断：比较 rot6d FK 后的 xyz 误差与 canonical rot6d 误差，定位是 rotation representation、translation 还是 root orientation 主导。
4. 训练目标诊断：记录高噪 timestep 下 `pred_xstart` 的 xyz 指标，判断 START_X 模型是否只学会中低噪去噪。
5. 若必须保留 diffusion 主线，可先以 direct xyz 或 rot6d teacher-forced 预测作为强初始化/条件先验，但该路线不再是本计划的严格随机初始化 `L_dm-only` 消融，必须另开设计文档。

## 受影响结论

当前可以写入工程记录的结论：

```text
代码实现、阶段 0 gate、copy-last baseline、direct xyz obs10/pred50 baseline 和 L_dm-only diffusion formal seed0 训练已完成。
direct xyz baseline 在 full test 上超过 copy-last。
L_dm-only diffusion 在 val gate 未超过 copy-last，因此阶段 2/3 暂停。
```

当前不能写入论文式结果的结论：

```text
不能声称 ReGenNet-style diffusion 已具备基本预测价值。
不能声称 L_inter 带来增益，因为阶段 2 未执行。
不能报告 diffusion test 指标、FID/action、多样本或视频作为正式结果。
```

## 回归检查

已通过：

```text
PYTHONPATH=. /home/rpartx3080/.local/bin/micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m py_compile \
  utils/ntu_2p_rot6d.py \
  data_loaders/forecasting/ntu_2p_diffusion.py \
  model/forecasting_ntu_2p_diffusion.py \
  train/train_ntu_2p_forecasting_diffusion.py \
  sample/sample_ntu_2p_forecasting_diffusion.py \
  eval/eval_ntu_2p_forecasting_diffusion.py \
  scripts/check_ntu_2p_diffusion_gates.py \
  scripts/build_ntu_2p_diffusion_xyz_cache.py \
  eval/action_ntu_2p_future50_classifier.py
git diff --check
```
