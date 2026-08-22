# NTU 双人 10->50 diffusion 实现与 smoke 测试结果

## 分支

当前实现分支：

```text
feature/ntu2p-diffusion-o10-p50
```

## 已实现文件

新增：

```text
utils/ntu_2p_rot6d.py
data_loaders/forecasting/ntu_2p_diffusion.py
model/forecasting_ntu_2p_diffusion.py
train/train_ntu_2p_forecasting_diffusion.py
sample/sample_ntu_2p_forecasting_diffusion.py
eval/eval_ntu_2p_forecasting_diffusion.py
scripts/check_ntu_2p_diffusion_gates.py
scripts/build_ntu_2p_diffusion_xyz_cache.py
eval/action_ntu_2p_future50_classifier.py
```

修改：

```text
data_loaders/forecasting/__init__.py
```

## 实现口径

- 新协议固定为 `window_len=60, obs_len=10, pred_len=50`。
- canonical 双人 rot6d 表示为 `[B,56,12,T]`。
- `slot 0:55, channel 0:6` 为 Person A rot6d。
- `slot 0:55, channel 6:12` 为 Person B rot6d。
- `slot 55, channel 0:3` 为 Person A translation。
- `slot 55, channel 6:9` 为 Person B translation。
- FK 统一走 `Rotation2xyz_x(..., pose_rep="rot6d", num_person=2)`，输出 `[B,T,2,55,3]`。
- diffusion 训练入口固定 cosine / START_X / FIXED_SMALL / uniform timestep / `one_step_noise_prob=0`。
- checkpoint 写入 `representation=two_person_rot6d`、`person_order=person_a_then_person_b_assumed`、manifest path/hash、protocol 和 diffusion config。

## 阶段 0 gate 结果

命令：

```text
PYTHONPATH=. micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python scripts/check_ntu_2p_diffusion_gates.py --save_dir results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates --max_samples 1 --batch_size 1 --latent_dim 32 --num_heads 4 --ff_size 64 --overwrite_manifest
```

结果：

```text
pass=true
manifest=results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json
manifest_hash=169a9c9e63d8b2a2211b2b37171e8c23ef891f51ee3b2aa546af3476e2a3d3ba
train_kept=1956
val_kept=198
test_kept=1253
```

覆盖：

- raw/rot6d/xyz shape。
- Person A/B split/join round-trip。
- rot6d -> SMPL-X FK finite。
- q_sample low/high timestep finite。
- `pred==target` 时 interaction loss 为 0。
- translation/root orientation 受控扰动有 loss 响应。
- `L_dm + L_inter` 对 prediction 的 backward 梯度 finite 且非空。
- 两步 optimizer smoke、checkpoint reload。
- DDIM2/DDPM2 finite、非全 0、非逐元素复制 obs 最后一帧。
- 小规模 val eval 输出模型与 copy-last 指标。

## 额外 smoke

### full `L_dm + L_inter` 训练 CLI

命令：

```text
PYTHONPATH=. micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python train/train_ntu_2p_forecasting_diffusion.py --save_dir save/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_train_inter_smoke --manifest_path results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json --num_steps 1 --save_interval 1 --max_samples 1 --batch_size 1 --latent_dim 32 --num_heads 4 --ff_size 64 --decoder_layers 1 --obs_encoder_layers 1 --inter_loss_weight 1.0 --overwrite
```

结果：

```text
checkpoint=save/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_train_inter_smoke/model000000001.pt
train_loss=3.239926
rot_mse=0.665609
inter_loss=2.574317
```

### sample CLI

输出：

```text
results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_sample_smoke/samples.pt
results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_sample_smoke/sampling_config.json
```

### future50 xyz cache + classifier smoke

cache 输出：

```text
results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_xyz_cache_smoke/train_xyz.pt
results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_xyz_cache_smoke/val_xyz.pt
results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_xyz_cache_smoke/test_xyz.pt
```

classifier 输出：

```text
results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_future50_classifier_smoke/classifier_model.pt
```

该 smoke 只使用每 split 1 条样本，因此 `classifier_gate_pass=false` 是预期结果，不能作为语义可分类性结论。

### semantic eval smoke

输出：

```text
results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_eval_semantic_smoke/metrics_val.json
```

覆盖 action/FID/diversity/multimodality 接口。由于样本数为 1，FID 为 `null` 是预期结果。

## 阶段 1A copy-last full baseline

### val

输出：

```text
results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_copy_last_full_val/metrics_val.json
```

核心指标：

```text
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

输出：

```text
results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_copy_last_full_test/metrics_test.json
```

核心指标：

```text
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

## 静态检查

通过：

```text
PYTHONPATH=. micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m py_compile ...
git diff --check
```

并确认新对象可从包入口导入：

```text
NTU2PDiffusionForecastDataset
ntu_2p_diffusion_collate
NTU2PForecastingDiffusionDecoder
two_person_rot6d
```

## 尚未完成的正式实验

本轮已完成代码实现、阶段 0 gate、阶段 1A copy-last full baseline 和各入口 smoke；尚未执行：

1. 阶段 1B direct xyz Transformer obs10/pred50 formal baseline。
2. 阶段 1C diffusion `L_dm-only` formal baseline。
3. 阶段 2 diffusion `L_dm + L_inter` formal training 与 test 对比。
4. 阶段 3 full future50 classifier、多样本生成质量、DDIM/DDPM latency 和视频。

这些属于后续正式实验，不应把当前 smoke checkpoint 写成论文结果。
