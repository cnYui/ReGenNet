# NTU 双人 10->50 diffusion L_dm + L_inter exploratory 启动记录

## 分支

```text
feature/ntu2p-diffusion-o10-p50
```

## 实验性质

```text
exploratory
```

原因：原 formal 计划要求 `L_dm-only diffusion` 先通过 copy-last gate；当前该 gate 未通过。用户要求先全量跑 `L_dm + L_inter` 看结果，因此本实验作为探索性诊断运行，不标为 formal stage 2。

## 输出目录

```text
save/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_ldm_inter_exploratory_s0_5000
```

## 启动命令

第一次使用 `micromamba run` 失败，原因是当前沙箱不能写 mamba lock：

```text
Could not open lockfile '/home/rpartx3080/.cache/mamba/proc/proc.lock'
```

随后改用环境 Python 直接运行：

```text
PYTHONPATH=. /home/rpartx3080/.local/micromamba/envs/regennet/bin/python train/train_ntu_2p_forecasting_diffusion.py \
  --save_dir save/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_ldm_inter_exploratory_s0_5000 \
  --manifest_path results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json \
  --num_steps 5000 \
  --save_interval 1000 \
  --max_samples -1 \
  --batch_size 8 \
  --latent_dim 256 \
  --num_heads 4 \
  --ff_size 1024 \
  --decoder_layers 4 \
  --obs_encoder_layers 2 \
  --inter_loss_weight 1.0 \
  --log_interval 100 \
  --overwrite
```

## 初始状态

训练已通过 step 100，loss 有限。

日志片段：

```text
step=112
train_loss=1.3200284242630005
rot_mse=0.08634503185749054
joint_mse=0.44080406427383423
orient_mse=0.4405698776245117
trans_mse=0.3523094356060028
inter_loss=1.2336833477020264
```

观察：

```text
rot_mse 从早期约 0.50 降到约 0.09。
inter_loss 波动较大，但没有 NaN/Inf。
L_inter 每步调用 SMPL-X FK，速度约 2.8-3.0 秒/step，明显慢于 L_dm-only。
```

## 下一步

等待：

```text
model000001000.pt
model000002000.pt
...
model000005000.pt
```

训练完成后先跑 val：

```text
eval/eval_ntu_2p_forecasting_diffusion.py --mode diffusion --split val --use_ddim --timestep_respacing ddim5/ddim10/ddim50
```

只有 val 结果有明确收益后，才讨论是否需要修订 formal stage 2 计划。
