# NTU 双人 10->50 diffusion L_dm + L_inter exploratory 全量实验计划

## 背景

用户要求先跑全量 `L_dm + L_inter`，再看结果。

原 `20260723-215149-ntu-two-person-forecasting-diffusion-regennet-implementation-plan.md` 的 formal gate 要求 `L_dm-only diffusion` 先超过 copy-last；当前已知该 gate 未通过。因此本实验不能标为 formal stage 2，只能标为 exploratory。

## 实验边界

```text
实验名：ntu2p_diffusion_o10_p50_ldm_inter_exploratory_s0_5000
性质：exploratory，不作为 formal stage 2 结论
数据：完整 train split，max_samples=-1
初始化：随机初始化，不从 L_dm-only warm start
协议：window_len=60, obs_len=10, pred_len=50
模型：与 L_dm-only 相同宽度
loss：L_dm + 1.0 * L_inter
训练步数：5000
val：训练完成后先跑 val，不直接用 test 调参
```

## 训练命令

```text
PYTHONPATH=. /home/rpartx3080/.local/bin/micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python train/train_ntu_2p_forecasting_diffusion.py \
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

## 判断标准

先看 val，至少比较：

```text
copy-last val
L_dm-only best val
L_dm + L_inter exploratory val
```

如果 exploratory 在 val 上同时超过 copy-last 的 `xyz_mse / xyz_mae / mpjpe`，再讨论是否修订 formal 计划；否则继续定位 free sampling / 表示 / 训练目标对齐问题。
