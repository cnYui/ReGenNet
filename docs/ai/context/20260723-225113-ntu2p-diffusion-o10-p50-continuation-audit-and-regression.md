# NTU 双人 10->50 diffusion 继续审计与回归结果

## 分支

```text
feature/ntu2p-diffusion-o10-p50
```

## 审计结论

继续对照 `20260723-215149-ntu-two-person-forecasting-diffusion-regennet-implementation-plan.md` 后，确认阶段 2/3 仍受阶段 1C gate 阻止：

```text
L_dm-only diffusion 在 val 上未同时超过 copy-last 的 xyz_mse、xyz_mae、mpjpe。
因此不得进入 L_dm + L_inter formal training，也不得执行 formal stage 3。
```

本轮没有改变这个实验结论；只补齐代码层回归保护和后续若 gate 通过时需要的评估字段/视频入口兼容性。

## 本轮补强

### 1. 阶段 0 gate 覆盖非 root 关节扰动

文件：

```text
scripts/check_ntu_2p_diffusion_gates.py
```

新增检查：

```text
body joint rot6d 受控扰动 -> joint_mse 必须为正
body joint rot6d 受控扰动 -> trans_mse/orient_mse 应接近 0
```

原因：原 gate 已覆盖 translation 与 root orientation，但非 root 关节扰动的 interaction joint response 不够显式。

### 2. eval 输出 latency

文件：

```text
eval/eval_ntu_2p_forecasting_diffusion.py
```

新增字段：

```text
per_seed[].latency.model_inference_seconds_total
per_seed[].latency.model_inference_batches
per_seed[].latency.model_inference_seconds_per_sample
summary.latency.model_inference_seconds_per_sample_mean
summary.latency.model_inference_seconds_per_sample_std
```

CUDA 下计时前后执行同步，避免异步 kernel 导致延迟低估。

### 3. 双人三色视频脚本支持 obs10/future50 diffusion eval 数组

文件：

```text
sample/visualize_ntu_label_xyz_tricolor.py
```

新增参数：

```text
--array_filename
--obs_len
--pred_len
--obs_label
--generated_label
--real_label
```

默认值保持旧 `obs20/future40` 路线兼容；新 diffusion eval 可传：

```text
--array_filename ntu2p_diffusion_eval_samples.pt
--obs_len 10
--pred_len 50
--obs_label "Input obs10"
--generated_label "Generated future50"
--real_label "Real future50"
```

## 回归命令与结果

### 编译与 whitespace

通过：

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
  eval/action_ntu_2p_future50_classifier.py \
  sample/visualize_ntu_label_xyz_tricolor.py

git diff --check
```

### phase0 gate rerun

命令：

```text
PYTHONPATH=. /home/rpartx3080/.local/bin/micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python scripts/check_ntu_2p_diffusion_gates.py \
  --save_dir results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates_rerun_20260723_2250 \
  --max_samples 1 \
  --batch_size 1 \
  --latent_dim 32 \
  --num_heads 4 \
  --ff_size 64 \
  --overwrite_manifest
```

结果：

```text
pass=true
body_joint_inter_loss=1.9118878924473393e-07
latency.model_inference_seconds_per_sample_mean=0.009198878891766071
```

输出：

```text
results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates_rerun_20260723_2250/gate_summary.json
```

### eval array + latency smoke

命令：

```text
PYTHONPATH=. /home/rpartx3080/.local/bin/micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python eval/eval_ntu_2p_forecasting_diffusion.py \
  --mode diffusion \
  --checkpoint save/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_ldm_only_s0_5000/model000004000.pt \
  --manifest_path results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json \
  --split val \
  --save_dir results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_eval_arrays_smoke_20260723_2250 \
  --batch_size 1 \
  --max_samples 1 \
  --use_ddim \
  --timestep_respacing ddim5 \
  --sample_seeds 0 \
  --save_arrays \
  --save_array_limit 1
```

结果：

```text
num_samples=1
latency.model_inference_seconds_per_sample_mean=0.039122424088418484
arrays=results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_eval_arrays_smoke_20260723_2250/arrays/ntu2p_diffusion_eval_samples.pt
```

### obs10/future50 双人视频 smoke

命令：

```text
PYTHONPATH=. /home/rpartx3080/.local/bin/micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python sample/visualize_ntu_label_xyz_tricolor.py \
  --source_dir results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_eval_arrays_smoke_20260723_2250 \
  --array_filename ntu2p_diffusion_eval_samples.pt \
  --save_dir results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_video_smoke_20260723_2250 \
  --obs_len 10 \
  --pred_len 50 \
  --obs_label "Input obs10" \
  --generated_label "Generated future50" \
  --real_label "Real future50" \
  --num_videos 1 \
  --overwrite
```

结果：

```text
videos=1
save_dir=results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_video_smoke_20260723_2250
```

## 当前边界

本轮补强后，代码层已覆盖后续 formal stage 3 所需的 latency 记录和 obs10/future50 双人 xyz 视频入口。但实验纪律不变：

```text
阶段 1C gate 未通过前，不执行阶段 2 formal L_dm + L_inter。
阶段 2 未通过前，不执行 formal future50 classifier、多样本 FID/action/diversity/multimodality 或正式视频表述。
```
