# NTU xyz 优化候选骨架视频计划

## 目标

为本轮 validation 优化结果生成双人三色 z-flip 骨架视频，并与 formal baseline 使用相同样本对照。

本次不把视频称为“最终最优模型视频”，因为 formal 多 seed 尚未证明优化候选全面稳定优于 baseline。

## 可视化对象

优化候选：

```text
checkpoint = save/forecasting/ntu120_label/xyz_loss_optimal_formal/stageA_lf_l050_f075_s0/model000001400.pt
velocity_loss_weight = 0.2
long_loss_weight = 0.05
final_frame_loss_weight = 0.075
```

对照 baseline：

```text
checkpoint = save/forecasting/ntu120_label/xyz_loss_optimal_formal/baseline_v0_s0/model000001200.pt
velocity_loss_weight = 0.2
其它新增 loss = 0.0
```

选择 seed0 的原因：

```text
1. 两个模型使用同一个 seed，便于同口径比较。
2. 两个 checkpoint 都由 val_opt 选择 best step。
3. 视频只作定性诊断，不代替三 seed formal 汇总。
```

## 数据边界

只使用：

```text
results/forecasting/ntu120_label/xyz_cache_len60_o20_p40_opt_split/val_opt_xyz.pt
```

不使用 test cache，避免在最终配置尚未稳定选定时提前查看 test qualitative 结果。

固定选择 val_opt 前 8 个样本；优化候选和 baseline 使用同样的样本顺序。

## 输出目录

推理数组：

```text
results/forecasting/ntu120_label/xyz_loss_optimal_visualization/optimized_s0_eval8
results/forecasting/ntu120_label/xyz_loss_optimal_visualization/baseline_s0_eval8
```

骨架视频：

```text
results/forecasting/ntu120_label/xyz_loss_optimal_visualization/optimized_s0_tricolor_zflip
results/forecasting/ntu120_label/xyz_loss_optimal_visualization/baseline_s0_tricolor_zflip
```

## 渲染规范

```text
蓝色 = observed obs20
橙色 = model predicted future40
绿色 = real future40
true two-person xyz = [T,2,55,3]
flip_z_axis = true
body_only = true
num_videos = 8
fps = 20
```

## 必要代码兼容

当前 `sample/visualize_ntu_label_xyz_tricolor.py` 固定读取 `metrics_test.json`。

本次需要让它自动识别：

```text
metrics_val.json
metrics_test.json
```

优先读取显式传入文件名；未传时按 `metrics_val.json -> metrics_test.json` 顺序查找。这样不会把 validation 输出错误命名为 test。

## 验收

```text
1. 候选和 baseline 各输出 8 个非空 mp4。
2. 各输出 8 个 first-frame png 和 8 个 npz。
3. 两组 selection.json 的 sample_id/action_code 顺序一致。
4. run_config.json 记录 metrics 文件、checkpoint、flip_z_axis=true。
5. 视频可以正常解码，帧数应为 obs20 + future40 = 60。
6. 结果文档明确视频只作 qualitative diagnosis。
```

