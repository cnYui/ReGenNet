# NTU xyz 优化候选骨架视频结果

## 对象

优化候选：

```text
checkpoint = save/forecasting/ntu120_label/xyz_loss_optimal_formal/stageA_lf_l050_f075_s0/model000001400.pt
velocity_loss_weight = 0.2
long_loss_weight = 0.05
final_frame_loss_weight = 0.075
```

formal baseline：

```text
checkpoint = save/forecasting/ntu120_label/xyz_loss_optimal_formal/baseline_v0_s0/model000001200.pt
velocity_loss_weight = 0.2
其它新增 loss = 0.0
```

两者均使用：

```text
val cache = results/forecasting/ntu120_label/xyz_cache_len60_o20_p40_opt_split/val_opt_xyz.pt
seed = 0
num_samples for metrics = 295
saved visualization cases = 8
```

没有使用 test cache。

## 代码兼容

更新：

```text
sample/visualize_ntu_label_xyz_tricolor.py
```

视频脚本现在支持自动识别：

```text
metrics_val.json
metrics_test.json
```

也可以用 `--metrics_filename` 显式指定。该修改避免把 validation 可视化错误命名成 test。

## 输出

优化候选数组和完整 val 指标：

```text
results/forecasting/ntu120_label/xyz_loss_optimal_visualization/optimized_s0_eval8
```

优化候选视频：

```text
results/forecasting/ntu120_label/xyz_loss_optimal_visualization/optimized_s0_tricolor_zflip
```

baseline 数组和完整 val 指标：

```text
results/forecasting/ntu120_label/xyz_loss_optimal_visualization/baseline_s0_eval8
```

baseline 视频：

```text
results/forecasting/ntu120_label/xyz_loss_optimal_visualization/baseline_s0_tricolor_zflip
```

每个视频目录包含：

```text
videos/       8 个 mp4
frames/       8 个 first-frame png
arrays/       8 个 case npz
selection.json
selection.csv
run_config.json
summary.md
```

## 渲染规范

```text
蓝色 = input obs20
橙色 = model predicted future40
绿色 = real future40
true two-person xyz = [T,2,55,3]
flip_z_axis = true
body_only = true
fps = 20
resolution = 768 x 768
frames per video = 60
```

## 同 case 验收

候选和 baseline 的 8 个 case 顺序完全一致：

```text
case0000 A009 S011C001P019R001A009
case0001 A009 S012C001P025R001A009
case0002 A019 S019C001P050R001A019
case0003 A001 S006C001P008R002A001
case0004 A004 S001C001P008R001A004
case0005 A006 S005C001P016R002A006
case0006 A005 S006C001P008R001A005
case0007 A011 S002C001P013R002A011
```

文件检查：

```text
optimized non-empty mp4 = 8
baseline non-empty mp4 = 8
optimized non-empty png = 8
baseline non-empty png = 8
all videos decodable = true
all videos frame_count = 60
same_cases = true
metrics file = metrics_val.json
flip_z_axis = true
```

## 完整 val 指标

优化候选：

```text
xyz_mse = 0.021234792
xyz_mae = 0.082965315
mpjpe = 0.172952901
long_xyz_mse = 0.032277689
final_frame_error = 0.254618221
contact_error = 0.214616575
beats_copy_last = true / true / true
```

baseline：

```text
xyz_mse = 0.021507683
xyz_mae = 0.083104069
mpjpe = 0.172961073
long_xyz_mse = 0.032541862
final_frame_error = 0.258979146
contact_error = 0.217306034
beats_copy_last = true / true / true
```

seed0 完整 val 上，优化候选小幅优于 baseline；这不代表三 seed formal 全面稳定优于 baseline。

## 8 case 定性子集对比

```text
optimized better MSE = 1 / 8
optimized better MAE = 1 / 8
mean optimized MSE = 0.017045358
mean baseline MSE = 0.013665999
mean optimized MAE = 0.075171674
mean baseline MAE = 0.067280530
```

判断：

```text
这组固定 8 case 中，优化候选没有表现出稳定视觉优势。
尤其 case0005 A006 的候选误差明显高于 baseline。
视频只能作为 validation qualitative diagnosis，不能支持“优化后视觉拟合更好”的结论。
```

