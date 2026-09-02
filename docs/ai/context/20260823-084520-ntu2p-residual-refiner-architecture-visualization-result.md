# NTU 双人 residual refiner 架构图与最佳视频结果

## 本次交付

- 当前 feature 分支的真实模型架构 Mermaid 源图：`docs/ai/context/20260823-ntu2p-residual-refiner-architecture.mmd`。
- 渲染图像：`docs/ai/context/20260823-ntu2p-residual-refiner-architecture.svg`、`docs/ai/context/20260823-ntu2p-residual-refiner-architecture.png`。
- 可复用推理导出入口：`sample/export_ntu2p_residual_refiner_xyz_visualization.py`。
- 最佳 checkpoint：`save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_inter001_s0_5000/model000005000.pt`。

## 架构口径

模型输入是双人 `obs_xyz [B,10,2,55,3]` 和动作标签。A/B 分别通过同一个冻结的 `NTULabelXYZTransformer(num_persons=1)` 得到单人预测锚点 `base_A/base_B`；残差分支使用共享输入投影、人物内 temporal self-attention、双向 A↔B cross-person attention、共享 future decoder 和零初始化输出头。最终输出为：

```text
pred_xyz = base_xyz + alpha * ramp(delta)
```

其中第一帧 residual 通过 ramp 固定为 0，保证首帧连续；`alpha` 是可学习共享 scalar，5000 step checkpoint 的值为 `0.8129278`。

## 验证结果

导出脚本使用 manifest 的固定 val split（198 条样本）和 `cuda:0` 完整推理，结果与既有评估一致：

| 指标 | residual refiner | inherited base | copy-last |
|---|---:|---:|---:|
| xyz_mse | 0.03874824 | 0.04130079 | 0.06199182 |
| xyz_mae | 0.10162356 | 0.10607266 | 0.12673091 |
| mpjpe | 0.21526696 | 0.22652291 | 0.27786174 |

可视化源数组保存于：
`results/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_inter001_s0_5000_visualization/source/`。

视频输出于：
`results/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_inter001_s0_5000_visualization/tricolor_zflip/videos/`。

共生成 8 个案例，按 `copy_last_xyz_mse - model_xyz_mse` 降序选择；每个视频为 60 帧、20 FPS、768×768，蓝色为 obs10，橙色为 residual refined future50，绿色为真实 future50。对应首帧 PNG、npz、`selection.json/csv` 和 `summary.md` 位于同一可视化目录。

## 合并记录

Mermaid 文件和导出脚本已加入 feature 分支，随后将该分支快进合并到本地 `main`。视频与训练产物位于项目的 `results/`、`save/` 忽略目录，不改写历史 checkpoint。
