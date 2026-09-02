# NTU2P residual refiner"平移无摆动"问题的定量确认结果

## 结论

诊断（`docs/ai/context/20260902-143842-ntu2p-residual-refiner-regression-to-mean-diagnosis.md`）中的"确定性 L2 回归对多模态未来姿态回归到均值，关节摆动被平均掉，root 轨迹相对被优先学好"这一判断，在 val split（198 条样本）上得到定量证实。

## 复现方式

```bash
PYTHONPATH=. <regennet-env-python> eval/analyze_ntu2p_residual_refiner_articulation.py \
  --manifest_path results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json \
  --checkpoint save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_inter001_s0_5000/model000005000.pt \
  --output results/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_inter001_s0_5000_articulation_analysis.json \
  --device cuda:0
```

原始结果 JSON：`results/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_inter001_s0_5000_articulation_analysis.json`。

## 关键数字

| 指标 | model (residual refiner) | base (inherited single-person) | copy_last | target (GT) |
|---|---:|---:|---:|---:|
| articulation energy（局部姿态帧间摆动能量，`local_pose = xyz - root`，均方） | 5.209e-06 | 4.783e-06 | 0.0（定义上） | 9.475e-04 |
| root energy（root 帧间位移能量，均方） | 3.081e-05 | 3.237e-05 | 0.0（定义上） | 5.311e-04 |
| articulation energy / target 比值 | **0.0055** | 0.0050 | 0.0 | 1.0 |
| root energy / target 比值 | **0.0580** | 0.0610 | 0.0 | 1.0 |
| frozen 关节-帧占比（局部姿态帧间位移 < GT 位移中位数的 10%） | 37.6% | 37.6% | 100.0%（定义上） | 9.8% |

## 解读

1. **摆动能量比值（0.55%）比 root 能量比值（5.8%）低一个数量级**：model 相对 GT 保留的"root 整体位移"信息，是"局部关节摆动"信息的约 10 倍。这直接证实了诊断中"MSE 优化预算被 root 轨迹主导、local_pose 被平均到接近静止"的机制，不是主观视觉印象。
2. **model 与 base 几乎没有区别**（articulation energy 5.209e-06 vs 4.783e-06，root energy 3.081e-05 vs 3.237e-05）：residual refiner 的跨人 cross-attention 分支并没有给角色带来实质性的额外摆动，它主要是在 root/整体轨迹上做了一次小幅精修——这也解释了为什么它能同时超过 base 与 copy-last 的 `xyz_mse/xyz_mae/mpjpe`（这些是 L2 距离类指标，root 轨迹精修足够拉低整体误差），但视频观感和 base 类似，都是"平移"。
3. **frozen 关节-帧占比**：GT 本身也有 9.8% 的关节-帧低于阈值（正常动作也有静止的关节/瞬间），但 model 高达 37.6%，接近 GT 的 4 倍，且逼近 `copy_last` 的 100%——量化确认了"模型输出的局部姿态大部分时间基本不动"。

## 与既有记录的关系

不改变 `docs/ai/context/20260823-084520-ntu2p-residual-refiner-architecture-visualization-result.md` 中已报告的 `xyz_mse/xyz_mae/mpjpe` 结论（该 checkpoint 在这些指标上确实全面超过 base 与 copy-last）；本记录补充说明这些指标本身对"关节摆动是否真实"不敏感，需要单独的 articulation energy 类指标才能捕捉到该问题。

## 新增文件

- `eval/analyze_ntu2p_residual_refiner_articulation.py`：只读推理 + 统计聚合脚本，复用现有 checkpoint 加载与数据管线，不改动任何训练/评估代码。
- `results/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_inter001_s0_5000_articulation_analysis.json`：本次运行的完整原始结果。

## 后续可能方向（未实现，供讨论）

- 对 root 与 local_pose 做独立归一化后再加权，或提高 local_pose 相对权重，缓解梯度被 root 主导的问题。
- 降低/去掉 `delta_reg_weight`，或延长训练步数，让残差分支有更多空间学习非零幅度的摆动。
- 根本性方案：在评估 gate 中新增 articulation energy / frozen-ratio 类指标，避免继续选出"L2 指标好但视觉静止"的 checkpoint；或引入显式多模态建模（如 diffusion 采样、对抗损失）来对抗回归到均值现象。
