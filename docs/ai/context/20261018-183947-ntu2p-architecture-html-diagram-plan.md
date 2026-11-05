# NTU2P 模型架构 HTML 图：设计与计划

## 目标

用户要求用 HTML 画出模型架构图。本轮只做可视化交付：不训练、不改模型代码、不改任何实验结论，也不修改 `AGENTS.md`（是否加入口由用户决定）。

## 画什么

- **主图**：当前主线 v2 + FD2（`A6-F-A5f0.05-A4-GH0.5-FD2`，3,649,533 参数）的前向数据流。每一级标注张量形状、参数量与关键超参，并把腿部专用流（GH）、刚体手、骨架投影单独展开。
- **副图**：2026-10-14 的接触事件图 / 腿相位 / DCT 扩散纠正器（C1–C4、F1–F3）。它建立在主线之上，且按预登记规则全部未采纳；页面显式标注"未采纳"，不与主线并列为"当前模型"。
- **训练目标一栏**：只列 `args.json` 中实际生效的项，帮助理解 GH、FD2 为何存在；不引入新结论。

## 准确性核对（先于写页面）

1. 以源码为准：`model/forecasting_ntu2p_intermixer.py`、`utils/ntu2p_canonical.py`、`utils/ntu2p_kinematic_projection.py`、`utils/ntu2p_gait_losses.py`、`model/forecasting_ntu2p_graphdiff.py`、`utils/ntu2p_contact_graph.py`、`utils/ntu2p_residual_codec.py`、`train/train_ntu2p_v2.py`、`train/train_ntu2p_graphdiff.py`。
2. 用真实 checkpoint 在 CPU 上实例化（`CUDA_VISIBLE_DEVICES=""`，不占共享 GPU），核对总参数量必须等于 checkpoint 的 `num_params`（主线 3,649,533；图纠正器可训练部分 496,867，F 组 4,146,400），并用前向钩子取各模块输入输出形状。
3. 渲染截图（宽屏与窄屏）检查文字溢出与遮挡，再交付。

## 交付

- 单文件 `docs/ai/context/<时间戳>-ntu2p-v2-fd2-architecture-diagram.html`：无外部依赖、含深色模式，可直接用浏览器打开。
- 结果文档记录核对结果、页面入口与已知边界。

## 边界

- 当前主线是确定性回归（DCT-Mixer + 骨架投影），**不是扩散模型**；页面不暗示扩散带来提升。
- 不把 L2 收益归因于交互建模（沿用 `AGENTS.md` 解释边界：没有屏蔽对方观测的消融，`inter_loss_weight=0.01` 在归一化下近似无 inter 项）。
- 图纠正器的 C/F 结果只引用 `20261014-153106-ntu2p-contact-graph-diffusion-final-result.md`，不新增数字。
