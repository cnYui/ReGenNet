<!-- prune:keep -->

# NTU2P 模型架构 HTML 图：结果

## 交付

- `docs/ai/context/20261018-184500-ntu2p-v2-fd2-architecture-diagram.html`：单文件、无外部依赖，含深色模式与窄屏布局，浏览器直接打开。页面内带 `<!-- prune:keep -->`，与本文一起长期保留；主线换代后若不再需要，可删除标记让每日维护清理。
- 计划：`20261018-183947-ntu2p-architecture-html-diagram-plan.md`。本轮没有改动任何已跟踪文件，也没有改 `AGENTS.md`。

## 页面内容

1. 总览：一行流程 + 四条设计要点（规范系、残差 DCT、刚体手 + 骨架投影、腿部专用流）。
2. 主线 v2 + FD2 前向数据流：输入 → ① 场景规范化 + 朝向对方系 → ② DCT 位移 token → ③ InterMixer ×6 → ④ root / local / 腿流三个头 → ⑤ IDCT × ramp → ⑥ 组装 55 关节 + 刚体手 → ⑦ 加回 obs_last → ⑧ 可微骨架投影 → 输出。每级标张量形状、参数量与关键超参，并列出零初始化清单。
3. 模块展开 D1–D4：Mixer 块、腿部专用流（GH）、SkeletonProjector 规则、刚体手与坐标回转。
4. 2026-10-14 接触图 / 腿相位 / DCT 扩散纠正器（C1–C4、F1–F3）的数据流与七个候选对照，显式标"未采纳"，结果只引用 `20261014-153106-ntu2p-contact-graph-diffusion-final-result.md` 的数字。
5. 主线训练目标（`args.json` 中实际生效的项）与优化设置。
6. 核对方法与解释边界。

## 核对结果

核对对象为真实 checkpoint `save/forecasting/ntu120_label/ntu2p_v2_A6-F-A5f0.05-A4-GH0.5-FD2_s0_10000/ema/model000010000.pt`，CPU 实例化（`CUDA_VISIBLE_DEVICES=""`，未占用共享 GPU）。

| 模块 | 参数量 | 备注 |
|---|---:|---|
| 输入嵌入（disp / pose / geometry / cond 投影、action / role embedding、coeff_pos） | 112,896 | |
| Mixer 主干 ×6 | 2,579,916 | 每块 429,986 |
| out_norm + root_head + local_head | 512 + 771 + 20,046 | |
| 腿部专用流 | 935,392 | |
| **合计** | **3,649,533** | 等于 checkpoint 的 `num_params`，逐项相加相等 |

- 配置：hidden 256、blocks 6、MLP 比 2、cond_dim 64、root_dct_k 8、local_dct_k 20、input_dct_k 30、dropout 0.1、saturate ramp 5 帧、`hand_mode=rigid`、`person_frame=toward_other`、`canonical_ab_fallback=camera_x`、`kin_proj=True`、`leg_stream=True`。
- 前向钩子实测形状（B=2）：`hidden [B,2,30,256]`；root 头 `[B,2,8,3]`；local 头 `[B,2,20,78]`；腿流 `[B,2,20,8,3]`（输入 456 / 54 / 256 / 64 维）；投影输入输出 `[B,50,2,55,3]`。
- 图纠正器：C4 / F3 的 5000 step EMA checkpoint。可训练参数 496,867（C 组），F 组 4,146,400（先验 3,649,533 + 纠正器 496,867，另有同样大小的冻结 reference）；节点 token `[B,50,2,22,96]`、邻接 `[B,50,2,22,22]`、系数 `[B,2,16,66]`。
- 渲染检查：Chrome 147 headless 在 1100 px、700 px、390 px 与强制深色下截图。发现并修复一处 CSS：`.stage` 的默认颜色变量覆盖了各阶段的功能色，导致所有阶段同为灰蓝；另调整了概览条的换行与窄屏标题折行。

## 读代码时确认、容易写错的细节

- ramp 是 `[0, .2, .4, .6, .8, 1, 1, …]`（0 → 1 在 5 帧内等步长上升），不是 `[0, .25, .5, .75, 1]`。
- 腿流的 `cond` 取的是 SiLU 之后的条件，且 ctx、cond、root 预测三路输入全部 detach；root 头本身仍由主干的损失训练。
- FD2 的 `hand2` 子集只作用于 s2_5 主损失及其归一化常数；脚接触、腿步态、`free_aux_mse` 仍用全部 55 个关节（`train/train_ntu2p_v2.py` 注释已说明）。
- 主线没有镜像增广（`augment_mirror_prob=0`、`mirror_embed=False`）。

## 边界

- 页面只描述结构，不新增任何实验数字；test 数字与 C / F 结果均来自既有文档。
- 主线是确定性回归，不是扩散模型；页面保留了"不能把 L2 提升归因于交互建模"与"图纠正器未采纳、不能声称首创"的解释边界。
- 页面是手写 HTML + CSS 变量（功能色类 `c-in / c-geo / c-emb / c-mix / c-head / c-leg / c-proj / c-out`），后续改结构直接改文件即可，没有生成脚本。
