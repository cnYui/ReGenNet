# AGENTS.md 每日压缩归档（20261019）

- 时间：2026-10-19 03:30（+09:00）
- 压缩前：57 行、24016 字节
- 压缩后：57 行、23939 字节（`wc -lc AGENTS.md`）

## 精简的条目

### 1. `## 当前研究入口` 中「2026-10-14 接触图/双腿/DCT扩散本轮已完成」条目

命中判据：引用文档已完整保存的成串数值。F1/F2/F3 三个 H-val mpjpe 精确值已完整记录在本条引用的 `docs/ai/context/20261014-153106-ntu2p-contact-graph-diffusion-final-result.md`「完整终点成绩」表中。保留结论"退化约2.2%"、FD2 参照 mpjpe .172228、C 组改善幅度及脚滑/穿地/接触距离结论；本条全部 4 个 `docs/ai/context/` 路径保留。

原文完整照抄：

> - 2026-10-14 接触图/双腿/DCT扩散本轮已完成：C1–C4与端到端F1–F3全部CUDA5000-step、EMA终点、subject-holdout val筛选均不采纳，主线仍为v2+FD2。FD2 mpjpe .172228；C组改善仅.57–.61%，图/腿流/扩散无增量；F组.176014/.175991/.176034（退化约2.2%），脚滑−21–22%但穿地.064→.081，接触距离亦变差。按预登记停止该路线，未做三seed复训或test。完整结果与固定23例、全12推人审查入口见 `docs/ai/context/20261014-153106-ntu2p-contact-graph-diffusion-final-result.md`；研究/公平性设计见 `docs/ai/context/20261014-112117-ntu2p-contact-graph-diffusion-design-and-plan.md`、`docs/ai/context/20261014-111722-ntu2p-contact-graph-diffusion-preregistration-amendment.md`、`docs/ai/context/20261014-120112-ntu2p-end-to-end-graphdiff-followup-design-and-plan.md`。DCT+图+扩散已有近邻，不能声称首创或两项核心问题已解决。

精简后：

> - 2026-10-14 接触图/双腿/DCT扩散本轮已完成：C1–C4与端到端F1–F3全部CUDA5000-step、EMA终点、subject-holdout val筛选均不采纳，主线仍为v2+FD2。FD2 mpjpe .172228；C组改善仅.57–.61%，图/腿流/扩散无增量；F组mpjpe退化约2.2%，脚滑−21–22%但穿地.064→.081，接触距离亦变差。按预登记停止该路线，未做三seed复训或test。完整结果与固定23例、全12推人审查入口见 `docs/ai/context/20261014-153106-ntu2p-contact-graph-diffusion-final-result.md`；研究/公平性设计见 `docs/ai/context/20261014-112117-ntu2p-contact-graph-diffusion-design-and-plan.md`、`docs/ai/context/20261014-111722-ntu2p-contact-graph-diffusion-preregistration-amendment.md`、`docs/ai/context/20261014-120112-ntu2p-end-to-end-graphdiff-followup-design-and-plan.md`。DCT+图+扩散已有近邻，不能声称首创或两项核心问题已解决。

（"mpjpe"指明原三个数值所属指标，与 final-result 文档表头一致，不是新事实。）

### 2. `## 当前研究入口` 中「本轮C4随机DDIM50不采纳」条目

命中判据：引用文档已完整保存的过程细节。"66项CPU真实训练/兼容/梯度/EMA检查通过"是已停止路线（F 模式）的实现验证计数，完整内容（含旧 EMA 权重 eval/train 与 RNG 逐位兼容、梯度、reference 冻结、双学习率、EMA 与重载）记录在上一条入口引用的 `docs/ai/context/20261014-153106-ntu2p-contact-graph-diffusion-final-result.md`「可复用产物与验证」一节，以及该文档引用的 `docs/ai/context/20261014-134222-ntu2p-end-to-end-graphdiff-implementation-and-cpu-check-result.md`。设计约束"F模型永久冻结原FD2 reference作为参照"保留；本条 2 个 `docs/ai/context/` 路径保留。

原文完整照抄：

> - 本轮C4随机DDIM50不采纳（ES .1556不如OOF bootstrap .1284，jerk/穿地增加）；F3随机模式未评价，旧采样入口拒绝微调底座与原FD2残差库混用。F模型永久冻结原FD2 reference作为参照，66项CPU真实训练/兼容/梯度/EMA检查通过。正式六段掌腕代理与早期五段审计的事件子集不同，按 `docs/ai/context/20261014-115547-ntu2p-graphdiff-evaluation-review-fixes-result.md` 区分；图是关系交换、腿流是MLP纠正，无硬接触保证。最小事件积分器合成保证通过，但固定门替换root使推人精度退化，完整脚状态/IK未实现，见 `docs/ai/context/20261014-120042-ntu2p-event-controller-feasibility-result.md`。后续须共同设计手部到达、接触/自主运动辨别与root/脚约束，不能重复包装硬门或调宽采纳线。

精简后：

> - 本轮C4随机DDIM50不采纳（ES .1556不如OOF bootstrap .1284，jerk/穿地增加）；F3随机模式未评价，旧采样入口拒绝微调底座与原FD2残差库混用。F模型永久冻结原FD2 reference作为参照。正式六段掌腕代理与早期五段审计的事件子集不同，按 `docs/ai/context/20261014-115547-ntu2p-graphdiff-evaluation-review-fixes-result.md` 区分；图是关系交换、腿流是MLP纠正，无硬接触保证。最小事件积分器合成保证通过，但固定门替换root使推人精度退化，完整脚状态/IK未实现，见 `docs/ai/context/20261014-120042-ntu2p-event-controller-feasibility-result.md`。后续须共同设计手部到达、接触/自主运动辨别与root/脚约束，不能重复包装硬门或调宽采纳线。

## 事实并入

无。两处删掉的内容在引用文档中都有完整、等价的记录，没有并入其它条目。

## 本次刻意保留的内容

- 10-14 两条中的负结果与坑：C4 随机采样 ES 不如 bootstrap、F 组穿地超护栏、F3 随机模式未评价、旧采样入口拒绝混用微调底座与原 FD2 残差库、六段与五段事件子集不同、无硬接触保证、硬门替换 root 使精度退化、完整脚状态/IK 未实现（未完成事项）、不能重复包装硬门或调宽采纳线、不能声称首创。
- 10-14 第一条中的 FD2 H-val 参照 mpjpe .172228 与 C 组 .57–.61%：后续在受试者留出 val 上做筛选时的参照数值和距采纳线的差距。
- 「NTU2P 当前主线与主数字」条目中的"NTU60 占比 0.547，test 为 0.555"：看似是划分统计，但它是预登记划分搜索的匹配目标，说明 subject-holdout val 为何能代表 test，有疑问即保留。
- Stage 0 条目"默认参数下与 inter001 checkpoint 逐位等价"与 EMA 条目"原始权重均与已有 run 逐位相同（`torch.equal`）"：同属验证细节，但关乎现行主线代码路径的默认兼容性，与第 2 条（已停止路线的验证计数）不同，保留。
- 「当前推进中的 NTU 双人 ReGenNet 风格 diffusion」「当前新的主要比较 gate」「当前最佳 residual refiner 的架构图…」：标题里的"当前"已过时，但仍承载窗口协议（60/10/50）、负结果与诊断文档路径；改写措辞有改变含义的风险。
- 「NTU2P 首次 test 评估」「权重 EMA 结论」「root 轨迹 DCT 头结论」「v2 端到端架构」：主数字已被取代，但含 control 在 test 上不过 gate、训练逐位确定、EMA 协议、余弦衰减与 inter 权重 1/3/10 不采纳、root"可修"假设未获支持、手指占 loss 81% 与腿骨拉长等根因诊断、步态护栏未过的偏离记录，以及对照 checkpoint 路径。
- 摆动恢复 Stage 0–3、训练步数预算、Track B 不采纳：现行指标口径、gate、步数预算或负结果。
- `## 文档入口` 全部条目（包括"首次正式运行（PR #7，删 111、精简 1 条）"中的过程数字）：均为仍在使用的流程与基础设施入口，删除会使对应文档失去清理保护；括号内数字收益太小，不值得改动。
