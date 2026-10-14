# NTU2P 图扩散：独立评价审查修复结果

反证审查与实施前计划：`20261014-114717-ntu2p-graphdiff-da-checkpoint2-and-evaluation-amendment-plan.md`。本次只改 `eval/eval_ntu2p_graphdiff_sampling.py` 与 `eval/eval_ntu2p_v2.py`；正在训练的 model、train、loss、helper 和候选 driver 均未改。本轮采纳规则、实际指标公式与随机噪声分布均未改变。

## 已完成修复

- 统一确定性和随机评价新增 `contact_geometry`，版本 `wrist_or_palm2_torso6_v1`：两只腕 + index1/pinky1 中点，两点取更近；躯干六段 `(0,3),(3,6),(6,9),(9,12),(9,16),(9,17)`。保存最近距离/全部 pair 距离聚合、固定四距离阈值×两后退速度、持续/容许/响应范围与离线平滑定义。
- 随机结果保存实际窗口数、完整验证窗口数、K、NFE、训练 seed/候选/步数、预登记 scope 与 is_smoke。不匹配 C4 seed0 EMA5000、H-val247、K10、NFE20/50、sampling seed2718 的结果标为“冒烟或预登记范围外诊断”，附各项不匹配原因。
- DDIM 与两种 bootstrap 都对固定 sample0/1/2 计算接触评价，不按 GT 选样；保留原 DDIM sample0 名称供旧读取方使用。
- 输出 JSON 与数组导出在计算前检查路径，拒绝已有产物及二者同路径；实际写入使用排他创建，导出父目录自动创建。历史采样 smoke、C1/C2 JSON 与原审计结果均未覆写。
- bootstrap 除 parent manifest/hash 外，核验真实 OOF 模式、未跳过 OOF 断言与旧 K12/saturate codec_config；输出实际库 config SHA256、模式、codec与按动作/步行状态回退层级的窗口计数。
- 明确输出 ES 是逐帧逐关节 xyz 边际 Energy Score；该评分改善不能单独证明联合时序概率质量。没有临时增加联合评分 gate。

## 实测验证

1. 两个评价文件 `py_compile` 和 `git diff --check` 通过。
2. 默认 OOF 库通过；临时改为 in-sample、跳过 OOF 检查、错 codec，均被拒绝。实际默认库未修改。
3. 正式元数据 247/K10/NFE20,50/C4s0@5000 被识别为匹配；4 窗/K2 被识别为诊断。临时已有数组与 JSON/数组同路径均被拒绝，原字节保持不变。
4. 新随机冒烟：旧 C4 100-step checkpoint、2 窗、K3、NFE2、batch1，CUDA 样本生成约 1.4s；实际 JSON 校验窗口2/完整247、K3/NFE2、is_smoke=True，三个分布各含 sample0/1/2 接触，共导出11种轨迹（point/prior＋三分布各三样本），新增父目录成功。产物：`results/forecasting/ntu120_label/ntu2p_graphdiff/review_sampling_20261014-115216.json`、`results/forecasting/ntu120_label/ntu2p_graphdiff/review_sampling_20261014-115216/arrays.pt`。
5. 统一确定性 CPU 冒烟 2 窗通过，实际 JSON 含六段几何定义。产物：`results/forecasting/ntu120_label/ntu2p_graphdiff/review_sampling_20261014-115216_point.json`。这些数字仅验证流水线，不用于架构选择或优化结论。

## 历史产物的口径更正

早期研究审计用三关节均值掌心和五躯干段；C1/C2 与当前所有正式候选使用本文件明确的腕/掌两点、六段 helper，尽管已生成 JSON 没有定义字段。它们的同 JSON prior 指标也由六段 helper 重算，候选与 prior 的比较仍公平，不需重评或覆写。

对原 FD2 247窗数组、后退阈值0.10m/s：

| 距离阈值 | 早期审计 GT 有效数 | 正式 helper GT 有效数 | 早期提前/漏接触数 | 正式 helper 提前/漏接触数 | 正式 helper 后退 bias(s) |
|---|---:|---:|---:|---:|---:|
| 0.15 | 4 | 5 | 3 | 2 | −0.2200 |
| 0.20 | 4 | 3 | 0 | 0 | −0.2500 |
| 0.25 | 4 | 5 | 1 | 2 | −0.5000 |
| 0.30 | 3 | 3 | 1 | 0 | −0.2333 |

早期连续 A→B 最近距离 RMSE 为0.202380m；六段正式 helper为0.202690m。数值相近不代表事件子集相同。旧研究简报的4/4/4/3与提前0.16–0.23s只能按其原代理定义引用，不能与新候选事件结果直接拼接。

## 保留的结论边界

当前图块保留一般跨人上下文和双向时间卷积，是可学习关系交换，不是“真正接触才后退”的硬物理机制；当前 phase 分支是腿观测/速度与 prior root plan 条件的 MLP 系数纠正器，未实现状态转换/显式事件 hazard。后续事件控制候选的机制不属于当前 C1–C4。

Checkpoint 2 的近邻综合未发现首创性或跨任务成绩滥用，但现行 graph/phase 的准确名称及代理定义必须遵循上面更正。训练终点、三 seed、一次 test 和最终图审仍由父代理继续；本修复不提前作采纳或两核心目标均解决的结论。

