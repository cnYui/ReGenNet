# 接触图与双腿分离几何模块实现结果

## 范围与设计依据

已按 `20261014-112117-ntu2p-contact-graph-diffusion-design-and-plan.md` 及 `20261014-111722-ntu2p-contact-graph-diffusion-preregistration-amendment.md` 新增 `utils/ntu2p_contact_graph.py`、`scripts/check_ntu2p_contact_graph.py`。未修改既有模型、训练或历史文档。前置数据诊断见 `20261014-111541-ntu2p-contact-gait-data-audit.md`。

## 接口约定

- `hand_body_distances(xyz)`：输入 `[B,T,2,55,3]`，输出 `[B,T,2,2,6]`；人物维为 self 手→other 身体，手维为左/右。每手取 wrist 与 index1/pinky1 均值掌心两个点，各点到线段做截断投影，取两点更近的距离。六段固定为 `(0,3),(3,6),(6,9),(9,12),(9,16),(9,17)`。
- `causal_contact_memory(contact)`：时间维固定为 1；输出 `1-cumprod(1-contact)`，只依赖 ≤t 的软概率。持续弱接近会累计，不能解释为真实物理事件概率，也不应硬锁人物自主运动。
- `contact_graph_features(xyz,threshold=.20,temperature=.05)`：返回 `distance/contact/memory/approach_velocity/adjacency`。接近速度为前一帧减当前帧距离乘 20 fps，首帧为零。`adjacency[B,T,p,row,col]` 是 self→other 的 body22 图；把掌腕关系映射到另一人的段端点，并使用 other 的图转置补反向边。重复端点取最强线段关系，图边权不因共享 spine3 累加而超 1。
- `contact_distance_loss(pred,target)`：对所有双向、左右手与六段的距离做 MSE，与最邻近曲线指标互补，纯 torch 可微。
- `bilateral_phase_loss(pred,target,obs)`：仅从 obs 最近 3 帧估计每个人前向轴；匹配左右踝前后有符号分离的低通轨迹 MSE（50帧保留11阶，即 ≤2 Hz）。没有 GT 步行mask，也不强迫静止/踢腿动作交替。
- `contact_metrics(obs,target,variants,actions)`：`variants` 为 name→未来轨迹张量，返回 `name->{all,pushing,events}`。`all/pushing` 含 `distance_rmse/distance_mae/pair_distance_rmse/num_samples`，单位米；全体取双向最近手—身体曲线，pushing(action=2) 只取 A→B。事件全报距离 .15/.20/.25/.30 ×退后速度 .10/.20；GT 有效子集、缺失数、事件配对数、接触/后退 MAE、后退有符号 bias 都单列。空子集以 null 表示无定义误差，可严格 JSON 序列化。

5 点双向平滑仅用于 `contact_metrics` 离线评价；模型特征没有时间平滑，前缀记忆与接近速度没有未来读取。模型整体若看全 future 先验，仍不称因果生成器。

## 有意义的验证

运行 `PYTHONPATH=. OMP_NUM_THREADS=2 /home/rpartx3080/.local/micromamba/envs/regennet/bin/python scripts/check_ntu2p_contact_graph.py`，在项目 torch 1.7.1 环境通过：

- 解析点—线段距离误差 `4.44e-16`，刚体旋转+平移后距离误差 `4.44e-16`；移动 A 的手靠近 B 后距离下降。
- 双向跨人图反向索引误差为 0。
- 改变未来 soft contact 对前缀记忆影响为 0，前缀 loss 对未来 contact 梯度为 0。
- 接触距离 loss 对 A 手和 B 躯干梯度均非零且有限；退化零长度线段梯度有限。
- 腿部前后分离扰动产生有效梯度，相同轨迹 loss 为 0。
- 空推人子集与 `json.dumps(...,allow_nan=False)` 通过。

实现途中自检发现 torch 1.7 对空 batch 的 amin 无 identity，已在 `_distance_metrics` 的空子集分支处理；未为缺失事件伪造误差零值。

## 真实 H-val 数组核验

读取同一现成 FD2/v2/A0/GT 的 247 条 H-val 数组，包含 12 条推人；全方法评价耗时约 1.90 s（CPU，两线程）。结果保存 `results/forecasting/ntu120_label/ntu2p_graphdiff_research/contact_graph_module_baseline_metrics.json`。

| 方法 | 全体最近距离 RMSE | 推人 A→B 最近距离 RMSE | 推人 A→B 全手×段距离 RMSE |
|---|---:|---:|---:|
| GT | 0 | 0 | 0 |
| FD2 | 0.187162 | 0.202690 | 0.219642 |
| v2 | 0.184196 | 0.208736 | 0.222461 |
| A0 | 0.191986 | 0.208269 | 0.230747 |

FD2 的原始连续距离 MSE 为 0.0410919，双腿分离 MSE 为 0.0226800；训练仍应按既定 train copy-last 统计归一化，不能用这些 val 数字调权重。

新模块按设计用六段与掌腕双点；前置审计用五段（含双肩横段）与腕/index1/pinky1 三点均值。二者数值略有差别，后续所有模型与对照统一按新模块重算，不能混用两套距离数字。任何一种口径都只是骨架 proxy，不是皮肤表面接触或物理因果保证。
