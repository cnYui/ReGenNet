# 图扩散随机采样评价补充计划

本计划先于 C4 完整训练终点评分；用于验证融入扩散后是否还能生成自然随机未来，不能用确定性端点结果代替。

- 评价脚本 `eval/eval_ntu2p_graphdiff_sampling.py`，固定 H-val 247窗，C4 seed0 EMA5000终点。
- K10标准高斯初始系数噪声，seed2718；NFE20/50，eta0，原始时间含0与999，关闭x0截断。50为主采样，20仅成本诊断，两者同噪声。
- 报告样本0的三项pairedL2、前3个固定样本自然度/接触、ES、APD、minADE@1/5/10。minADE只标GT选样上界，不能与单样本主数字混淆。
- 参照：同prior的确定性端点；既有subject-holdout OOF真残差库按动作/步行状态抽样bootstrap，温度0.5/1.0，旧K12/saturate基先解码再编码到同smoothstepK16基，统一SkeletonProjector。校验parent manifest与hash。
- 生成器仍是in-sample残差训练，因此此次随机模式是探索性评价；若不优于bootstrap或自然度护栏失败，明确不推荐随机展示。若通过也须另行OOF复现实验才能作概率采纳，不能直接进入test。
- 数字包括模型自身脚滑、jerk、穿地/悬空、左右分離相位、步数与骨长；渲染GT预选案例，同一固定样本0/1/2展示，不按效果选样。

新增sampling脚本不改变正在训练的任何参数、loss或网络结构。
