# NTU2P Track B 结果文档措辞澄清（PR #21 本机审查跟进）

`docs/ai/context/20260926-192241-ntu2p-trackb-residual-diffusion-result.md` 的"解读"第 2 条写道："同一份评估中，3 seed 集成均值 mpjpe 为 0.1665（−3.3%），但 dct_mid 从 0.63 掉到 0.24"。

这里的"3 seed 集成均值"是评估参照项 `v2_ensemble`，即 3 个确定性 v2+FD2 部署底座输出的逐帧平均（`scripts/run_ntu2p_trackb.py` 传入的是 `paths.deploy(s)`），**不是**生成器多个 seed 的集成。

引用它的目的是说明：同为均值估计时，L2 下降伴随四肢摆动被压低。因此生成器单步条件均值带来的 L2 −2%，在摆动与自然度未验证之前，不能当作确定性改进。这一结论不变。

AGENTS.md 中对应的措辞已改为"v2 三 seed 输出平均"。
