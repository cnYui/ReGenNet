# NTU 双人显式跨人 attention diffusion 训练结果

## 实现与验证

- 新增模块：`model/two_person_transformer.py`。
- diffusion 入口：`model/forecasting_ntu_2p_diffusion.py` 改为 A/B 共享单人投影、人物内 temporal self-attention、双向共享 cross-attention、24-token memory、A/B future decoder 和共享输出投影。
- checkpoint 类型：`ntu2p_forecasting_diffusion_cross_person`。
- 训练参数：`latent_dim=256`、obs encoder 2 层、future decoder 4 层、4 heads、FFN 1024、batch 8、AdamW `1e-4`、seed 0、5000 step、`L_dm + 1.0 * L_inter`。
- 协议：`window_len=60, obs_len=10, pred_len=50`；cosine 1000 timestep；`START_X`；uniform sampler；CUDA `cuda:0`。
- 设备：NVIDIA GeForce RTX 3080，PyTorch 1.7.1；训练峰值显存约 5.4 GiB。
- 参数量：10,918,992。

静态检查、CPU shape/finite/反向 smoke、CUDA batch 2 forward/backward smoke 和 batch 8 正式配置单步显存探测均通过。修改 Person B 观测时，关闭 dropout 的 A 输出发生变化，说明跨人路径实际建立。

## 训练产物

训练目录：

```text
save/forecasting/ntu120_label/ntu2p_cross_person_attention_o10_p50_cuda_full_s0_5000
```

最终 checkpoint：

```text
save/forecasting/ntu120_label/ntu2p_cross_person_attention_o10_p50_cuda_full_s0_5000/model000005000.pt
```

manifest 使用：

```text
results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json
```

最终 step 日志保持 finite：`train_loss=0.106139`、`rot_mse=0.019917`、`inter_loss=0.086221`。

## Val DDIM50

198 条 val 样本，sample seed 0，batch 8，所有 checkpoint 使用相同 `DDIM50` 口径：

| step | xyz_mse | xyz_mae | MPJPE | 三项均超过 copy-last |
|---:|---:|---:|---:|---|
| 1000 | 0.1496479500 | 0.2936344267 | 0.5964104804 | 否 |
| 2000 | 0.1447633184 | 0.2842176027 | 0.5796653990 | 否 |
| 3000 | 0.1295562353 | 0.2696618378 | 0.5506797150 | 否 |
| 4000 | 0.1304981142 | 0.2703133266 | 0.5517053405 | 否 |
| 5000 | 0.1344293773 | 0.2716835645 | 0.5549476664 | 否 |
| copy-last | 0.0619918183 | 0.1267309102 | 0.2778617372 | - |

本轮显式跨人 attention diffusion 尚未通过 `copy-last` 主 gate，也没有超过 independent single-person baseline 的 test gate。该结果只说明在当前训练目标、rot6d 表示、自由 DDIM50 采样和 `L_inter=1.0` 配置下，架构收益未转化为 paired xyz 主指标收益；不能据此否定跨人信息路径。

Val 结果文件：

```text
results/forecasting/ntu120_label/ntu2p_cross_person_attention_o10_p50_cuda_full_s0_5000_val_step{1000,2000,3000,4000,5000}_ddim50/metrics_val.json
```

## 后续边界

本轮尚未完成 direct xyz 的参数量受控三组消融，也未进行 test checkpoint 冻结评估；后续应先在 val 预注册 checkpoint，再执行 independent baseline、copy-last 和 test 对照。
