# NTU2P 图扩散：实现独立审查与结构验证结果

审查依据：`20261014-111804-ntu2p-contact-graph-diffusion-design-review.md`、`20261014-111722-ntu2p-contact-graph-diffusion-preregistration-amendment.md`。验证计划：`20261014-112009-ntu2p-graphdiff-structural-check-plan.md`。只新增结构检查脚本与本文件；模型、训练、driver 的修订由父代理完成。

本次是实现与协议核对，不能替代候选结果出来后的 deep-research Checkpoint 2 或最终 Checkpoint 3。结论：核心结构可进入完整候选训练；概率质量、两核心目标与新颖性均未获实验证明。

## 结构验证

新增 `scripts/check_ntu2p_graphdiff.py`，实际运行：

`/home/rpartx3080/.local/micromamba/envs/regennet/bin/python scripts/check_ntu2p_graphdiff.py --device cpu`

44 项检查全部通过，结果 `results/forecasting/ntu120_label/ntu2p_graphdiff/structural_checks.json`。

| 检查 | 实测 |
|---|---|
| C1–C4 state_dict 初始化一致 | 每个 tensor torch.equal |
| 零头相对先验最大误差 | 7.30e-7 m，来自再次投影 |
| 首帧连续、prior 冻结 | 首帧 torch.equal；train 模式中 prior 仍 eval、无梯度 |
| K16 smoothstep 编解码 | encode(decode(c)) 最大误差 1.49e-8 |
| 平滑基条件数 | 8.47，满秩 |
| 最终身体骨长均值误差 | 7.06e-8 |
| 刚体手指尖到腕距离误差 | 5.96e-8 m |
| 扩散时间与端点 | 1000 步 START_X；NFE20/50 保留 0/999 |
| 零头 DDIM | 与 prior 误差 7.30e-7 m |
| 一次 head 更新后的梯度 | 整体 2.236；图 0.01437；腿 0.00578；观测编码 0.00157，均有限非零 |
| 接触记忆 | 扰动未来概率/人物 B 几何不改变过去前缀；邻接 [B,T,2,22,22] |
| 接触图跨人依赖 | 固定图几何，仅扰动对方节点，图边增量改变 0.03845 |
| checkpoint | 非零 head 重载后最大输出误差 0 |
| 正式训练设备 | cpu 无授权、CUDA 不可用均抛错 |

最初随机采样失败 `KeyError('y')`：旧 GaussianDiffusion 的 `p_mean_variance` 会直接读取 `model_kwargs['y']`。父代理补齐条件字典后完整重跑通过。没有静默跳过失败检查。

额外用 `git show HEAD:utils/ntu2p_residual_codec.py` 的旧实现与新默认构造直接对照：basis、encoder、encode、decode 全部 torch.equal，config 相等。扩展 smoothstep 未改变旧 Track B 默认行为。脚本 `py_compile` 通过。

## 包络和 DCT 的 oracle 表达诊断

运行全 247 个 H-val 中心窗的 `--only geometry --batch_size 247`，仅诊断既定 K16/8帧基的表示能力，未调整超参数。结果 `results/forecasting/ntu120_label/ntu2p_graphdiff/oracle_hval_checks.json`。

| 指标 | prior | 用 GT 残差编码的 oracle |
|---|---:|---:|
| 全程 MPJPE | 0.17223 | 0.03825 |
| 前 10 帧 MPJPE | 0.07704 | 0.02538 |
| 第 2–10 帧 MPJPE | 0.08310 | 0.02569 |
| 起步 f1–10 相位相关 | 0.32772 | 0.87604 |
| 起步 f11–20 相位相关 | 0.40212 | 0.95381 |
| 已行走 f1–10 相位相关 | 0.90857 | 0.98029 |
| 已行走 f11–20 相位相关 | 0.83292 | 0.99216 |
| 12 推人窗 A 手→B 躯干距离 RMSE | 0.20271 | 0.03233 |

145 个步行人中 35 已行走、110 起步。当前基有足够表达空间，未显示“包络让起步绝对不可修”的阻断证据。oracle 使用 GT，以上数字绝不可当模型预测结果、采纳理由或概率质量。平滑基的一阶连续函数导数为零，但随机系数在第一离散帧差上仍可非零（小样本检查均值 0.02671），因此仍只保证首帧零纠正，不声称离散速度严格延续观测。

## 实现/协议审查

### 已修复的 Major

1. **评估结果误复用**：初版 driver 的 JSON 名称无 steps，已存在即直接返回。同 tag 请求 10000 step 可误读 5000 step 结果并重标预算。父代理新增 checkpoint/step/split/manifest 核验、训练 args 核验及 request.json 请求锁。使用临时缓存中 100-step smoke JSON 请求 10000，实测抛错“已有评估与请求的checkpoint/步数/split/manifest不一致”。
2. **test 采纳记录不完整**：初版仅检查 adopted 和视觉通过布尔值。现已核验采纳记录 subjval/val/10000/[0,1,2] 与正式 test 同步数三 seed。临时 5000-step 单 seed 记录实测被拒绝，未触及 test 数据。
3. **扩散接口条件遗漏**：上文 KeyError 已修复并实测。

driver summarize 用真实 `smoke_eval.json` 在临时目录执行成功；prior extra_variant、自然度、接触与高频 gate 索引有效。未把 smoke 数字作为筛选结果。

### 需要保留的 Major 解释边界

1. **图仍是预测条件而非严格接触反应机制**：点部署在 x=0,t=999 下的图几乎来自冻结 prior；低噪声随机状态会更新几何，但它不是上一轮 x0 的真实皮肤接触。模型保留自由跨人上下文和双向时序卷积，后退不被硬事件门约束。连续距离改善只支持接触几何改善。父代理已预登记四阈值离散事件仅作诊断；结果与结论需继续遵守。
2. **结构对照的归因范围**：C1→C2 同时替换人物内全局均值为骨架邻接、打开软接触邻接和 event_embed，因此收益归于整个类型图包，不能单独称“接触边的因果收益”。C3 的专用腿网络只读取腿观测和 prior root plan，没有显式左右消息/运动状态门；它是相位敏感监督的腿纠正流，不能称已验证周期相位状态机。
3. **概率分布**：train in-sample prior 残差、x=0 的显式点监督和低噪声训练只证明可训练性，不证明部署随机残差校准。若要采纳随机样本，仍需匹配 bootstrap/旧 Track B 的 ES、脚滑、jerk、穿地和分组步態。架构标有 diffusion=True 不是有效生成模型的证据。

### Minor / 后续结果核对

- C1–C4 参数集合相同，开关关闭的 event/leg 参数未参与梯度；“总可训练参数集合相同”可以报告，但有效活跃容量不同，需列明。所有候选两次 forward、同目标系数辅助监督与独立 noise generator 已落实；phase=True 不新增 dropout 调用，配对 RNG 设计合理。
- 继承 parser 的部分开关在新训练入口未实现相应行为，例如 cosine-tail 或 scene augmentation。正式驱动使用预固定默认值可继续；若开放这些开关，应支持或拒绝，不能让 args.json 声称已启用而实际忽略。
- 初审曾疑 delta_reg 正则口径不一致；继续核对发现 InterMixer 原本就正则相对 obs_last 的完整位移，且 prior args.json 权重为 0.01，故撤销“应为0/原先仅refiner残差”的判断。新模型用投影后 delta、旧模型用投影前 delta，只有细小口径差，不能据此阻断。
- summarize 用平均 L2 与分组 phase 护栏；首次筛选可用。最终三 seed 应逐 seed 报退化/改善与自然度，不能均值掩盖某 seed 的新失败。
- request 锁与采纳记录现在核验预算和协议，但未绑定代码版本/架构 config hash。父代理已冻结本轮模型与训练代码；如改配置，应另建 run 前缀/记录，不复用已有权重名。
- 本轮筛选若只有某一核心目标改善，应明示部分改进；不能用自然度 OR 条件宣布“接触先后和规律起步全解决”。

## 最强反对论点

即便 C4 点预测好于 C3，也只证明“相同纠正器上的噪声辅助训练”有效；不证明 DDIM 单样本质量、接触因果或组合首创。现有结构与运行证据足以进行候选实验，但结论只能由同口径的最终 H-val/test 与逐例自然度支持。

