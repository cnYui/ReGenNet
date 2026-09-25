# NTU2P 腿部步态变体 GL / GH：设计、实现与 CPU 验证

前置：Stage 1 结果与 Stage 2 计划 `docs/ai/context/20260925-200529-ntu2p-v2-stage1-result-and-stage2-plan.md`；架构设计 `docs/ai/context/20260925-121154-ntu2p-v2-architecture-exploration-design-and-plan.md`。

## 1. 问题

用户要求：不能出现"分数高，但两条腿不前后迈步、整体在漂移"。

- **视觉审查**：所有模型在步行时两腿不交替，整体滑行。
- **定量**：val 上的步行人（GT 未来 pelvis 水平位移 ≥ 0.5 m，共 99 人）。比较预测与 GT 的"左右踝沿行进方向的前后分离" s(t)：
  - A6 在 f1-10 的时间相关只有 0.13；
  - f1-10 的 RMSE 为 0.174，比 copy-last 的 0.159 还差；
  - A5（脚接触损失）修好了脚接触，但没有修好相位：f1-10 相关 −0.02。
- **判断**：观测后 0.5 s 的相位本来可以预测（只看腿的 MLP 在已在走的人上 f1-20 相关 0.87–0.92），所以这是学习问题，不是内在不确定性。

**梯度分析（A6 s0 EMA 终点，150 个 train batch）**：
- 共享主干中，腿的梯度份额约 3%，手指约 83%；
- 腿的梯度约 60% 来自 local_velocity，其中 97% 落在 >2 Hz 的拟合抖动带；
- dct_mid 幅度项与相位无关，它在腿上产生的运动与 GT 的余弦约为 0；
- A6 腿部的低通速度误差是 copy-last 的 1.06 倍。

## 2. 两个变体（嵌套设计：损失完全相同，只差"腿由谁预测"）

| token | 内容 | CLI |
|---|---|---|
| `GL<w>` | 共享主干内重新分配腿部监督，不改结构 | `--leg_gait_loss_weight w --dct_mid_exclude_legs` |
| `GH<w>` | 个人朝向系腿部专用流替换主干的腿局部输出，损失同 GL | `--leg_stream --leg_gait_loss_weight w --dct_mid_exclude_legs` |

### 损失（`utils/ntu2p_gait_losses.py`）

- **腿关节**：L = (1,2,4,5,7,8,10,11)，ℓ(x) = local_pose(x)[..., L]，相机系。
- **leg_pos** = mse(ℓ(pred), ℓ(gt))：全频段、有符号，罚相位。
- **leg_lpvel** = mse(Δ LP ℓ(pred), Δ LP ℓ(gt))：
  - LP 为投影到前 K = 11 个 DCT 基，即 ≤ 2 Hz；
  - 帧差只在 future 内部取，不含观测末帧。
- **总项**：L_GL = w·(leg_pos / c_leg_pos + leg_lpvel / c_leg_lpvel)。常量 c 在 copy-last 上估计，窗口与现有 `_estimate_scales` 相同。
- **dct_mid_exclude_legs**：
  - 旧 `_loss_terms` 用 `copy.copy(args)` 并把 dct_mid 权重置 0；
  - 另加 `dct_mid_nonleg`，即只在 47 个非腿关节上的 dct_mid 幅度项，权重沿用 `--dct_mid_amplitude_loss_weight`，常量为 `c_dct_mid_nonleg`；
  - dct_low 仍覆盖腿：起步者占步行人的 70%，其步态能量在 ≤ 1 Hz。
- **与分析 A 一致**：定义与分析 A 的 lpvel 候选逐式相同。w = 0.5 的标定口径为 A6 s0 EMA 终点，在该点上 GL 约占总 loss 30%，主干腿部梯度份额从 3% 升到约 30%。
- **训练日志**：`leg_pos_share`、`leg_lpvel_share`、`dct_mid_nonleg_share` 自动写入。
- **预登记核对**：step 9000–10000 的腿部两项占比之和中位数应在 20–40%；超出只记为标定偏差，不改权重重跑。

### GH 腿部流（`model/forecasting_ntu2p_intermixer.py` 的 `NTU2PLegStream`，`leg_stream=True`）

- **个人朝向系**：在场景规范系（+Y 为上）中，取观测末 3 帧单位朝向之和的水平方向为"前"；三行依次为前、左、上。
- **输入**（全部只旋转，每人一份，A/B 共享权重）：
  - 腿相对当帧 pelvis 的位置（10×8×3）与速度 ×20（9×8×3）；
  - pelvis 相对观测末帧的轨迹（10×3）；
  - detach 后的主干 root DCT 预测（8×3），转到个人系；
  - detach 后的主干上下文：`out_norm(hidden)` 对 token 取均值，256 维；
  - detach 后的 cond：SiLU 之后，64 维。
- **网络**：LayerNorm(W_leg + W_root + W_ctx + W_cond + b)，宽 512 → GELU → Linear → GELU → Linear(512, 20·8·3)。
  - 末层零初始化，不用 dropout；
  - 输出经 `local_idct` × ramp，旋回场景系再转到朝向对方系；
  - 在 `_decode` 中用 `index_copy` 替换 joints 的腿行。
- **梯度走向**：
  - 主干腿行的梯度为 0；
  - 腿局部类损失只训练腿部流：实测主干梯度 4e-11，是 root 在 local_pose 中解析抵消后的舍入残差；
  - 腿的绝对位置类损失（mse/mae/velocity/foot 等）经 root 回到主干。
- **初始化与兼容**：
  - 腿部流在 `torch.random.fork_rng(devices=[])` 内创建，全局随机流不前进；
  - `CONFIG_KEYS` 增加 `leg_stream`，旧 checkpoint 按 False 加载；
  - `leg_rows` 是非持久 buffer。
- **参数量**：GH 为 3,649,533，其中 A6 为 2,714,141，腿部流为 935,392。

## 3. 步态相位指标（`utils/ntu2p_naturalness.gait_phase_stats`，接在 `compute_naturalness_stats` 末尾）

- **口径**：
  - 竖直方向取 `estimate_up(obs)`；
  - 步行人：GT 未来末帧 pelvis 相对观测末帧的水平位移 ≥ 0.5 m；
  - 分离量 s(t) = (L_ankle − R_ankle)·f，f 为该位移方向，相机系，不平滑。
- **相关与 RMSE**：
  - 分 f01_10 / f11_20 / f21_30 / f31_50 四段，逐人计算 Pearson 相关与 RMSE；
  - 预测在段内的离均差范数 < 1e-6 时，相关不计入，所以 copy-last 的相关为 NaN。
- **分组与准确率**：
  - moving：观测末 5 个帧差的平均水平速度 > 0.2 m/s；其余步行人为 starting；
  - 先迈脚准确率 `gait_lead_foot_acc`。
- **评估输出**（`eval/eval_ntu2p_v2.py`）：
  - 新增 29 个 gait 键；
  - 另加 `gait_sep_rmse_{f01_10,f11_20}_ratio_to_copy_last`；
  - 块级 `num_gait_walk_persons` / `num_gait_moving_persons`。
- **与参考实现一致**：与裁决参考实现逐位相同，并复现现状面板：A6 f1-10 相关 0.128、RMSE 0.174，copy-last RMSE 0.159，99 个步行人 / 29 个 moving。

## 4. 预登记采纳规则（驱动 `decide` → `gait_criteria`，对参照 R = 去掉最后一个步态 token 的配置，3 seed 配对）

**A. 步态主判据（全过）**
- corr_f01_10：均值 Δ ≥ +0.15，绝对值 ≥ 0.25，逐 seed 全正。
- corr_f11_20：均值 Δ ≥ +0.10，绝对值 ≥ 0.35，至少 2 个 seed 为正。
- rmse_f01_10 / copy-last ≤ 1.00。
- 步行 GT 站定帧脚速：A5 底座 ≤ 1.00× R，无 A5 底座 ≤ 0.90× R。
- 滑行帧比 ≤ 1.10× R。

**B. 护栏（全过）**
- 三项 L2 的配对 Δ% 均值各 ≤ +0.5%。
- gate 3/3。
- rmse_f31_50 ≤ 1.05× R。
- A0 否决项通过。

**判定**
- A、B 全过：候选采纳（待审查图）。
- A 与非 L2 护栏都过、只有 L2 越界：不确定，允许补跑一次 w=0.25。
- 其余情况：不采纳。

**GH 与 GL 的取舍**
- GL 与 GH 同底座都完成时，汇总表加一行"GH vs GL 配对"。
- GH 须满足以下任一条才采纳：Δcorr_f01_10 ≥ +0.05 且逐 seed 全正；或 Δmpjpe ≤ −0.3%。否则采纳 GL。
- 两者都失败：本轮不调权重。下一轮候选为"手指去重 α=1/15 + GL w≈0.06"，以及把起步者交给 Track B。

**审查图 C 仍需人工看**：seed 0 的 18 例，加 walk_096、walk_070。

**驱动其它改动**
- 步态 token 必须放在最后，否则 `config_args` 报错。
- `config_stats` 另存 `natural_per_seed`；缺键（旧 JSON）时汇总表显示"—"。
- `_needs_eval` 发现 JSON 缺 `gait_sep_corr_f01_10` 时重算。重算前备份为 `*.pre_gait.json`，重算后模型 L2 与旧 JSON 相差 ≥ 1e-9 时，该 run 报失败。

## 5. CPU 验证（GPU 被 Stage 3 占用，全部在 CPU、2 线程完成）

**默认关闭时逐位等价**
- 训练：A6-F-A5f0.05、A6、A0 用同一 CLI，分别以改动前、改动后的代码各训 5 step。以下全部 `torch.equal`：state_dict、EMA、optimizer 状态、loss_scales、逐步 train_loss。
  - args.json 只新增 4 个开关键；
  - InterMixer 的 `model_config` 多一项 `leg_stream: false`。
- 评估：A6 s0 EMA 终点在 val 198 上以下项全部相同：导出数组逐位相同；L2、articulation、gate、全部已有自然度键数值完全相同。

**新损失**
- leg_pos / leg_lpvel 与分析 A 的 `leg_raw` 相同（`torch.equal`）。
- 掩码取全部关节时，`dct_mid_nonleg` 与原 dct_mid 项相同（`torch.equal`）。
- 常量：c_leg_pos = 0.006456、c_leg_lpvel = 0.000116，与分析 A 的相对差为 0；c_dct_mid_nonleg = 0.021293（全关节为 0.019586）。

**GH**
- 非腿部流参数的初始化与 GL 相同（`torch.equal`），构造后全局 RNG 状态相同。
- 初始输出逐位等于 copy-last（train 与 eval 模式都是）。
- 两个底座上，第 1 步 loss 都与 GL 逐位相同。
- 在 A6 权重上接一个末层随机的腿部流：非腿关节与 A6 逐位相同，首帧误差为 0。
- 个人系正交误差 1e-7，行列式为 +1；刚体等变误差 3.6e-7。

**冒烟**
- 驱动在临时目录跑完 A6-F-{GL,GH}0.5 与 A6-F-A5f0.05-{GL,GH}0.5（外加 A0、A6-F、A6-F-A5f0.05），每个 5 step，保存、评估、汇总全部跑通。
- 用评估的加载函数加载 raw 与 EMA 终点，在 val 前 16 条上前向：输出有限，首帧误差为 0。
- 训练日志含 `*_share`。初始化时（copy-last）两个腿部项各约占 6–7%；30% 的标定是在收敛点上做的。
- 汇总表的 GH vs GL 行正常；补评估路径的备份与 L2 核对正常，差为 0。
- `--dry_run` 列出 6 个训练命令与 30 个评估命令：新 run 各 2 个，A0、A6-F、A6-F-A5f0.05 的 EMA 与 raw 终点全部补评估。

## 6. GPU 启动

在 Stage 3 驱动结束、GPU 空闲后执行：

```bash
PYTHONPATH=. nohup python -u scripts/run_ntu2p_v2_screen.py --stage 2 --steps 10000 \
  --configs A6-F A6-F-A5f0.05 A6-F-A5f0.05-GL0.5 A6-F-A5f0.05-GH0.5 --workers 3 \
  >> results/forecasting/ntu120_label/ntu2p_v2_screen/gait_driver.log 2>&1 &
```

- **规模**：新增 6 个 run，已有 run 只补评估，预计 35–40 min。
- **换底座**：若 A5 审查图被否，改跑 `A6-F-GL0.5 A6-F-GH0.5`；若最终采纳 A5f0.2，把 A5 token 换掉。
- **test**：只对最终采纳方案评估一次。
