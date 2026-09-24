# NTU2P residual refiner 学习率余弦衰减与权重 EMA：设计与计划

前置：`docs/ai/context/20260902-212338-ntu2p-training-step-budget-design.md`（配套约定 2：用余弦衰减或 EMA 降低终点抖动）、`docs/ai/context/20260924-130524-ntu2p-final-10k-3seed-test-result.md`（结论 5：恒定学习率下相邻终点抖动 ≥ seed 标准差）。

## 问题

当前最终数字的协议是 3 seed × 10000 step、取终点。学习率恒定为 3e-4，终点落在哪个点有随机性：

| seed | val mpjpe @9000 | @10000 | \|差\| |
|---:|---:|---:|---:|
| 0 | 0.18272 | 0.18485 | 0.00213 |
| 1 | 0.17735 | 0.18241 | 0.00506 |
| 2 | 0.18394 | 0.18173 | 0.00221 |
| 均值 | | | **0.0031** |

相邻 checkpoint 的差（0.0031）大于 3 seed 终点的标准差（0.0016）。也就是说，主数字里有一部分只取决于在哪一步停。另外，seed 0 的 20000 step 终点在 val / test 上都比 10000 终点低约 2–3%，但按协议不能据此改选 checkpoint。目标是让终点本身更稳定，而不是换个挑选方式。

## 设计

两个互相独立的开关，默认都关，关闭时训练逐位不变：

1. **学习率余弦尾部衰减** `--lr_schedule cosine_tail --lr_decay_start_frac 0.8 --lr_min 3e-5`：前 80% 步数保持 3e-4，最后 20%（10000 step 下是 8000→10000）余弦降到 3e-5。
   - 只衰减尾部，前 8000 step 的轨迹与恒定学习率 run 逐位相同，可以直接当等价性测试。
   - 衰减段对应预算文档里的膝点之后（6000–8000 已进平台），不影响学习阶段。
   - 衰减起点按 `num_steps` 比例计算，因此 5000 step 筛选时是 4000→5000。
2. **权重 EMA** `--ema_decay 0.999`（0 表示关闭）：
   - 每次 optimizer.step 后更新一份影子权重：可训练参数做指数滑动平均，冻结的 base 参数和常量 buffer（`ramp`、`future_sin_pos`）直接复制。
   - 影子权重按与原始权重相同的 checkpoint 格式存到 `save_dir/ema/model{step}.pt`。评估脚本和驱动的 `model*.pt` 匹配都不需要改；放在子目录也不会被原始权重的 glob 误匹配。
   - 用固定 decay、不做 warmup：窗口约 1000 step，等于一个 checkpoint 间隔。到 10000 step 时初始权重的残留占比为 0.999^10000 ≈ 4.5e-5，可以忽略；≤3000 step 的 EMA checkpoint 偏向初始权重，只用来画曲线，不参与判断。
   - EMA 不消耗随机数、不改变训练，原始权重仍逐位不变，所以同一次训练能同时给出 raw 与 EMA 两组结果。

## 实验（2 × 2，全部 s2_5 配置、10000 step、seed 0/1/2、cuda:0）

| 变体 | 训练 | 来源 |
|---|---|---|
| const-raw | 恒定学习率 | 已有：`..._artic_final_s2_5_dct1_root01_s{0,1,2}_10000/` |
| const-EMA | 恒定学习率 + EMA | 新 run `..._artic_ema_s2_5_dct1_root01_s{0,1,2}_10000/ema/` |
| cos-raw | 余弦尾部衰减 + EMA 跟踪 | 新 run `..._artic_cosema_s2_5_dct1_root01_s{0,1,2}_10000/` |
| cos-EMA | 同上 | 新 run `..._artic_cosema_..._10000/ema/` |

新训练 6 个 run，每个约 30 min，共约 3 h。每个 1000-step checkpoint（raw 与 EMA）都评估 val。

用 10000 而不是 5000 做筛选：这次要解决的是最终协议终点的稳定性，衰减段必须落在平台期，5000 step 时模型还在下降段。

### 顺带的等价性测试（不额外花算力）

- const-EMA run 的原始权重：10 个 checkpoint 的 val 指标必须与 const-raw 逐位相同。这证明 EMA 跟踪不扰动训练。
- cos run 的原始权重：1000–8000 的 8 个 checkpoint 必须与 const-raw 逐位相同。这证明衰减只作用在尾部。
- 启动长训前先做 1000 step 快检：默认参数的新代码，以及开启 EMA 的新代码，两者的 `model000001000.pt` 都要与已有 final s0 的同名权重逐张量 `torch.equal`。

## 预先登记的判断规则（只用 val）

对每个变体计算（3 seed）：

- **J（终点抖动）**：各 seed 的 |mpjpe@10000 − mpjpe@9000| 的均值；同时报告 8000–10000 三点的标准差。
- **L（水平）**：终点 mpjpe 均值，并附 xyz_mse / xyz_mae。
- **S（seed 方差）**：终点 mpjpe 的样本标准差。
- **gate**：终点摆动 gate 须 3/3 通过。

采纳条件：J 相对 const-raw（0.0031）下降 ≥ 50%，L 不比 const-raw（0.1830）差出 1 个 seed 标准差（0.0016）以上，且 gate 3/3。

- 多个变体满足时，取 J 最小者；J 相差不到 0.0005 时优先更简单的一个，依次为 const-EMA（训练不变）、cos-raw、cos-EMA。
- 没有变体满足时，保持 const-raw，作为负结果记录。
- 3 seed 下 J 本身噪声较大，结果文档会同时给出逐 seed 数值，不只报均值。

**test 纪律**：只对按上述规则选中的变体的 3 个终点评估 test，其余变体不看 test。const-raw 已有 test 结果。

## 实现

1. `train/train_ntu2p_residual_refiner_xyz.py`：
   - 新增 `--lr_schedule {constant,cosine_tail}`、`--lr_decay_start_frac`、`--lr_min`、`--ema_decay`，默认值保持历史行为；
   - `constant` 时不改 param_group 的 lr；
   - EMA 用一份 deepcopy 的模型，更新放在 `optimizer.step()` 之后；
   - `_save_checkpoint` 增加可选的 `save_dir` 参数，optimizer 可为 None，EMA checkpoint 不存 optimizer；
   - 训练日志记录当前 lr。
2. `scripts/run_ntu2p_articulation_stage3.py`：`_train` 增加 `extra_args` 参数，默认为空，现有调用不变。
3. 新增 `scripts/run_ntu2p_lr_ema.py`：串行训练 const-EMA 与 cos-EMA 两组 × 3 seed；评估 raw 与 `ema/` 的每个 checkpoint；读取已有 const-raw 的 val JSON；输出 `results/forecasting/ntu120_label/ntu2p_lr_ema/`：
   - `summary.md/json`：逐 checkpoint 曲线；
   - `decision.md/json`：J / L / S / gate 与采纳判断；
   - `equivalence.md`：逐位等价核对。
   - 驱动支持 `--test_variant` 参数，判断完成后再单独调用，对选中变体的终点评估 test。
4. 顺带处理 PR #9 审查的两个 minor：horizon 脚本 docstring 把 copy-last 的"外推"改为"重复观测末帧"；`run_ntu2p_final_10k.py` 的 test 评估在权重缺失时给出明确报错。

## 运行与交付

- 先做 1000 step 快检（约 6 min），通过后 `setsid nohup` 后台启动驱动，日志 `results/forecasting/ntu120_label/ntu2p_lr_ema/driver.log`。
- 结果文档 `docs/ai/context/<时间戳>-ntu2p-lr-cosine-ema-result.md`。
- 若采纳新变体：更新 AGENTS.md 的步数预算条目与主数字（val / test），并把后续最终数字的默认训练选项改为选中变体；若不采纳：在 AGENTS.md 记录负结果。
- feature 分支提交后运行 `python3 scripts/ship_pr.py`。
