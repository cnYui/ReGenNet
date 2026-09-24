# NTU2P LR/EMA 驱动：PR #10 审查跟进

前置：`docs/ai/context/20260924-155201-ntu2p-lr-cosine-ema-implementation-result.md`；审查来自 `scripts/ship_pr.py` 的本机 Claude Code 审查（PR #10，两条 minor，均已处理）。

## 处理

| 审查意见 | 处理 |
|---|---|
| `_decide` 不看等价性核对结果：核对失败时仍会选出变体 | `_decide(stats, equivalence_passed)`：核对不通过时直接返回 const-raw、不采纳任何变体（实现没有按设计工作时，变体间的差异不能归因于开关本身）；`decision.md` 醒目标注。 |
| `_equivalence` 通过比较 val 评估 JSON 间接判断权重相同，隐含假设评估可复现；reference 为空时 `steps[0]` 抛 IndexError | 改为像启动前快检那样直接对权重 `torch.equal`（逐 key、key 集合也须一致）；缺任何应比较的 checkpoint 或集合为空都判为不通过。 |

## 验证

- 同 seed 权重（final s0 与 Stage 3 `s0_20000` 的 `model000010000.pt`）判定为相同；不同 seed（final s0 与 s1）判定为不同。
- 在 2×2 实验进行中调用：尚未跑完的 run 缺 checkpoint，6 行全部判为 ✗，`_decide` 返回 const-raw。行为符合预期。
- 正在训练的 ema seed 0 已保存的 `model000001000.pt` 与 final s0 同名权重逐位相同。

## 对运行中实验的影响

驱动进程在 15:50 启动时已把旧版模块加载进内存，本次修改不影响它的训练与评估。6 个 run 跑完后，用修复后的代码重跑同一命令 `scripts/run_ntu2p_lr_ema.py`：已有权重与评估 JSON 会全部跳过，只按新逻辑重新生成 `summary.md`、`equivalence.md`、`decision.md/json`。结果文档以重跑后的输出为准。
