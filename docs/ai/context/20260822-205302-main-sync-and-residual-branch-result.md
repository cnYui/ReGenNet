# NTU 双人基线同步与新架构分支结果

## 已完成

- 当前工作区全部 NTU 双人 diffusion、独立单人 baseline、显式跨人 attention 代码和上下文文档已提交。
- 本地 `main` 已快进到提交 `cb88f9f`：`Record NTU two-person attention training and audit`。
- 已将本地 `main` 推送到 `fork/main`（`https://github.com/cnYui/ReGenNet.git`）。
- 推送后本地与远程 `main` 的 commit 均为：

```text
cb88f9fabb4b2ccfdd0c9ffa67a01211b9a34eeb
```

- 已从该干净基线创建本地分支：

```text
feature/ntu2p-baseline-residual-refinement
```

## 后续范围

该分支用于实现并重新训练“冻结 independent single-person xyz baseline + 双人跨人 residual refiner”架构。未经验证的新代码和模型不会回写 `main` 或远程 `fork/main`。
