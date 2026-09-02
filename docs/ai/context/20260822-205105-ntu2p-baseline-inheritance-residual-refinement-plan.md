# NTU 双人单人能力继承与跨人残差修正设计

## 文档状态

- 状态：待实现设计，尚未产生训练结果。
- 目标：让双人模型初始时严格继承 independent single-person xyz baseline，再由跨人 attention 学习可验证的增量。
- 主 gate：paired `xyz_mse`、`xyz_mae`、`mpjpe` 同时优于 independent single-person；`copy-last` 仍保留为 persistence reference。

## 1. 根本修改

当前 diffusion 模型从纯噪声直接预测联合 rot6d，没有调用 independent checkpoint，因此不存在单人模型能力继承。

新的第一阶段模型必须显式分解为：

```text
base_A = frozen independent model(obs_A, action)
base_B = frozen independent model(obs_B, action)
delta_A, delta_B = cross_person_refiner(obs_A, obs_B, base_A, base_B, action)
pred = base + gate * delta
```

其中 A/B 使用同一个单人模型实例或同一份共享参数。`gate` 初始化为 0，且修正量第一帧固定为 0；因此模型初始输出严格等于 independent baseline，并保持首帧连续。

## 2. 第一阶段：直接 xyz residual-refinement

### 2.1 为什么先做 xyz

现有 independent checkpoint 的输入和输出都在 xyz 空间，而当前 diffusion 的主干在 canonical rot6d 空间。rot6d checkpoint 无法无损加载到 xyz Transformer，强行迁移会混淆“继承失败”和“表示不匹配”。

第一阶段直接在 xyz 空间使用同一份 independent checkpoint，隔离验证跨人模块本身是否有净收益。

### 2.2 Baseline 分支

- 从现有 `independent_single_person_xyz` checkpoint 加载 `NTULabelXYZTransformer(num_persons=1)`。
- A、B 分别取 paired xyz 的单人观测，并调用同一份共享参数。
- baseline 分支默认冻结；训练和推理均保持 deterministic。
- `base_A/base_B` 的第一帧已经等于各自 obs 最后一帧。

### 2.3 Residual 修正分支

- 输入：A/B 的 obs10、`base_A/base_B` future50、动作标签。
- A/B 保持独立 token 流，人物内 temporal self-attention 后执行双向 cross-person attention。
- 输出为 `delta_A/delta_B`，形状与 future xyz 相同。
- 使用共享输入/输出投影和共享 cross-person 参数，避免收益来自 A/B 专属参数量。
- `delta[:, 0] = 0`，不允许修正首帧。
- 输出：`pred = base + alpha * delta`；最终 `delta_head` 的 weight/bias 零初始化，`alpha` 初始化为 1，使模型初始输出逐元素等于 baseline，同时保留残差 head 的有效梯度。首版不把 `alpha` 过 sigmoid，避免有限值 sigmoid 永远无法达到严格的 0；后续若出现过度修正，再单独评估有界参数化。实现中保留 `alpha=0` 的确定性回归测试。

### 2.4 损失

第一版只使用最终 paired xyz 目标：

```text
L = L_xyz_mse + 0.1 * L_xyz_mae
  + 0.1 * L_velocity
  + 0.01 * ||delta||²
  + lambda_inter * L_inter
```

`lambda_inter` 首轮使用 `0`，随后单独扫描 `0.01/0.05/0.1`。关系损失只能作用于最终 `pred`，不能替代绝对 xyz 监督。

### 2.5 训练阶段

1. 冻结 baseline，只训练 refiner 和 gate，验证跨人残差是否能在不破坏 baseline 的情况下带来收益。
2. 只有第一阶段通过 gate，才允许以小学习率解冻 baseline 的 decoder；输入投影和 encoder 仍保持共享且优先冻结。
3. 每个 checkpoint 必须与 `base`、`copy-last` 在同一 val manifest 上比较，并记录 `gate`、首帧误差和 `||delta||`。

## 3. 第二阶段：迁移到 diffusion（条件残差，而非替换 baseline）

若 xyz residual-refinement 通过 gate，再实现 diffusion 版本：

- baseline future xyz 作为显式条件输入，不从 rot6d 重新猜测单人预测。
- diffusion 目标改为 `r = future_xyz - base_xyz`，采样得到 `r_hat` 后输出 `base_xyz + gate * r_hat`。
- 残差的第一帧固定为 0；必要时对采样过程执行 first-frame inpainting。
- 首版关闭 `L_inter`，先验证 residual diffusion 是否超过 deterministic base；关系权重之后再扫描。
- 评估同时报告 teacher-forced、one-step、DDIM5/10/50，固定采样 seed。

不直接把 xyz baseline 权重加载到当前 rot6d `shared_input_proj/shared_output_proj`，因为两者输出表示不同，无法声称这是能力继承。

## 4. 必须通过的消融

在同一 val manifest、`obs10 -> future50` 协议下固定比较：

1. independent baseline；
2. baseline + zero gate（应逐项等于 baseline）；
3. baseline + residual refiner，无 cross-person；
4. baseline + residual refiner，有显式 cross-person；
5. 仅在 4 通过后加入 `L_inter`。

任何模型若首帧误差大于 baseline，或 `gate=0` 输出不等于 baseline，视为实现错误，不进入正式训练。

## 5. 预期判断

- 如果 3 不优于 baseline，说明新增 decoder/损失本身已造成退化。
- 如果 3 优于 baseline、4 不优于 3，说明当前跨人 attention 没有提供可利用增量。
- 如果 4 优于 baseline、加入 `L_inter` 后退化，说明关系损失尺度或定义不合适。
- 只有 4 或 5 稳定超过 independent baseline 后，才值得继续投入 diffusion 采样链路。
