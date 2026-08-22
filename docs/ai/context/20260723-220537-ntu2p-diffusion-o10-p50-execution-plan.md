# NTU 双人 10->50 diffusion 实施执行计划

## 目标

严格落实 `docs/ai/context/20260723-215149-ntu-two-person-forecasting-diffusion-regennet-implementation-plan.md`，在新分支 `feature/ntu2p-diffusion-o10-p50` 上完成代码实现和测试。

## 当前约束

- 不覆盖既有 `obs20/pred40` checkpoint、cache、视频和 Stage D 结果。
- 新增主线使用 `window_len=60, obs_len=10, pred_len=50`。
- 新表示为 canonical 双人 rot6d：`[B,56,12,T]`。
- 训练扩散目标预测 clean x0，正式模型为 `L_dm + 1.0 * L_inter`。
- 阶段 0 只做工程和数学 gate，不产生论文结论。

## 实施顺序

1. 新增 `utils/ntu_2p_rot6d.py`，集中处理 raw rotvec、canonical rot6d、FK、root rotation、interaction targets/loss。
2. 新增 `data_loaders/forecasting/ntu_2p_diffusion.py`，实现 H5 数据读取、sample_id 级 manifest、train/val/test dataset 和 collate。
3. 新增 `model/forecasting_ntu_2p_diffusion.py`，实现 ReGenNet 风格 Transformer Decoder。
4. 新增 `train/train_ntu_2p_forecasting_diffusion.py`，实现 uniform timestep diffusion 训练、日志、checkpoint schema。
5. 新增 `sample/sample_ntu_2p_forecasting_diffusion.py`，实现 DDIM/DDPM 采样和 rot6d/xyz 输出。
6. 新增 `eval/eval_ntu_2p_forecasting_diffusion.py`，实现 copy-last、direct/diffusion checkpoint 统一 paired 与 interaction 指标。
7. 新增 `scripts/check_ntu_2p_diffusion_gates.py`，覆盖阶段 0 自动 gate。
8. 运行可负担的阶段 0 / smoke 测试；若失败，新建诊断文档并修复。

## 验证口径

- 单元级：shape、finite、split/join round-trip、rotvec->rot6d、FK、零值 interaction loss、受控扰动、梯度。
- 集成级：manifest 无泄漏、dataset/collate、forward、两步 train、checkpoint reload、DDIM2/DDPM2、copy-last/model 小规模 eval。
- 完成前必须按计划文件逐项审计证据，不能用“无明显错误”替代完成证明。
