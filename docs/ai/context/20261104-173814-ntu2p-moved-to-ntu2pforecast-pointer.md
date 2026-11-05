<!-- prune:keep -->

# NTU2P 研究已迁到新项目 NTU2PForecast（指引）

2026-11-03 起，NTU120 双人 xyz 预测的训练与研究在新项目 `/home/rpartx3080/CodeSpace/NTU2PForecast` 中继续（用户要求："在/home/rpartx3080/CodeSpace 这个目录下新建一个项目，专门放我这里的训练代码"）。本仓库不再推进 NTU2P 主线；新项目的数据缓存、`dataset/`、`body_models/` 是指向本仓库的只读软链接，**不要删除或改写本仓库的 `results/forecasting/ntu120_label/ntu2p_xyz_seq_cache*`、`dataset/`、`body_models/`**。

## 截至 2026-11-04 的状态

- 新主线：**扩散 BR:ode0 + 落脚规划层**（STG 去噪器 + JiT 流，局部 DCT 11 阶，采样路径施加 FD2 配方；输出端锁脚迈步），取代本仓库的 v2+FD2。受试者留出 val 3 seed：mpjpe .1722 → .1688、滑行帧比 .180 → .032；xyz_mse +3.6%、四肢摆动减少、交互时序未改善。采纳时用户授权放宽了两条预登记门限。
- 进行中：原协议 3 seed 复训与一次 test；关节级交互图（"A 的手 ↔ B 的肩"等跨人关节组边）G1/G2 训练。
- 完整上下文：新项目 `docs/ai/context/20261104-173814-session-context-20261102-20261104.md`；入口为新项目的 `AGENTS.md`。

## 本仓库遗留

- 分支 `feat/ntu2p-jit-flow-rl` 上 2026-10-19 JiT-RL 轮的代码与文档仍未提交；新项目提取时已原样带走这些文件（逐字节一致）。是否在本仓库提交由用户决定；本仓库 `AGENTS.md` 与 main 上被每日维护压缩过的版本可能冲突。
