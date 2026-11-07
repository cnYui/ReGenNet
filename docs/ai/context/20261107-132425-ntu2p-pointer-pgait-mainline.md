# NTU2P 指针更新：主线改为 BR:ode0 + PGait v4b

NTU2P 研究在独立项目 NTU2PForecast 中推进，本仓库只维护 `AGENTS.md` 中的指针条目。

- **新主线**（用户 2026-11-07 "并入主线"）：扩散 BR:ode0 + 训练得到的相位流形步态模块 PGait v4b，取代"BR:ode0 + 落脚规划层"。
- **原协议 test**：
  - 新主线：mpjpe .17989 ± .0005，滑行帧比 .035（GT .030），步数 2.09（GT 2.19）。
  - 对照：BR:ode0 不加模块 .17906，FD2 .18052。
- **永久约束**：用户要求永久去除规则式落脚规划层，之后不再用任何不参与训练的方法提分或改善观感。
- **详细记录**：在 NTU2PForecast 仓库（https://github.com/cnYui/NTU2PForecast ，PR #2），采纳记录为该仓库的 `docs/ai/context/20261107-114111-ntu2p-mainline-adoption-br-ode0-pgait-v4b.md`。
