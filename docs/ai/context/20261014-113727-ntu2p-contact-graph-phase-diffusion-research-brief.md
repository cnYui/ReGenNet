# NTU2P 接触图、步态相位与 DCT 扩散研究简报

日期：2026-10-14。范围：已核验的 10 篇原论文、既有 FD2 的 subject-holdout val 数据审计，以及本轮训练前设计/修订。编写阶段：deep-research quick 的 report_compiler_agent。论断意图在正文编写前保存至 `results/forecasting/ntu120_label/ntu2p_graphdiff_research/20261014-113502-research-brief-claim-intent-manifest.json`。

## 摘要

本轮应围绕接触时序和起步协调检验结构，而不是仅按 DCT 的年代判断架构价值。DCT 系数扩散、接触图校正、相位扩散与事件同步都有已发表先例。已有模型审计显示，正在行走者的短期相位并非最主要短板，起步与脚滑更突出；接触代理事件在验证集很少且依赖阈值。候选需共享监督与训练预算，逐项比较图、相位流和前向加噪。此简报没有新候选训练成绩，也不承诺首次提出、性能提升或两项问题均已解决。

## 问题与证据

DCT 提供可解释的平滑/频率表示，其表达限制取决于截断和解码方式。HumanMAC 已将截断 DCT 系数加噪并做动作预测，不能把 DCT 与扩散组合本身作为创新依据（[Chen et al., 2023，§3.2、4.1–4.2](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_HumanMAC_Masked_Motion_Completion_for_Human_Motion_Prediction_ICCV_2023_paper.pdf#page=3)<!--ref:humanmac--><!--anchor:page:3-4-->）。本轮的判断问题是：表示能否保存迈步和短接触变化，去噪能否改善这些误差，而非算子是否足够新。

ReGenNet 根据 actor 的运动与意图生成 reactor，并提供在线接收动作的设置（[Xu et al., 2024，§3.1–3.2](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_ReGenNet_Towards_Human_Action-Reaction_Synthesis_CVPR_2024_paper.pdf#page=3)<!--ref:regennet--><!--anchor:section:3.2-->）。本项目任务为双方 obs10 与动作标签共同条件下预测 future50，双方未来均不可用。可以复用扩散训练组件，却必须重新验证条件充分性。原论文还把击拳时及时躲避列为合理反应，说明自主退让不能一律压成接触前不动（同文 p.1<!--ref:regennet--><!--anchor:page:1-->）。

图关系提供几何信息交换，不自动提供物理因果保证。JRT 监督未来关节距离，InterGen 用互注意力与近距离关节图损失协调人物（[Xu et al., 2023，§4.2、4.4](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Joint-Relation_Transformer_for_Multi-Person_Motion_Prediction_ICCV_2023_paper.pdf#page=4)<!--ref:jrt--><!--anchor:section:4.4-->；[Liang et al., 2024，§4.2–4.3](https://arxiv.org/pdf/2304.05684#page=8)<!--ref:intergen_arxiv--><!--anchor:section:4.3-->）。InterDiff 明确承认 skeleton contact 定义不充分，即使关节距离能选参考系，也不能等同皮肤接触或受力（[Xu et al., 2023，§3.2.1](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_InterDiff_Generating_3D_Human-Object_Interactions_with_Physics-Informed_Diffusion_ICCV_2023_paper.pdf#page=5)<!--ref:interdiff--><!--anchor:page:5-->）。

[项目数据审计](20261014-111541-ntu2p-contact-gait-data-audit.md#接触监督覆盖与阈值脆弱性)读取 train/val 缓存及旧 FD2 预测，没有运行新模型。12 个推人中心窗口，在后退速度 0.10 m/s、中心距 0.15/0.20/0.25/0.30 m 下，有效事件数为 4/4/4/3；全部阈值共同有效的 val 序列仅 1 条。旧 FD2 的后退起点在各阈值子集平均提前约 0.16–0.23 s，但不能将这些小子集外推为稳定因果规律。**数据口径更正**：本轮前两版计划曾写初步 4/3/4/3；按纯 obs 边界校正后的[原始审计 JSON](../../../../results/forecasting/ntu120_label/ntu2p_graphdiff_research/contact_gait_audit.json)为 4/4/4/3。历史文件保留。

[步态审计](20261014-111541-ntu2p-contact-gait-data-audit.md#步态分解)含 145 个步行人窗口，其中 35 个已走动、110 个起步。旧 FD2 已行走者前 0.5/1.0 s 相位相关为 0.909/0.833，起步者为 0.328/0.402；步数 1.46 对比 GT 2.29，滑行帧比 0.200 对比 0.034。因此相位流需检验起步、左右分离与 root/站立脚协调，同时保留静止状态和既有短程相位。0.5 s 静止观测不能被当作已知未来第一脚或起步时刻。

## 近邻差异矩阵

表 1 对比已核验机制与本轮需要单独验证的条件差异；最后一列表示待检验内容，不表示新颖性判定。

| 原始工作与原文定位 | 已存在的机制 | 本轮需要区分的内容 |
|---|---|---|
| [HumanMAC，Chen et al. (2023)，§4.1–4.2](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_HumanMAC_Masked_Motion_Completion_for_Human_Motion_Prediction_ICCV_2023_paper.pdf#page=3)<!--ref:humanmac--><!--anchor:section:4.1--> | DCT 系数扩散、观测补全 | 双人手/身体时序与步態监督 |
| [ReGenNet，Xu et al. (2024)，§3.2](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_ReGenNet_Towards_Human_Action-Reaction_Synthesis_CVPR_2024_paper.pdf#page=4)<!--ref:regennet--><!--anchor:section:3.2--> | 反应去噪、显式距离损失 | 双方未来均未知的 obs10/future50 |
| [InterGen，Liang et al. (2024)，§4.2–4.3](https://arxiv.org/pdf/2304.05684#page=8)<!--ref:intergen_arxiv--><!--anchor:section:4.3--> | 共享去噪、互注意力、距离图 | 类型边与接触状态记忆的增量 |
| [JRT，Xu et al. (2023)，§4.2、4.4](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Joint-Relation_Transformer_for_Multi-Person_Motion_Prediction_ICCV_2023_paper.pdf#page=4)<!--ref:jrt--><!--anchor:section:4.4--> | 显式 relation 与未来距离监督 | 关系结构相对共享监督对照的收益 |
| [InterDiff，Xu et al. (2023)，§3.2.2，p.6](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_InterDiff_Generating_3D_Human-Object_Interactions_with_Physics-Informed_Diffusion_ICCV_2023_paper.pdf#page=6)<!--ref:interdiff--><!--anchor:page:6--> | 接触参考系、STGNN、DCT/IDCT、扩散校正 | 人/人连续接触代理与输出时序 |
| [SyncDiff，He et al. (2025)，§3.3、3.5–3.6](https://openaccess.thecvf.com/content/ICCV2025/papers/He_SyncDiff_Synchronized_Motion_Diffusion_for_Multi-Body_Human-Object_Interaction_Synthesis_ICCV_2025_paper.pdf#page=4)<!--ref:syncdiff--><!--anchor:section:3.6--> | 动态图同步与 FFT 分解 | DCT 预测协议下的接触/步态归因 |
| [POMP，Ji et al. (2025)，§3.2–3.4](https://openaccess.thecvf.com/content/CVPR2025/papers/Ji_POMP_Physics-consistent_Motion_Generative_Model_through_Phase_Manifolds_CVPR_2025_paper.pdf#page=4)<!--ref:pomp--><!--anchor:page:4-5--> | 相位条件扩散、模拟冲量反馈 | 无真实冲量条件的起步与脚协调 |
| [TIMotion，Wang et al. (2025)，§3.3](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_TIMotion_Temporal_and_Interactive_Framework_for_Efficient_Human-Human_Motion_Generation_CVPR_2025_paper.pdf#page=4)<!--ref:timotion--><!--anchor:section:3.3.1--> | 双人时序交错、角色变化 | 几何接触记忆与动作顺序的差异 |
| [Interact2Ar，Ruiz-Ponce et al. (2026)，§3.2–3.3](https://openaccess.thecvf.com/content/CVPR2026/papers/Ruiz-Ponce_Interact2Ar_Full-Body_Human-Human_Interaction_Generation_via_Autoregressive_Diffusion_Models_CVPR_2026_paper.pdf#page=4)<!--ref:interact2ar--><!--anchor:section:3.3--> | 全身专用分支、自回归扩散、长短记忆 | 固定预测窗、冻结先验后的纠正 |
| [SyncMos，Li et al. (2026)，§4.2、5.2](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_SyncMos_Scalable_Motion_Synchronisation_for_Multi-Agent_Scene_Interaction_CVPR_2026_paper.pdf#page=3)<!--ref:syncmos--><!--anchor:section:4.2--> | 事件依赖、time-warping 与 DPS | 由观测推断时机而非外部文字排程 |

## 候选选择与可检验结论

按[设计](20261014-112117-ntu2p-contact-graph-diffusion-design-and-plan.md#预登记矩阵与判据)及[公平性修订](20261014-111722-ntu2p-contact-graph-diffusion-preregistration-amendment.md#消融公平性修订)，C1 提供新增容量/监督对照，C2 增加接触图，C3 增加双腿相位流，C4 仅再增加前向加噪。各候选共享先验、确定性损失、优化次数与预算。若 C2 优于旧模型却未优于 C1，只能归于整个改造，不能归于图。C3、C4 的结构和扩散增量同理。

接触用投影后轨迹的连续距离曲线作主要几何量，多阈值时序仅作辅助诊断，同时计入漏接触与漏后退。相位采用有符号左右分离及状态转换，避免把静止、踢腿或自主退让硬改成周期步行。GT 接触、相位与起步时刻只能作训练标签或评价子集，不能作为未来输入。

确定性端点与随机 DDIM 分开判断。前者的 paired 三项 L2 收益不证明生成分布更好，后者必须检查固定样本的脚滑、jerk、相位及概率质量，禁止按 GT 选最好样本。选型遵循受试者留出 val、终点 3 seed 复训与一次 test；本简报提供验证问题，不替代预登记采纳规则。

## 局限与披露

检索覆盖 2023–2026 年的 10 项近邻原文，不能证明不存在其他相同设计。接触是骨架中心距代理，阈值与拟合误差影响事件时刻，验证接触子集不足以支撑统计显著或物理因果承诺。旧模型审计限定于指定 H-val；本轮新候选成绩不在本文范围。

AI 完成了检索、原文核对、审计材料整理与简报编写。书目与来源复核由同一代理分阶段进行，未冒充独立人员审查；本文未声称用户人工读过原文、已复现论文代码或进行了新模型训练。完整 APA 7.0 书目见[注释书目](20261014-111922-ntu2p-contact-graph-diffusion-annotated-bibliography.md)，作者/标题版本差异与证据定位更正见[来源验证报告](20261014-112953-ntu2p-contact-graph-diffusion-source-verification.md)。
