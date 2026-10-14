# NTU2P 接触图与步态扩散：检索策略与注释书目

日期：2026-10-14（Asia/Tokyo）。阶段：deep-research quick 的 bibliography_agent，只提交检索与原文注释；未做模型设计、性能结论或代码实现。研究范围见 `20261014-110806-ntu2p-contact-graph-diffusion-research-scope.md`。

## 检索策略

- 数据源：CVF 的 ICCV 2023、CVPR 2024、CVPR 2025、ICCV 2025、CVPR 2026 官方全论文目录；arXiv 原文与元数据；InterGen 出版元数据用 DOI 对应的 Crossref 记录补齐。ECVA 全目录也已获取，但最终条目均来自前述来源。
- 范围：2019—2026 年，保留真正相关的奠基工作；原始论文优先，英文原文可纳入；最终条目年份为 2023—2026。
- 纳入：直接覆盖 DCT/频域扩散动作预测、双人/多体扩散互动、显式关节关系、接触校正、动作相位或互动事件同步；有官方原文可检查；优先同行评审版本。
- 排除：仅图像生成或姿态识别；仅综述/博客；找不到原文；相对本轮问题更远、且同主题已被更直接工作覆盖的候选。排除不意味着论文无价值。
- 检索串：`("human motion prediction" OR forecasting OR completion) AND (DCT OR "discrete cosine") AND (diffusion OR denoising)`；`("human-human" OR multi-person OR interaction) AND (diffusion OR contact OR graph OR relation OR synchronization)`；`(phase OR gait OR foot OR contact) AND (motion OR diffusion)`；定向补查 `ReGenNet`、`InterGen`。
- 执行方式：CVF 目录本地按标题关键词匹配，先标题/摘要筛查，再对定向候选下载全文并检查方法段。arXiv API 三个实际查询为 `all:DCT AND all:diffusion AND all:motion`、`all:ReGenNet`、`all:contact AND all:diffusion AND all:interaction`，均返回 HTTP 429；它们不能视作零检索结果。随后改用官方会议索引与已验证 arXiv 编号获取原文。
- 去重：按标准化标题、作者和官方论文 URL 去重，InterGen 预印本与期刊版视为同一项。Semantic Scholar 的 HumanMAC 标题查询经三次重试仍为 HTTP 429，因此本次降级为标题/原文去重，没有伪造 S2 ID。

所有网页、PDF、提取文本及实际 URL/状态/下载时间保存在 `results/forecasting/ntu120_label/ntu2p_graphdiff_research/`。检索日志为 `search_initial.json`、`search_selected.json`、`search_fulltext.json`、`search_latest_fulltext.json`、`search_2026_selected.json`、`cvpr2026_motion_hits.json`。`bibliography_source_map.json` 提供最终 10 项元数据、PDF SHA256、原文证据窗口和页码。PDF 文本页码从 1 开始；InterGen 为本次获取的 arXiv 版页码。

## 筛选记录与覆盖范围

官方五个 CVF 目录共解析 14,486 个标题；此数字是索引规模，不等同于逐篇全文筛查规模。定向候选 16 项，标题/摘要筛查后均获取全文；选入注释书目 10 项，重复项 0。全文筛选排除 6 项：PhysDiff（物理投影背景，正文优先保留更直接的相位工作 POMP）、Social Diffusion（一般多人长期预测）、Nonisotropic Gaussian Diffusion（一般噪声协方差）、Ponimator（以静态近接触姿态展开动态图像/姿态先验，非本轮观测窗口预测）、A Unified Framework for Motion Reasoning and Generation in Human Interaction（语言推理生成任务更远）、Stability-Driven Co-Manipulation（共享物体、flow matching 与载荷稳定性任务更远）。这些 6 项原文仍保留供后续使用。

覆盖分布提示：10/10 为计算机视觉/图形学的计算方法论文，9/10 来自 CVPR/ICCV，1/10 为 IJCV。计算方法占比和会议族集中度均超过 70%，触发 `DISTRIBUTIONAL_SKEW_ADVISORY`；这是本轮技术架构近邻检索的范围特征，不是质量判定。年份分布为 2023：3、2024：2、2025：3、2026：2，无单年达到 70%。未从作者机构推断地域或数据人群。

## 注释书目（N = 10）

### 1. HumanMAC：DCT 系数域扩散预测的直接先例

Chen, L.-H., Zhang, J., Li, Y., Pang, Y., Xia, X., & Liu, T. (2023). HumanMAC: Masked motion completion for human motion prediction. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 9544–9555). https://openaccess.thecvf.com/content/ICCV2023/html/Chen_HumanMAC_Masked_Motion_Completion_for_Human_Motion_Prediction_ICCV_2023_paper.html

- 相关性与方法：将完整动作变换为截断 DCT 系数，对系数加噪并训练去噪模型；推理的 DCT-Completion 将已观察部分与去噪预测结合。
- 主要发现：论文在 Human3.6M/HumanEva-I 上报告随机预测效果；观测一致性通过采样过程中的掩码完成，而非只靠一个最终位置损失。
- 贡献边界：DCT/逆 DCT 与扩散结合做动作预测已经明确存在，不能作为本轮首次提出的概念。
- 质量与限制：ICCV 同行评审原始工作；单人随机预测任务、原文多样性与误差指标不能直接转换为当前 NTU 双人 paired 指标。
- 原文位置：PDF pp. 3–4，Sec. 3.2、4.1、4.2；本地 `humanmac.pdf`、`humanmac.txt`。

### 2. ReGenNet：动作—反应扩散与显式距离损失

Xu, L., Zhou, Y., Yan, Y., Jin, X., Zhu, W., Rao, F., Yang, X., & Zeng, W. (2024). ReGenNet: Towards human action-reaction synthesis. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 1759–1769). https://openaccess.thecvf.com/content/CVPR2024/html/Xu_ReGenNet_Towards_Human_Action-Reaction_Synthesis_CVPR_2024_paper.html

- 相关性与方法：以 actor 运动及可选意图条件生成 reactor；Transformer decoder 扩散去噪器预测干净动作，加入基于相对身体姿态/朝向/平移距离的显式互动损失，并覆盖在线/离线设置。
- 主要发现：论文强调身份角色非对称与互动同步，报告动作反应生成及在线延迟实验。
- 贡献边界：动作—反应扩散、相对距离互动损失及在线反应已有直接先例；原文同时把“对击拳及时躲避”列为合理反应例子，不能从该文推出所有后退都应等待接触。
- 质量与限制：CVPR 同行评审原始工作；反应合成条件与当前只观察双方 obs10 的联合 forecasting 不完全相同，不能照搬其指标声称本项目提升。
- 原文位置：PDF p. 1 的同步/躲避例子，Sec. 3.1–3.2（pp. 3–5）；本地 `regennet.pdf`、`regennet.txt`。

### 3. InterGen：协同去噪与两人距离图正则

Liang, H., Zhang, W., Li, W., Yu, J., & Xu, L. (2024). InterGen: Diffusion-based multi-human motion generation under complex interactions. *International Journal of Computer Vision, 132*(9), 3463–3483. https://doi.org/10.1007/s11263-024-02042-6

- 相关性与方法：参数共享的两个 Transformer 去噪器通过 mutual attention 交换信息；增加全局相对位姿表示、masked joint distance map 与相对朝向损失，并按扩散时间阻尼正则。
- 主要发现：论文报告 InterHuman 双人生成效果与各机制消融；距离图显式关注近距离关节关系，缓解仅生成各自合理动作但关系失配的问题。
- 贡献边界：双人扩散的互注意力、关节距离图与近距离关系监督已有先例，不能仅改称“图关系”作为首次提出。
- 质量与限制：IJCV 同行评审；以文本条件生成双人动作，未等同本项目 paired forecasting；距离图提供几何约束，不能单独证明接触造成后退的物理因果关系。
- 原文位置：arXiv 原文 Sec. 4.2–4.3，pp. 8–10；本地 `intergen_arxiv.pdf`、`intergen_arxiv.txt`。期刊卷期页码由 `intergen_crossref_attempt0.json` 与 arXiv DOI 字段交叉核对。

### 4. Joint-Relation Transformer：显式关节关系与未来距离监督

Xu, Q., Mao, W., Gong, J., Xu, C., Chen, S., Xie, W., Zhang, Y., & Wang, Y. (2023). Joint-relation transformer for multi-person motion prediction. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 9816–9826). https://openaccess.thecvf.com/content/ICCV2023/html/Xu_Joint-Relation_Transformer_for_Multi-Person_Motion_Prediction_ICCV_2023_paper.html

- 相关性与方法：联合维护 joint 与 relation 表示，关系包括相对距离及人物内/跨人物约束；通过 relation-aware attention 融合，额外预测未来关系距离。
- 主要发现：论文报告多人预测基准上的关系建模收益；它直接提供“预测运动同时预测关系”的学习方式。
- 贡献边界：多人关节关系编码、关系增强 attention、未来距离辅助监督均非首次。
- 质量与限制：ICCV 同行评审；确定性多人预测，不是扩散；几何关系预测效果并不保证某类接触事件的时间次序。
- 原文位置：PDF p. 1 摘要与 Sec. 4.1–4.4（pp. 3–5）；本地 `jrt.pdf`、`jrt.txt`。

### 5. InterDiff：DCT 图预测器校正扩散互动

Xu, S., Li, Z., Wang, Y.-X., & Gui, L.-Y. (2023). InterDiff: Generating 3D human-object interactions with physics-informed diffusion. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 14928–14940). https://openaccess.thecvf.com/content/ICCV2023/html/Xu_InterDiff_Generating_3D_Human-Object_Interactions_with_Physics-Informed_Diffusion_ICCV_2023_paper.html

- 相关性与方法：扩散生成未来人—物动作；以接触相关身体点为参考系，用 spatial-temporal graph neural network 预测相对运动并校正扩散输出。图预测器明确使用 DCT/IDCT 预处理，训练用干净互动数据、推理作用于去噪输出。
- 主要发现：参考系下相对运动更容易预测；论文展示校正可缓解接触悬浮/穿透，并在无接触时保持静止物体。
- 贡献边界：扩散 + 接触参考系 + 图网络 + DCT/IDCT 已在同一工作中出现，不能宣称这四个组件的组合从未存在。
- 质量与限制：ICCV 同行评审；任务是人—物互动。Sec. 3.2 明确说明骨架表示下的 contact 定义不充分，因此接触/穿透条件仅施加于 mesh HOI；这与当前 xyz 接触代理的解释边界直接相关。
- 原文位置：PDF pp. 4–5，Sec. 3.2.1–3.2.2；DCT/IDCT 原文在印刷页 14933；本地 `interdiff.pdf`、`interdiff.txt`。

### 6. SyncDiff：动态图同步与频域多体扩散

He, W., Liu, Y., Liu, R., & Yi, L. (2025). SyncDiff: Synchronized motion diffusion for multi-body human-object interaction synthesis. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 11731–11743). https://openaccess.thecvf.com/content/ICCV2025/html/He_SyncDiff_Synchronized_Motion_Diffusion_for_Multi-Body_Human-Object_Interaction_Synthesis_ICCV_2025_paper.html

- 相关性与方法：联合表示各实体与实体对的相对运动；训练 alignment scores，推理对 dynamic graphical model 做显式同步；FFT 分离低/高频成分并分别建模，避免局部高频互动被整体低频运动盖住。
- 主要发现：论文在五个不同多体配置数据集评估同步与互动质量；目标包括接触丢失、穿透和不同步。
- 贡献边界：频域分解、关系图、扩散与同步约束已有较新的直接近邻。它采用 FFT，不能把它错误描述为 DCT 实现。
- 质量与限制：ICCV 同行评审；多体人—物/手—物合成条件与当前 NTU 双人 forecasting 不同，alignment 不等于接触后反应的专门事件判据。
- 原文位置：PDF pp. 1–2，Sec. 3.2–3.5；本地 `syncdiff.pdf`、`syncdiff.txt`。

### 7. POMP：动作相位条件化扩散与物理反馈

Ji, B., Pan, Y., Liu, Z., Tan, S., Jin, X., & Yang, X. (2025). POMP: Physics-consistent motion generative model through phase manifolds. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 22690–22701). https://openaccess.thecvf.com/content/CVPR2025/html/Ji_POMP_Physics-consistent_Motion_Generative_Model_through_Phase_Manifolds_CVPR_2025_paper.html

- 相关性与方法：frame-by-frame autoregressive 系统结合 phase-guided diffusion 运动学模块、物理模拟模块与 aligned-PAE 相位编码；使用周期特征及 Ortho-MoE，把模拟接触冲量反馈给运动生成。
- 主要发现：论文讨论脚滑、漂浮与穿透，以及运动学先验与模拟器之间的差距；它显式利用相位协调周期运动与外部动态。
- 贡献边界：动作相位、周期表示与扩散/接触反馈结合已有先例；“相位扩散解决迈步”本身不可作为首次概念。
- 质量与限制：CVPR 同行评审；依赖额外模拟/接触冲量，不能假设当前 skeleton 数据拥有相同监督。官方 HTML 标题为 Physics-consistent，而 PDF 首标题为 Physics-constrainable；本条采用 CVF 元数据，记录该真实差异，不把二者误拆成两篇论文。
- 原文位置：PDF pp. 1–2，Sec. 3.2–3.4；本地 `pomp.pdf`、`pomp.txt`。

### 8. TIMotion：双人时序与角色顺序的直接近邻

Wang, Y., Wang, S., Zhang, J., Fan, K., Wu, J., Xue, Z., & Liu, Y. (2025). TIMotion: Temporal and interactive framework for efficient human-human motion generation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 7169–7178). https://openaccess.thecvf.com/content/CVPR2025/html/Wang_TIMotion_Temporal_and_Interactive_Framework_for_Efficient_Human-Human_Motion_Generation_CVPR_2025_paper.html

- 相关性与方法：Causal Interactive Injection 把两人序列交错组织；Role-Evolving Scanning 对称处理主动/被动角色变化；Localized Pattern Amplification 增强短程模式，作用于互动生成框架。
- 主要发现：论文以握手先伸手等例子说明角色顺序重要，并报告双人生成与模块消融。
- 贡献边界：交互时序建模与主动/被动角色变化已被直接研究；论文的 causal 命名指序列建模，不能据此等同已测量的物理接触因果。
- 质量与限制：CVPR 同行评审；未为当前 NTU 观测协议提供“手贴到身体后才后退”的专门实验结论。
- 原文位置：Sec. 3.3.1–3.3.3，PDF pp. 4–5（印刷页 7172 起）；本地 `timotion.pdf`、`timotion.txt`。

### 9. Interact2Ar：2026 年双人自回归全身扩散

Ruiz-Ponce, P., Escalera, S., García-Rodríguez, J., Deng, J., & Potamias, R. A. (2026). Interact2Ar: Full-body human-human interaction generation via autoregressive diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 23559–23569). https://openaccess.thecvf.com/content/CVPR2026/html/Ruiz-Ponce_Interact2Ar_Full-Body_Human-Human_Interaction_Generation_via_Autoregressive_Diffusion_Models_CVPR_2026_paper.html

- 相关性与方法：协同去噪器共享全身编码后，用身体、手、全局轨迹专用并行头；自回归扩散与 Mixed Memory 用不同采样率保留短期与长期上下文。
- 主要发现：论文强调一次性整段生成对动态反应的限制，展示扰动适应/时间组合；另报告 jerk 与分身体部位质量评估。
- 贡献边界：双人自回归扩散、专用部位分支、长短记忆与对扰动的反应已有 2026 年近邻，不能只用上述模块作为本轮新颖性依据。
- 质量与限制：CVPR 同行评审；训练主数据为含手部细节的 Inter-X，文本生成与当前 NTU obs10/future50 监督不同。
- 原文位置：PDF pp. 1–2，Sec. 3.2–3.3；本地 `interact2ar.pdf`、`interact2ar.txt`。

### 10. SyncMos：2026 年互动事件依赖与时间同步

Li, L., Kim, D., Ruan, L., Chen, B., Kwon, T., & Rhee, T. (2026). SyncMos: Scalable motion synchronisation for multi-agent scene interaction. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 8174–8182). https://openaccess.thecvf.com/content/CVPR2026/html/Li_SyncMos_Scalable_Motion_Synchronisation_for_Multi-Agent_Scene_Interaction_CVPR_2026_paper.html

- 相关性与方法：dependency-aware story planner 将文字变成有时间依赖的事件；在预训练单人扩散上结合 time-warping 与 diffusion posterior sampling，使多个人物按依赖顺序同步。
- 主要发现：论文直接讨论交接/接收/反应事件的先后与时间错配，即各自姿态自然仍不足以保证整体互动同步；以事件规划与时序修正处理该问题。
- 贡献边界：互动事件依赖与扩散时间同步已有今年先例，也说明需要把“空间靠近”与“事件发生顺序”分开识别。
- 质量与限制：CVPR 同行评审；有文字事件计划及场景控制条件，不是只由 obs10 推断未来接触的 NTU 预测任务；不能直接迁移其同步成功率。
- 原文位置：PDF p. 1，Sec. 4.2、5.1–5.2；本地 `syncmos.pdf`、`syncmos.txt`。

## 检索限制与交接

这是面向模型改造的 quick 近邻检索，不是穷尽所有数据库的系统性综述，不能据此证明“别人没做过”。arXiv API 与 S2 的限流降低了检索覆盖，已保留失败响应；没有把索引规模包装为全文筛查规模。论文的公开性能均来自原工作、没有在本项目复现，不能拿跨任务数字作为本次架构优化结果。

本阶段仅核对存在性、作者/年份/会议或期刊信息，以及与问题有关的原文方法段。后续需由独立 source_verification_agent 复查，模型设计与训练由调用方继续。原文获取与 AI 原文核对不代表用户已人工阅读。本文使用 AI 辅助检索、元数据整理与原文注释。
