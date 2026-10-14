# NTU2P 接触图与步态扩散：来源验证报告

日期：2026-10-14（Asia/Tokyo）。阶段：deep-research 的 source_verification_agent。范围为 10 项书目身份、出版信息及指定关键方法论断；不做架构设计、综合研究结论或模型实现。

## 复核结果

10 项论文身份与作者/年份/出版信息通过，来源拒绝 0；POMP 有 1 项标题版本措辞差异，需随引用保留。10 条指定方法事实通过；4 条无法成立的外推标为 FAIL，不允许进入模型改造结论。

本次重新请求 10 项官方页面，均 HTTP 200；9 个 CVF 官方页面与相应 accepted-version PDF 交叉核对。InterGen 同时重新读取 arXiv 原文、Crossref DOI 记录与 DOI 跳转后的 Springer 正式论文页（均 HTTP 200）。10 个 PDF 的 SHA256 均与已保存原文一致。

作者共 59 个名字条目：58 项文本核对通过；Interact2Ar 中 José García-Rodríguez 因 PDF 字形抽取的 dotless ı 与重音导致自动字符串匹配失败，经原 PDF 首页面像检查后通过。未把这一抽取问题当作作者不存在。

证据入口：`results/forecasting/ntu120_label/ntu2p_graphdiff_research/verification/metadata_recheck.json`、`claim_recheck.json`、`manual_verification_notes.json`。该目录保存重新获取的 `*_official.html`、直接从原 PDF 重新提取的 `*_pdf_read.txt`、InterGen 出版页/DOI 注册记录和 Interact2Ar 首页面像。原文证据没有取自之前的书目摘要或 source map 的摘要窗口。

## 10 项身份与出版信息

| 条目 | 作者数及核对 | 年份与正式载体 | 页码 | 结果 |
|---|---:|---|---|---|
| HumanMAC | 6/6 | ICCV 2023 | 9544–9555 | PASS |
| ReGenNet | 8/8 | CVPR 2024 | 1759–1769 | PASS |
| InterGen | 5/5 | IJCV 2024，132(9) | 3463–3483 | PASS；预印本首次年份 2023 不替代期刊年份 |
| Joint-Relation Transformer | 8/8 | ICCV 2023 | 9816–9826 | PASS |
| InterDiff | 4/4 | ICCV 2023 | 14928–14940 | PASS |
| SyncDiff | 4/4 | ICCV 2025 | 11731–11743 | PASS |
| POMP | 6/6 | CVPR 2025 | 22690–22701 | PASS，标题版本标注见下 |
| TIMotion | 7/7 | CVPR 2025 | 7169–7178 | PASS |
| Interact2Ar | 5/5 | CVPR 2026 | 23559–23569 | PASS，1 项作者名经原页面像补核 |
| SyncMos | 6/6 | CVPR 2026 | 8174–8182 | PASS |

10 项标题均与重新取得的官方元数据一致。POMP 官网元数据标题为 `Physics-consistent Motion Generative Model through Phase Manifolds`，PDF 首标题为 `Physics-constrainable Motion Generative Model through Phase Manifolds`。作者、PDF URL、会议与页码完全对应同一项，不能拆成两篇。此处记录差异，不伪造一致。

InterGen DOI `10.1007/s11263-024-02042-6` 成功跳转至 `https://link.springer.com/article/10.1007/s11263-024-02042-6`；正式页面的标题、5 位作者、IJCV、卷期和页码均与 Crossref 一致。arXiv v3 是 2024-03-28 的作者稿，封面有未填的 Received/Accepted 模板字段；期刊已出版身份的证据来自出版页与 DOI 注册信息。

S2 在上一检索阶段 HumanMAC 查询的初次及三次重试均 HTTP 429，原始响应与时间仍保留；本次按不可用时降级的规则跳过 Tier 0，日志标记 `[S2-API-UNAVAILABLE]`。没有将任何条目标为 `S2_VERIFIED`，也没有填造 S2 ID。官方原文核验与 S2 索引核验是不同证据。

## 关键方法事实：逐条 PASS/FAIL

下述 PDF 页码从原文件第 1 页开始，文本行号指 `verification/*_pdf_read.txt`。节标题与短引文是定位提示，完整段落位于已保存原文。

| ID | 待核论断 | 原文定位与依据 | 结论 |
|---|---|---|---|
| C1 | HumanMAC 在 DCT 系数域进行扩散训练 | `humanmac` PDF p.3，Sec. 4.1，文本 L226 起：`conduct the DCT operation ... first L frequency components`，随后定义 noisy DCT spectrum 与噪声预测损失 Eq. 3–4；Sec. 4.2 为 DCT-Completion | PASS |
| C2 | InterDiff 将接触参考系校正与扩散组合 | `interdiff` PDF p.3，Sec. 3.1 的 interaction diffusion；p.5，Sec. 3.2.1，L383 起：无接触保留 ground reference，接触时选身体表面 reference；校正再用于去噪 | PASS |
| C3 | InterDiff 的图预测器使用 DCT/IDCT | `interdiff` PDF p.5，Sec. 3.2.2 的 spatial-temporal graph 与 STGNN；p.6，L428 起：`in this STGNN, we use DCT/IDCT ... as a preprocessing step`，并明确 clean HOI 训练、denoised HOI 推理 | PASS |
| C4 | InterDiff 承认 skeleton contact 定义不充分 | `interdiff` PDF p.5，Sec. 3.2.1，L371：`contact in skeletal HOIs is ill-defined`；原文仍使用关节距离选代理参考系，但 mesh 接触/穿透条件不直接套给骨架 | PASS |
| C5 | SyncDiff 在动态图中显式同步多体扩散 | `syncdiff` PDF p.1，L28，摘要写 `dynamic graphical model` 的 maximum likelihood sampling；Sec. 3.5–3.6 描述 alignment 与显式同步 | PASS |
| C6 | SyncDiff 用 FFT 频域分解 | `syncdiff` PDF p.4，Sec. 3.3，L214 起：`Fast Fourier Transform (FFT)`，分开 low-frequency 与 high-frequency 系数/轨迹，并在 Sec. 3.4 的 diffusion 中建模 | PASS；不能写作 DCT |
| C7 | POMP 是相位条件化扩散 | `pomp` PDF p.4，Sec. 3.2 为 phase 编码与 Ortho-MoE；p.5，L338 起：`POMP executes the diffusion process within the phase domain, conditioned on the motion phase embedding w`，后续有扩散与条件噪声网络公式 | PASS |
| C8 | SyncMos 显式表示事件依赖 | `syncmos` PDF p.3，Sec. 4.2，L245 起为 `structured event dependency graph`，R 分为 sequential 与 parallel；L264 起列出 prerequisite/同步示例 | PASS |
| C9 | SyncMos 用 time-warping 与 DPS 修正时序 | `syncmos` PDF p.5，Sec. 5.2，L432 起的 Algorithm 2 与正文：time-warped trajectory 作为时序观测，以 constraint-guided denoising 保持生成先验 | PASS |
| C10 | JRT 同时预测关节关系距离 | `jrt` PDF p.1，L28 起：`supervise the relation information by forecasting future distance`；正文 Sec. 4.2 与 4.4 定义 relation-aware 融合与训练目标 | PASS |

三条方法事实还得到不同作者的原论文相关工作段支持：C1 由 `nonisotropic.pdf` p.2 明确描述 HumanMAC 在 DCT 时域表示中扩散；C2 由 `comanipulation.pdf` p.2 描述 InterDiff 把 contact objectives 注入 diffusion；C5 由同篇 p.2 描述 SyncDiff 的多体同步。交叉核对率为 3/10 = 30%。这是独立作者对方法内容的佐证，未构成独立性能复现；InterDiff 的 DCT/STGNN 细节、SyncDiff 的 FFT 细节及各算法公式仍以相应原文为证据。

以下外推未通过，不能写进结论：

| ID | 外推论断 | 结果与依据 |
|---|---|---|
| F1 | DCT + diffusion 作为动作预测概念此前无人做过 | FAIL：C1 提供反例 |
| F2 | DCT/图网络/接触/扩散这四项首次组合 | FAIL：C2–C3 提供同一已发表工作的反例；C5–C6 另有更近的频域图扩散先例 |
| F3 | 加入图或相位扩散即可保证当前 NTU 的真实接触因果与规律迈步 | FAIL：原文没有在当前 obs10/future50/受试者留出协议上验证此保证；C4 明确限制 skeleton contact 的含义 |
| F4 | 这些论文的成绩已经证明本次项目改造提高 paired MPJPE | FAIL：任务、数据、观测条件与指标不同，且本次验证没有训练或复现本项目新模型 |

## 来源质量、时效与利益关联

证据层级按技术领域的“有对照的计算方法/基准实验”记为 Level III，不把它们误当作人体物理因果的随机对照试验。下表评级针对文献身份与论文明确采用的机制：A 表示可作为原始方法证据；没有给跨协议性能保证评级。

| 来源 | Level | 出版/作者身份 | 方法事实 | 时效 | 原文披露的机构关联 | 适用评级 |
|---|---|---|---|---|---|---|
| HumanMAC | III | PASS | PASS | 2023，DCT 扩散先例 | 清华/西电/南洋理工/悉尼大学；本轮未见专门 COI 声明 | A（方法事实） |
| ReGenNet | III | PASS | PASS | 2024 | 原文署名 WeChat/Tencent；另有学术机构与资助，企业关联已披露 | A（方法事实） |
| InterGen | III | PASS | PASS | 2024 | ShanghaiTech，未从缺少 COI 字段推断完全无利益关联 | A（方法事实） |
| JRT | III | PASS | PASS | 2023，显式关系先例 | 上海交大/上海 AI 实验室 | A（方法事实） |
| InterDiff | III | PASS | PASS | 2023，接触/DCT 图校正先例 | UIUC | A（方法事实） |
| SyncDiff | III | PASS | PASS | 2025 | 清华/上海期智/上海 AI 实验室 | A（方法事实） |
| POMP | III | PASS，标题差异有记录 | PASS | 2025 | 上海交大/浙江大学；正文有公共项目资助 | A（方法事实） |
| TIMotion | III | PASS | PASS | 2025 | Youtu Lab/Tencent 与浙江大学/上海交大，企业关联已披露 | A（方法事实） |
| Interact2Ar | III | PASS | PASS | 2026 | Huawei Noah’s Ark Lab 与高校，企业关联已披露；正文列公共项目资助 | A（方法事实） |
| SyncMos | III | PASS | PASS | 2026 | 墨尔本大学/汉阳大学；正文列公共项目资助 | A（方法事实） |

机构关联本身不证明财务冲突或结果偏差。作者姓名已核对，但未逐人核验 ORCID、完整履历或私人财务关系；相关未核验内容不能写成“均无 COI”。

出版检查：9 项具备 CVF 官方会议元数据与 accepted-version PDF 声明，InterGen 具备 Springer 正式论文与 DOI 注册。未发现来源身份层面的掠夺性出版信号；未使用 Cabell’s 付费名单、Scopus/WoS 全库或全集撤稿库，因此不出具这些数据库上的排除性认证。

## 更正、局限与 AI 披露

对先前注释书目的定位更正以本新文件记录，历史文件不修改：InterDiff 的 DCT/STGNN 段是 PDF p.6（印刷页 14933），而非先前笼统写的 pp.4–5；SyncDiff 的显式同步在 Sec. 3.6，先前给出的 Sec. 3.2–3.5 未覆盖此节。

本次是职责切换后的原文证据复核，书目与验证由同一代理执行，并非另一位人员或另一模型的独立审查。所有通过项都重新读取了官方原文/元数据；摘要与以前的解释未充当证据。未复现论文代码、公开数字或统计功效；未穷尽所有后续论文；不能据此证明具体新方案无人做过。

AI 参与了官方页面获取、元数据核对、PDF 文本定位与首页面像检查。`source_verified_against_original` 所表达的 AI 核对不能替代用户人工阅读，也不代表已完成独立复现实验。本阶段结束后由调用方继续模型设计与验证。
