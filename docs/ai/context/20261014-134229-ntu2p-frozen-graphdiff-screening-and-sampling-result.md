# 冻结底座图扩散：完整筛选与随机采样结果

## 决定

C1–C4 全部完成 CUDA seed0×5000、EMA终点、subject-holdout val247窗评估，**均不采纳**。图、腿专用流与扩散的增量没有超过时间纠正器。C4 的随机 DDIM 同样不采纳：概率评分不如 OOF 真残差抽样，jerk 与穿地明显增加。原主线 v2+FD2 保留。

设计与公平性：`20261014-112117-ntu2p-contact-graph-diffusion-design-and-plan.md`、`20261014-111722-ntu2p-contact-graph-diffusion-preregistration-amendment.md`。正式接触六段口径及早期审计差异见 `20261014-115547-ntu2p-graphdiff-evaluation-review-fixes-result.md`。以下数字只在同 H-val 口径比较，不能与既有 test 0.1805±0.0005拼接。

## 确定性终点

| 模型 | xyz_mse | xyz_mae | mpjpe | Δmpjpe | 滑行帧比 | 步数 | 推人距离RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| FD2参照 | .019305 | .084198 | .172228 | — | .2004 | 1.4589 | .202690 |
| C1 时间纠正 | .019103 | .083736 | .171175 | −.61% | .1988 | 1.4589 | .203592 |
| C2 +图 | .019114 | .083756 | .171238 | −.57% | .1992 | 1.4589 | .204057 |
| C3 +腿流 | .019113 | .083759 | .171238 | −.57% | .1985 | 1.4658 | .204230 |
| C4 +扩散辅助 | .019114 | .083747 | .171228 | −.58% | .1991 | 1.4658 | .204480 |

均通过原三项L2/摆动gate与本轮自然度护栏，但没有达到mpjpe改善≥.8%或定向改善≥10%的采纳线。起步者前0.5s相位相关：FD2 .3277，C1–C4为.3280/.3248/.3211/.3204；已行走者约.909→.911，变化很小。不能声称左右腿起步或接触顺序已改善。

对15个留出受试者做3000次簇配对bootstrap（seed2718），C1–C4的Δmpjpe95%诊断区间分别为[−.750,−.492]%、[−.694,−.452]%、[−.692,−.448]%、[−.696,−.450]%。这只描述同一训练seed的验证样本敏感性，不替代训练3seed，也未改变采纳规则。

完整输出：`results/forecasting/ntu120_label/ntu2p_graphdiff/subjval/screen5000/summary.{json,md}`。原权重：`save/forecasting/ntu120_label/ntu2p_graphdiff_subjval_C{1,2,3,4}_s0_5000/ema/model000005000.pt`。每组496867个新参数，双纠正器forward/step，batch8、lr3e-4、EMA.999，单3080并发两组。训练日志记录实际用时与CUDA，未用test。

## 标准随机 DDIM 与 bootstrap

按 `20261014-113541-ntu2p-graphdiff-sampling-evaluation-plan.md`，C4 EMA5000、全247窗、K10、seed2718、eta0、NFE20/50，实际范围标记is_smoke=False。OOF库核验通过，旧K12/saturate残差解码后投到共同smoothstepK16基。表中L2、jerk、穿地为固定sample0，ES对K10计算。

| 分布 | 单样本mpjpe | ES↓ | APD | jerk↓ | 穿地帧比↓ | 步数 |
|---|---:|---:|---:|---:|---:|---:|
| 原FD2点预测 | .172241 | — | — | 20.44 | .0640 | 1.459 |
| C4确定性端点 | .171239 | — | — | 20.03 | .0686 | 1.466 |
| DDIM20 | .203453 | .152648 | .098675 | 49.69 | .1538 | 1.582 |
| DDIM50（主采样） | .211314 | .155616 | .108828 | 54.05 | .1535 | 1.644 |
| OOF bootstrap×.5 | .192928 | .135313 | .117995 | 27.98 | .0937 | 1.671 |
| OOF bootstrap×1 | .243353 | .128381 | .232398 | 40.55 | .1340 | 1.890 |

采样脚本batch4，确定性标准脚本batch16，数值有约1e−5的矩阵批处理差异；各采样分布在同脚本、同batch下比较。前三固定样本都报告自然度与接触；DDIM50的jerk约53.5–54.1、穿地.150–.154，不能将sample0当偶发异常。部分滑行帧比下降并不足以抵消抖动/穿地与paired误差退化。

ES是逐帧逐关节xyz边际Energy Score，不能独立证明联合时序概率质量。生成器仍用in-sample残差，bootstrap用OOF真残差；本轮随机模式仅探索，未通过，也未在test评价。minADE是GT选样上界，不作部署主数字。产物：同目录`sampling_C4.json`、`sampling_C4_arrays.pt`，完整评价约268s。

## 视觉与根因诊断

GT预选18例（6推人/6最大GT位移/6随机），另固定seed随机6起步（去重后共23例），全部12推人曲线；名单先于C3/C4终点评价固定，未挑改善最大案例。

`analysis/all_pushing_contact_retreat.png`显示FD2与四候选曲线基本重合；例如020、028、123仍有预测B root速度过早抬升。`review_c1c4/idx_002_S001C001P004R001A003.png`中GT后期双腿明显换步，预测仍为缓慢脚位漂移，四候选无可辨别修复。这些是按固定名单检查到的负例，不能用均值L2收益替代。

val残差系数RMS为.12829m，train估计sigma RMS为.09569m，归一化val残差RMS1.431；推人子集1.043。表明分布偏移仍存在，但不能单凭这个统计判定它就是全部失败原因。先前编码GT残差oracle .03825仅是表达容量，不能作预测数字。

最小事件root积分器可行性见`20261014-120042-ntu2p-event-controller-feasibility-result.md`：合成先触后新增响应保证成立，真实FD2轨迹固定.20门却有3/12推人不接触，推人子集mpjpe+6.1%、B rootRMSE+30.9%。因此没有把硬门事后包装为成功优化，也没有通过自由路线排除失败样本。

## 后续已登记工作

按`20261014-120112-ntu2p-end-to-end-graphdiff-followup-design-and-plan.md`，只进一步检验端到端F1–F3：原底座可训练但另存冻结FD2 reference，公平比较相同监督/训练的时间头、图+腿流、扩散辅助。原损失/阈值/采纳判据不变。本文件不含尚未完成的F成绩或采纳承诺。

可复用分析工具：`scripts/analyze_ntu2p_graphdiff_results.py`。它核验同seed数据/manifest/reference，一次合并候选数组、输出受试者簇bootstrap与全推人曲线；拒绝已有输出目录。图使用本机CJK字体，避免中文缺字；旧缺字试绘保留，但正式审查引用`analysis/`内修正版。
