# 图/相位/扩散实现与冒烟验证

实现依据：`20261014-112117-ntu2p-contact-graph-diffusion-design-and-plan.md`、`20261014-111722-ntu2p-contact-graph-diffusion-preregistration-amendment.md`；审查：`20261014-111804-ntu2p-contact-graph-diffusion-design-review.md`。

## 已实现

- `model/forecasting_ntu2p_graphdiff.py`：冻结 v2+FD2 + body22逐帧类型图、因果前缀接触记忆、低噪声更新图、双腿专用纠正流；确定性端点与原 ReGenNet DDIM 分开接口。
- `utils/ntu2p_contact_graph.py`：掌腕—六段身体距离、双向图边、接触/记忆/接近速度、连续距离与有符号左右踝分离损失、固定四距离/二速度阈值事件评价。
- `utils/ntu2p_residual_codec.py`：兼容旧默认的 smoothstep 斜坡基扩展，K16、8帧，新配置完整记录。
- `train/train_ntu2p_graphdiff.py`：两次纠正器 forward，同主loss/归一化/采样/EMA/初始化；C1–C3 clean重建控制，C4 q_sample去噪。所有正式训练强制cuda:0。
- `scripts/run_ntu2p_graphdiff.py`：单卡双并发、终点自动评价、恢复已完成run、不覆盖历史；test要求冻结H-val采纳与审查记录。
- generic checkpoint分派与v2评估扩展：保存全部先验，推理无需外部prior；新模型同步报告prior自然度与contact，旧类型无需contact路径。

## 验证证据

结构验证脚本 `scripts/check_ntu2p_graphdiff.py`：C1–C4初始state_dict相同；零输出与先验最大差7.3e-7 m；K16smoothstep基Gram条件数8.47，encode(decode(c))误差1.5e-8；首帧严格为0；图/腿/观测梯度有限且非零、对方节点能通过指定接触边影响本方节点、记忆无未来依赖；checkpoint精确复载、严格CUDA策略。采样初次发现旧GaussianDiffusion要求model_kwargs['y']，已补适配并重跑采样检查通过。

4窗oracle诊断：prior mpjpe 0.129→编码GT残差后0.0304，前10帧0.0499→0.0158。该小样本仅验证表达能力，不用于选包络。smoothstep只保证连续基函数起点导数零，**不保证第二个离散帧的纠正为零**，最终离散速度/jerk仍须实测。

几何合成验证 `scripts/check_ntu2p_contact_graph.py`：解析点段距离、刚体变换不变、prefix无未来梯度、双方梯度有限、空子集及严格JSON通过。

CUDA C4 100step冒烟：496867可训练参数，双forward，12.18 s（约0.122s/step），loss/梯度全有限；EMA16窗评价mpjpe0.118596，首帧0，摆动gate通过。此分数不是正式结果，不据此采纳。

## 训练前冻结

将按原预登记 C1–C4 seed0×5000step 在subject-holdout val筛选，双进程并发。源文献已取得，独立来源核验进行中；已知DCT/图/扩散/相位/事件都有近邻，最终不宣称无人做过。随机采样只为探索，若候选确定性不通过则不做test。

新接触评价使用六段与掌/腕两点取min，和最初数据审计的五段掌心均值不同，参照与候选必须全部按新函数重算；不可拼接旧0.2024m数字作改进百分比。
