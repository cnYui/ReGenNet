"""检索锚点（设计代号 A3）：把同动作检索邻居的未来位移融进冻结 base 的预测，并把邻居编码成 exemplar token。

全部张量处在查询自己的场景规范系（`utils/ntu2p_canonical`）；邻居位移 disp 在邻居自己的规范系中，
两个规范系的定义相同（原点 = 末帧双人 pelvis 中点，+X = A->B，+Y = 上），因此可以直接相加。

anchor = base + ramp · (β_root ⊙ D_root + β_local ⊙ D_local)，其中 D = Δ_knn - Δ_base：
- 分 root / 局部两个通道：root 通道 = 所有关节共同的平移（pelvis 位移），局部通道 = 相对 pelvis 的部分。
  R2 分析显示 kNN 与学习模型在 root 上互补最明显，而把局部姿态平均会压低摆动幅度，
  所以 root β 初始化 0.5、局部 β 初始化 0（局部完全沿用 base，保护已恢复的摆动）。
- β 用线性参数化（逐未来帧 × 人物）：局部通道需要精确为 0 的初值，sigmoid 做不到；
  线性参数不设上下界，允许学到外推（β > 1）或抑制（β < 0），首帧连续性由 ramp[0] = 0 单独保证。
- 邻居权重 w = softmax(-dist/τ + s)：τ 可学习、以 train 查询的 top-k 距离中位数初始化；打分 MLP 末层零初始化，
  因此初始时就是按距离的软平均，训练只需学习"偏离距离排序"的部分。
"""

import math

import torch
from torch import nn

from data_loaders.forecasting.ntu2p_retrieval_bank import DEFAULT_KEY_JOINTS, NUM_TIERS
from utils.ntu_smplx_2p_xyz import NTU_NUM_PERSONS, NTU_SMPLX_BODY_JOINTS, XYZ_COORD_DIM, dct_matrix


PELVIS = 0
# 与 build_ntu2p_retrieval_bank.py 中 train 查询（去重、排除同受试者与同序列）K=16 的距离中位数同量级；
# 实际使用时应传入检索库 config["median_topk_dist"]。
DEFAULT_INIT_TAU = 0.074


def _ramp_view(ramp, pred_len):
    """[T] / [B,T] / [B|1,T,1,1,1] -> [B|1,T,1,1,1]。"""
    ramp = torch.as_tensor(ramp)
    if ramp.dim() == 1:
        ramp = ramp.view(1, -1)
    if ramp.dim() == 2:
        ramp = ramp.view(ramp.shape[0], ramp.shape[1], 1, 1, 1)
    if ramp.dim() != 5 or int(ramp.shape[1]) != int(pred_len):
        raise ValueError("ramp 必须是 [T]、[B,T] 或 [B,T,1,1,1] 且 T={}，当前为 {}".format(pred_len, tuple(ramp.shape)))
    return ramp


class RetrievalAnchor(nn.Module):
    def __init__(
        self,
        pred_len=50,
        obs_len=10,
        num_joints=NTU_SMPLX_BODY_JOINTS,
        key_joints=DEFAULT_KEY_JOINTS,
        d_model=256,
        num_coeffs=15,
        max_neighbors=64,
        init_tau=DEFAULT_INIT_TAU,
        root_beta_init=0.5,
        local_beta_init=0.0,
        num_tiers=NUM_TIERS,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.obs_len = int(obs_len)
        self.num_joints = int(num_joints)
        self.key_joints = int(key_joints)
        self.d_model = int(d_model)
        self.num_coeffs = int(num_coeffs)
        self.max_neighbors = int(max_neighbors)
        if float(init_tau) <= 0.0:
            raise ValueError("init_tau 必须 > 0")
        # 低阶 DCT：前 15 阶覆盖 <= 2.8 Hz，包含步态频段，去掉拟合抖动，token 输入维度从 8250 压到 2475。
        self.register_buffer("dct_basis", dct_matrix(self.pred_len)[: self.num_coeffs].t().contiguous())
        # 距离 embedding 用固定尺度归一化，避免与可学习 τ 耦合。
        self.register_buffer("tau_ref", torch.tensor(float(init_tau)))
        self.log_tau = nn.Parameter(torch.tensor(math.log(float(init_tau))))
        self.disp_proj = nn.Linear(self.num_joints * XYZ_COORD_DIM * self.num_coeffs, self.d_model)
        self.query_proj = nn.Linear(self.obs_len * NTU_NUM_PERSONS * self.key_joints * XYZ_COORD_DIM, self.d_model)
        self.dist_proj = nn.Linear(1, self.d_model)
        self.rank_embedding = nn.Embedding(self.max_neighbors, self.d_model)
        self.person_embedding = nn.Embedding(NTU_NUM_PERSONS, self.d_model)
        self.tier_embedding = nn.Embedding(int(num_tiers), self.d_model)
        for embedding in (self.rank_embedding, self.person_embedding, self.tier_embedding):
            nn.init.normal_(embedding.weight, std=0.02)
        self.score_mlp = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )
        nn.init.zeros_(self.score_mlp[-1].weight)
        nn.init.zeros_(self.score_mlp[-1].bias)
        self.beta_root = nn.Parameter(torch.full((self.pred_len, NTU_NUM_PERSONS), float(root_beta_init)))
        self.beta_local = nn.Parameter(torch.full((self.pred_len, NTU_NUM_PERSONS), float(local_beta_init)))

    @property
    def tau(self):
        return self.log_tau.exp()

    def _dct_code(self, displacement):
        """[...,T,2,J,3] -> [...,2,J*3*C]：每人一段低阶 DCT 系数。"""
        lead = displacement.shape[:-4]
        value = displacement.reshape(lead + (self.pred_len, NTU_NUM_PERSONS, self.num_joints * XYZ_COORD_DIM))
        value = value.transpose(-3, -2).transpose(-2, -1)  # [...,2,J*3,T]
        coeff = torch.matmul(value, self.dct_basis)  # [...,2,J*3,C]
        return coeff.reshape(lead + (NTU_NUM_PERSONS, -1))

    def forward(self, obs_canon, base_canon, disp, dist, ramp, tier=None):
        """obs_canon [B,obs_len,2,J,3]、base_canon [B,T,2,J,3]、disp [B,K,T,2,J,3]、dist [B,K]、
        ramp（首帧必须为 0）、tier [B,K] 可选。返回 anchor [B,T,2,J,3]、tokens [2K,B,d]（第 2k+p 个是
        第 k 名邻居的人物 p）、weights [B,K]、delta_knn [B,T,2,J,3]。"""
        batch_size, num_neighbors = int(dist.shape[0]), int(dist.shape[1])
        if num_neighbors > self.max_neighbors:
            raise ValueError("邻居数 {} 超过 max_neighbors={}".format(num_neighbors, self.max_neighbors))
        if tuple(disp.shape[:3]) != (batch_size, num_neighbors, self.pred_len):
            raise ValueError("disp 必须是 [B,K,T,2,J,3]，当前为 {}".format(tuple(disp.shape)))
        obs_last = obs_canon[:, -1:]
        delta_base = base_canon - obs_last

        content = self.disp_proj(self._dct_code(disp))  # [B,K,2,d]
        rank = torch.arange(num_neighbors, device=dist.device)
        meta = self.dist_proj((dist / self.tau_ref).unsqueeze(-1)) + self.rank_embedding(rank).unsqueeze(0)
        if tier is not None:
            meta = meta + self.tier_embedding(tier.to(device=dist.device, dtype=torch.long))
        neighbor = content.mean(dim=2) + meta
        query = self.query_proj(obs_canon[:, :, :, : self.key_joints].reshape(batch_size, -1))
        query = query + self.disp_proj(self._dct_code(delta_base)).mean(dim=1)
        score = self.score_mlp(torch.cat((query.unsqueeze(1).expand(-1, num_neighbors, -1), neighbor), dim=-1))
        weights = torch.softmax(score.squeeze(-1) - dist / self.tau, dim=1)

        # 逐元素加权求和而非 einsum：Ampere 上 einsum 走 TF32，位移会有毫米级误差。
        delta_knn = (weights.view(batch_size, num_neighbors, 1, 1, 1, 1) * disp).sum(dim=1)
        gap = delta_knn - delta_base
        gap_root = gap[:, :, :, PELVIS : PELVIS + 1]
        beta_root = self.beta_root.view(1, self.pred_len, NTU_NUM_PERSONS, 1, 1)
        beta_local = self.beta_local.view(1, self.pred_len, NTU_NUM_PERSONS, 1, 1)
        correction = beta_root * gap_root + beta_local * (gap - gap_root)
        # 以 base 为起点相加（而不是 obs_last + Δ_base），β = 0 时输出与 base 逐位相同。
        anchor = base_canon + _ramp_view(ramp, self.pred_len).to(correction) * correction

        tokens = content + meta.unsqueeze(2) + self.person_embedding.weight.view(1, 1, NTU_NUM_PERSONS, -1)
        tokens = tokens.reshape(batch_size, num_neighbors * NTU_NUM_PERSONS, self.d_model).transpose(0, 1)
        return dict(anchor=anchor, tokens=tokens, weights=weights, delta_knn=delta_knn)
