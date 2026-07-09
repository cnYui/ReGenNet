import torch
from torch import nn
from torch.nn import functional as F

from utils.ntu_smplx_2p_xyz import (
    NTU_DEFAULT_CONTACT_THRESHOLD,
    acceleration_with_last_obs,
    NTU_NUM_PERSONS,
    NTU_SMPLX_BODY_JOINTS,
    XYZ_COORD_DIM,
    check_ntu_xyz,
    horizon_slices,
    interaction_pair_distances,
    local_pose,
    masked_contact_mse,
    relative_root,
    relative_root_velocity,
    root_positions,
    velocity_with_last_obs,
)


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _normalize_action(action, batch_size, num_actions, device):
    if action is None:
        raise ValueError("action 不能为空")
    if not torch.is_tensor(action):
        action = torch.as_tensor(action)
    if action.dim() == 2 and action.shape[1] == 1:
        action = action[:, 0]
    if action.dim() != 1:
        raise ValueError("action 必须是 [B] 或 [B,1]，当前为 {}".format(tuple(action.shape)))
    if int(action.shape[0]) != int(batch_size):
        raise ValueError("action batch 必须是 {}，当前为 {}".format(int(batch_size), int(action.shape[0])))
    action = action.to(device=device, dtype=torch.long)
    if int(action.min().item()) < 0 or int(action.max().item()) >= int(num_actions):
        raise ValueError("action 必须在 [0,{}] 内".format(int(num_actions) - 1))
    return action


class NTULabelXYZTransformer(nn.Module):
    def __init__(
        self,
        obs_len=20,
        pred_len=40,
        num_actions=26,
        num_persons=NTU_NUM_PERSONS,
        num_joints=NTU_SMPLX_BODY_JOINTS,
        coord_dim=XYZ_COORD_DIM,
        latent_dim=256,
        num_heads=4,
        encoder_layers=3,
        decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        velocity_loss_weight=0.2,
        continuity_loss_weight=0.0,
        first_step_loss_weight=0.0,
        mae_loss_weight=0.1,
        root_loss_weight=1.0,
        local_pose_loss_weight=1.0,
        mpjpe_loss_weight=0.0,
        short_loss_weight=0.0,
        mid_loss_weight=0.0,
        long_loss_weight=0.2,
        final_frame_loss_weight=0.2,
        acceleration_loss_weight=0.1,
        relative_root_loss_weight=0.2,
        relative_velocity_loss_weight=0.1,
        key_joint_relation_loss_weight=0.0,
        contact_loss_weight=0.0,
        contact_threshold=NTU_DEFAULT_CONTACT_THRESHOLD,
        action_feature_loss_weight=0.0,
        action_logit_loss_weight=0.0,
    ):
        super(NTULabelXYZTransformer, self).__init__()
        self.model_type = "ntu_label_xyz_transformer"
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.num_actions = int(num_actions)
        self.num_persons = int(num_persons)
        self.num_joints = int(num_joints)
        self.coord_dim = int(coord_dim)
        self.latent_dim = int(latent_dim)
        self.num_heads = int(num_heads)
        self.encoder_layers = int(encoder_layers)
        self.decoder_layers = int(decoder_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.velocity_loss_weight = float(velocity_loss_weight)
        self.continuity_loss_weight = float(continuity_loss_weight)
        self.first_step_loss_weight = float(first_step_loss_weight)
        self.mae_loss_weight = float(mae_loss_weight)
        self.root_loss_weight = float(root_loss_weight)
        self.local_pose_loss_weight = float(local_pose_loss_weight)
        self.mpjpe_loss_weight = float(mpjpe_loss_weight)
        self.short_loss_weight = float(short_loss_weight)
        self.mid_loss_weight = float(mid_loss_weight)
        self.long_loss_weight = float(long_loss_weight)
        self.final_frame_loss_weight = float(final_frame_loss_weight)
        self.acceleration_loss_weight = float(acceleration_loss_weight)
        self.relative_root_loss_weight = float(relative_root_loss_weight)
        self.relative_velocity_loss_weight = float(relative_velocity_loss_weight)
        self.key_joint_relation_loss_weight = float(key_joint_relation_loss_weight)
        self.contact_loss_weight = float(contact_loss_weight)
        self.contact_threshold = float(contact_threshold)
        self.action_feature_loss_weight = float(action_feature_loss_weight)
        self.action_logit_loss_weight = float(action_logit_loss_weight)
        self.frame_dim = self.num_persons * self.num_joints * self.coord_dim

        self.input_proj = nn.Linear(self.frame_dim, self.latent_dim)
        self.output_proj = nn.Linear(self.latent_dim, self.frame_dim)
        self.obs_pos = nn.Parameter(torch.zeros(1, self.obs_len, self.latent_dim))
        self.future_pos = nn.Parameter(torch.zeros(1, self.pred_len, self.latent_dim))
        self.action_embed = nn.Embedding(self.num_actions, self.latent_dim)
        self.action_type = nn.Parameter(torch.zeros(1, 1, self.latent_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.decoder_layers)
        self.norm = nn.LayerNorm(self.latent_dim)

        # 初始输出严格等价于 copy-last，后续训练只学习相对最后一帧的位移。
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def config(self):
        return {
            "model_type": self.model_type,
            "obs_len": self.obs_len,
            "pred_len": self.pred_len,
            "num_actions": self.num_actions,
            "num_persons": self.num_persons,
            "num_joints": self.num_joints,
            "coord_dim": self.coord_dim,
            "latent_dim": self.latent_dim,
            "num_heads": self.num_heads,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "velocity_loss_weight": self.velocity_loss_weight,
            "continuity_loss_weight": self.continuity_loss_weight,
            "first_step_loss_weight": self.first_step_loss_weight,
            "mae_loss_weight": self.mae_loss_weight,
            "root_loss_weight": self.root_loss_weight,
            "local_pose_loss_weight": self.local_pose_loss_weight,
            "mpjpe_loss_weight": self.mpjpe_loss_weight,
            "short_loss_weight": self.short_loss_weight,
            "mid_loss_weight": self.mid_loss_weight,
            "long_loss_weight": self.long_loss_weight,
            "final_frame_loss_weight": self.final_frame_loss_weight,
            "acceleration_loss_weight": self.acceleration_loss_weight,
            "relative_root_loss_weight": self.relative_root_loss_weight,
            "relative_velocity_loss_weight": self.relative_velocity_loss_weight,
            "key_joint_relation_loss_weight": self.key_joint_relation_loss_weight,
            "contact_loss_weight": self.contact_loss_weight,
            "contact_threshold": self.contact_threshold,
            "action_feature_loss_weight": self.action_feature_loss_weight,
            "action_logit_loss_weight": self.action_logit_loss_weight,
        }

    def forward(self, obs_xyz, action):
        check_ntu_xyz("obs_xyz", obs_xyz, seq_len=self.obs_len)
        batch_size = int(obs_xyz.shape[0])
        action = _normalize_action(action, batch_size, self.num_actions, obs_xyz.device)

        obs_flat = obs_xyz.reshape(batch_size, self.obs_len, self.frame_dim)
        obs_tokens = self.input_proj(obs_flat) + self.obs_pos
        action_token = self.action_embed(action).unsqueeze(1) + self.action_type
        memory = torch.cat((action_token, obs_tokens), dim=1).transpose(0, 1).contiguous()
        memory = self.encoder(memory)

        query = self.future_pos.expand(batch_size, -1, -1)
        query = query + self.action_embed(action).unsqueeze(1)
        query = query.transpose(0, 1).contiguous()
        hidden = self.decoder(query, memory).transpose(0, 1).contiguous()
        delta = self.output_proj(self.norm(hidden))
        delta = delta.view(
            batch_size,
            self.pred_len,
            self.num_persons,
            self.num_joints,
            self.coord_dim,
        )
        ramp = torch.linspace(
            0.0,
            1.0,
            self.pred_len,
            device=delta.device,
            dtype=delta.dtype,
        ).view(1, self.pred_len, 1, 1, 1)
        delta = delta * ramp
        pred = obs_xyz[:, -1:].expand(-1, self.pred_len, -1, -1, -1) + delta
        check_ntu_xyz("pred_xyz", pred, seq_len=self.pred_len)
        return pred

    def training_loss(self, obs_xyz, target_xyz, action, action_classifier=None, action_normalizer=None):
        check_ntu_xyz("target_xyz", target_xyz, seq_len=self.pred_len)
        pred = self.forward(obs_xyz, action)
        loss = F.mse_loss(pred, target_xyz)
        if self.mae_loss_weight > 0:
            loss = loss + self.mae_loss_weight * F.l1_loss(pred, target_xyz)
        if self.root_loss_weight > 0:
            loss = loss + self.root_loss_weight * F.mse_loss(root_positions(pred), root_positions(target_xyz))
        if self.local_pose_loss_weight > 0:
            loss = loss + self.local_pose_loss_weight * F.mse_loss(local_pose(pred), local_pose(target_xyz))
        if self.mpjpe_loss_weight > 0:
            loss = loss + self.mpjpe_loss_weight * torch.norm(pred - target_xyz, dim=-1).mean()
        short_slice, mid_slice, long_slice = horizon_slices(self.pred_len)
        if self.short_loss_weight > 0:
            loss = loss + self.short_loss_weight * F.mse_loss(pred[:, short_slice], target_xyz[:, short_slice])
        if self.mid_loss_weight > 0:
            loss = loss + self.mid_loss_weight * F.mse_loss(pred[:, mid_slice], target_xyz[:, mid_slice])
        if self.long_loss_weight > 0:
            loss = loss + self.long_loss_weight * F.mse_loss(pred[:, long_slice], target_xyz[:, long_slice])
        if self.final_frame_loss_weight > 0:
            loss = loss + self.final_frame_loss_weight * F.mse_loss(pred[:, -1], target_xyz[:, -1])
        if self.velocity_loss_weight > 0:
            loss = loss + self.velocity_loss_weight * F.mse_loss(
                velocity_with_last_obs(pred, obs_xyz),
                velocity_with_last_obs(target_xyz, obs_xyz),
            )
        if self.acceleration_loss_weight > 0:
            loss = loss + self.acceleration_loss_weight * F.mse_loss(
                acceleration_with_last_obs(pred, obs_xyz),
                acceleration_with_last_obs(target_xyz, obs_xyz),
            )
        if self.continuity_loss_weight > 0:
            loss = loss + self.continuity_loss_weight * F.mse_loss(pred[:, 0], obs_xyz[:, -1])
        if self.first_step_loss_weight > 0:
            loss = loss + self.first_step_loss_weight * F.mse_loss(pred[:, 0], target_xyz[:, 0])
        if self.relative_root_loss_weight > 0:
            loss = loss + self.relative_root_loss_weight * F.mse_loss(
                torch.norm(relative_root(pred), dim=-1),
                torch.norm(relative_root(target_xyz), dim=-1),
            )
        if self.relative_velocity_loss_weight > 0:
            pred_full = torch.cat((obs_xyz[:, -1:], pred), dim=1)
            target_full = torch.cat((obs_xyz[:, -1:], target_xyz), dim=1)
            loss = loss + self.relative_velocity_loss_weight * F.mse_loss(
                relative_root_velocity(pred_full),
                relative_root_velocity(target_full),
            )
        if self.key_joint_relation_loss_weight > 0 or self.contact_loss_weight > 0:
            pred_pair_dist = interaction_pair_distances(pred)
            target_pair_dist = interaction_pair_distances(target_xyz)
            if self.key_joint_relation_loss_weight > 0:
                loss = loss + self.key_joint_relation_loss_weight * F.mse_loss(pred_pair_dist, target_pair_dist)
            if self.contact_loss_weight > 0:
                contact_loss, _ = masked_contact_mse(pred_pair_dist, target_pair_dist, self.contact_threshold)
                loss = loss + self.contact_loss_weight * contact_loss
        if self.action_feature_loss_weight > 0 or self.action_logit_loss_weight > 0:
            if action_classifier is None or action_normalizer is None:
                raise ValueError("启用 action loss 时必须提供冻结 action_classifier 和 action_normalizer")
            from eval.action_xyz_classifier import extract_xyz_action_features

            pred_logits, pred_features = extract_xyz_action_features(action_classifier, pred, action_normalizer)
            if self.action_feature_loss_weight > 0:
                _, target_features = extract_xyz_action_features(action_classifier, target_xyz, action_normalizer)
                loss = loss + self.action_feature_loss_weight * F.mse_loss(pred_features, target_features.detach())
            if self.action_logit_loss_weight > 0:
                action_label = _normalize_action(action, int(obs_xyz.shape[0]), self.num_actions, obs_xyz.device)
                loss = loss + self.action_logit_loss_weight * F.cross_entropy(pred_logits, action_label)
        if not torch.isfinite(loss):
            raise ValueError("ntu_label_xyz_transformer loss 为非有限数值")
        return loss


def create_ntu_label_xyz_model_from_config(config):
    model_type = config.get("model_type")
    if model_type != "ntu_label_xyz_transformer":
        raise ValueError("unsupported NTU xyz model_type: {}".format(model_type))
    return NTULabelXYZTransformer(
        obs_len=config.get("obs_len", 20),
        pred_len=config.get("pred_len", 40),
        num_actions=config.get("num_actions", 26),
        num_persons=config.get("num_persons", NTU_NUM_PERSONS),
        num_joints=config.get("num_joints", NTU_SMPLX_BODY_JOINTS),
        coord_dim=config.get("coord_dim", XYZ_COORD_DIM),
        latent_dim=config.get("latent_dim", 256),
        num_heads=config.get("num_heads", 4),
        encoder_layers=config.get("encoder_layers", 3),
        decoder_layers=config.get("decoder_layers", 3),
        dim_feedforward=config.get("dim_feedforward", 1024),
        dropout=config.get("dropout", 0.1),
        velocity_loss_weight=config.get("velocity_loss_weight", 0.2),
        continuity_loss_weight=config.get("continuity_loss_weight", 0.0),
        first_step_loss_weight=config.get("first_step_loss_weight", 0.0),
        mae_loss_weight=config.get("mae_loss_weight", 0.1),
        root_loss_weight=config.get("root_loss_weight", 1.0),
        local_pose_loss_weight=config.get("local_pose_loss_weight", 1.0),
        mpjpe_loss_weight=config.get("mpjpe_loss_weight", 0.0),
        short_loss_weight=config.get("short_loss_weight", 0.0),
        mid_loss_weight=config.get("mid_loss_weight", 0.0),
        long_loss_weight=config.get("long_loss_weight", 0.2),
        final_frame_loss_weight=config.get("final_frame_loss_weight", 0.2),
        acceleration_loss_weight=config.get("acceleration_loss_weight", 0.1),
        relative_root_loss_weight=config.get("relative_root_loss_weight", 0.2),
        relative_velocity_loss_weight=config.get("relative_velocity_loss_weight", 0.1),
        key_joint_relation_loss_weight=config.get("key_joint_relation_loss_weight", 0.0),
        contact_loss_weight=config.get("contact_loss_weight", 0.0),
        contact_threshold=config.get("contact_threshold", NTU_DEFAULT_CONTACT_THRESHOLD),
        action_feature_loss_weight=config.get("action_feature_loss_weight", 0.0),
        action_logit_loss_weight=config.get("action_logit_loss_weight", 0.0),
    )
