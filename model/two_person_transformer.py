"""双人 token 的人物内时序与显式跨人 attention。"""

import torch
import torch.nn as nn


def _activation(name):
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError("不支持的 activation: {}".format(name))


class _FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, activation):
        super(_FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, value):
        return self.net(value)


class TwoPersonInteractionEncoderLayer(nn.Module):
    """一层 obs encoder，A/B 的 temporal attention 与 FFN 保留来源。"""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="gelu"):
        super(TwoPersonInteractionEncoderLayer, self).__init__()
        self.temporal_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm_temporal_a = nn.LayerNorm(d_model)
        self.norm_temporal_b = nn.LayerNorm(d_model)
        self.norm_cross_a = nn.LayerNorm(d_model)
        self.norm_cross_b = nn.LayerNorm(d_model)
        self.norm_ffn_a = nn.LayerNorm(d_model)
        self.norm_ffn_b = nn.LayerNorm(d_model)
        self.ffn_a = _FeedForward(d_model, dim_feedforward, dropout, activation)
        self.ffn_b = _FeedForward(d_model, dim_feedforward, dropout, activation)

    def forward(self, person_a, person_b, temporal_mask=None):
        a_temporal, _ = self.temporal_attn(person_a, person_a, person_a, attn_mask=temporal_mask)
        b_temporal, _ = self.temporal_attn(person_b, person_b, person_b, attn_mask=temporal_mask)
        person_a = self.norm_temporal_a(person_a + a_temporal)
        person_b = self.norm_temporal_b(person_b + b_temporal)

        a_cross, _ = self.cross_attn(person_a, person_b, person_b)
        b_cross, _ = self.cross_attn(person_b, person_a, person_a)
        person_a = self.norm_cross_a(person_a + a_cross)
        person_b = self.norm_cross_b(person_b + b_cross)

        person_a = self.norm_ffn_a(person_a + self.ffn_a(person_a))
        person_b = self.norm_ffn_b(person_b + self.ffn_b(person_b))
        return person_a, person_b


class TwoPersonInteractionEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout=0.1, activation="gelu"):
        super(TwoPersonInteractionEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                TwoPersonInteractionEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, person_a, person_b, temporal_mask=None):
        for layer in self.layers:
            person_a, person_b = layer(person_a, person_b, temporal_mask=temporal_mask)
        return person_a, person_b


class TwoPersonForecastingDecoderLayer(nn.Module):
    """一层 future decoder，先人物内/间交互，再查询条件 memory。"""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="gelu"):
        super(TwoPersonForecastingDecoderLayer, self).__init__()
        self.temporal_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_person_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.memory_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm_temporal_a = nn.LayerNorm(d_model)
        self.norm_temporal_b = nn.LayerNorm(d_model)
        self.norm_cross_a = nn.LayerNorm(d_model)
        self.norm_cross_b = nn.LayerNorm(d_model)
        self.norm_memory_a = nn.LayerNorm(d_model)
        self.norm_memory_b = nn.LayerNorm(d_model)
        self.norm_ffn_a = nn.LayerNorm(d_model)
        self.norm_ffn_b = nn.LayerNorm(d_model)
        self.ffn_a = _FeedForward(d_model, dim_feedforward, dropout, activation)
        self.ffn_b = _FeedForward(d_model, dim_feedforward, dropout, activation)

    def forward(self, person_a, person_b, memory, temporal_mask=None):
        a_temporal, _ = self.temporal_attn(person_a, person_a, person_a, attn_mask=temporal_mask)
        b_temporal, _ = self.temporal_attn(person_b, person_b, person_b, attn_mask=temporal_mask)
        person_a = self.norm_temporal_a(person_a + a_temporal)
        person_b = self.norm_temporal_b(person_b + b_temporal)

        a_cross, _ = self.cross_person_attn(person_a, person_b, person_b)
        b_cross, _ = self.cross_person_attn(person_b, person_a, person_a)
        person_a = self.norm_cross_a(person_a + a_cross)
        person_b = self.norm_cross_b(person_b + b_cross)

        a_memory, _ = self.memory_attn(person_a, memory, memory)
        b_memory, _ = self.memory_attn(person_b, memory, memory)
        person_a = self.norm_memory_a(person_a + a_memory)
        person_b = self.norm_memory_b(person_b + b_memory)

        person_a = self.norm_ffn_a(person_a + self.ffn_a(person_a))
        person_b = self.norm_ffn_b(person_b + self.ffn_b(person_b))
        return person_a, person_b


class TwoPersonForecastingDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout=0.1, activation="gelu"):
        super(TwoPersonForecastingDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                TwoPersonForecastingDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, person_a, person_b, memory, temporal_mask=None):
        for layer in self.layers:
            person_a, person_b = layer(
                person_a,
                person_b,
                memory,
                temporal_mask=temporal_mask,
            )
        return person_a, person_b
