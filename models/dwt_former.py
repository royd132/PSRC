"""DWT-Former baseline ported from

https://github.com/MVTC-Lab/DWT-Former
(models/model.py, models/multiscalemix.py, models/bone.py,
models/attation.py, models/embed.py).

Faithful to the published architecture: a dmey level-5 discrete wavelet
split feeding two stacked multi-scale trend/season mixing blocks, fused
with an inverted full-attention transformer branch.

Adaptations required by the local shared protocol (documented in the
final report):
* ``c_in``/``c_out`` follow each station's actual channel count instead
  of the published 8; only the last (target) channel is returned.
* The five calendar mark channels are supplied as zeros.  With the
  published ``timeF`` pipeline the wavelet branch casts the continuous
  marks to ``long`` (all truncate to zero), and every local baseline
  receives time information through the four clock variates that are
  already part of the input bundle, so this preserves both the exact
  layer widths and information parity across baselines.
* The per-sample CPU ``pywt`` loop is batched over the sample axis; the
  numerical result is identical (verified) and the decomposition stays
  detached, as in the original implementation.
"""

import math

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

WAVELET = "dmey"
WAVELET_LEVEL = 5
TRANSFORMER_DIM = 128
MARK_DIMS = {"h": 4, "t": 5}


def _wavelet_split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised copy of Multibolck.wavelet_decompose over the batch.

    Returns trend/season tensors truncated to the input length.  The
    wavelet transform has no learnable parameters, so the inputs are
    detached (the original code moves them to CPU and back, severing the
    autograd link through the transform).
    """
    _, length = x.shape
    device = x.device
    values = x.detach().to("cpu").numpy()
    coeffs = pywt.wavedec(values, WAVELET, level=WAVELET_LEVEL, axis=1)
    trend = pywt.waverec(
        [coeffs[0]] + [None] * WAVELET_LEVEL, WAVELET, axis=1
    )[:, :length]
    season = pywt.waverec([None] + coeffs[1:], WAVELET, axis=1)[:, :length]
    trend = torch.as_tensor(np.ascontiguousarray(trend), device=device)
    season = torch.as_tensor(np.ascontiguousarray(season), device=device)
    return trend.unsqueeze(-1), season.unsqueeze(-1)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        ).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super().__init__()
        minute_size, hour_size = 4, 24
        weekday_size, day_size, month_size = 7, 32, 13
        embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = embed(minute_size, d_model)
        self.hour_embed = embed(hour_size, d_model)
        self.weekday_embed = embed(weekday_size, d_model)
        self.day_embed = embed(day_size, d_model)
        self.month_embed = embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        return (
            self.hour_embed(x[:, :, 3])
            + self.weekday_embed(x[:, :, 2])
            + self.day_embed(x[:, :, 1])
            + self.month_embed(x[:, :, 0])
            + minute_x
        )


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = (
            self.value_embedding(x)
            + self.position_embedding(x)
            + self.temporal_embedding(x_mark)
        )
        return self.dropout(x)


class DataEmbeddingInverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        if x_mark is not None:
            x = torch.cat([x, x_mark.permute(0, 2, 1)], 1)
        return self.dropout(self.value_embedding(x))


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        b, _, heads, e_dim = queries.shape
        scale = self.scale or 1.0 / math.sqrt(e_dim)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                raise NotImplementedError("causal masks are unused by DWT-Former")
            scores.masked_fill_(attn_mask, -torch.inf)
        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attention, values)
        return (out.contiguous(), attention if self.output_attention else None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        batch, length, _ = queries.shape
        _, source, _ = keys.shape
        heads = self.n_heads
        queries = self.query_projection(queries).view(batch, length, heads, -1)
        keys = self.key_projection(keys).view(batch, source, heads, -1)
        values = self.value_projection(values).view(batch, source, heads, -1)
        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        return self.out_projection(out.view(batch, length, -1)), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class MultiScaleTrendMixing(nn.Module):
    """Top-down mixing trend pattern (verbatim semantics)."""

    def __init__(self, seq_len, sampling_layers):
        super().__init__()
        self.up_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(seq_len // 2 ** (i + 1), seq_len // 2**i),
                    nn.GELU(),
                    nn.Linear(seq_len // 2**i, seq_len // 2**i),
                )
                for i in reversed(range(sampling_layers))
            ]
        )

    def forward(self, trend_list):
        trend_reversed = trend_list.copy()
        trend_reversed.reverse()
        if len(trend_list) == 1:
            out_low = out_high = trend_reversed[0]
        else:
            out_low = trend_reversed[0]
            out_high = trend_reversed[1]
        out_list = [out_low]
        for i in range(len(trend_reversed) - 1):
            out_high_res = self.up_sampling_layers[i](
                out_low.permute(0, 2, 1)
            ).permute(0, 2, 1)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_reversed) - 1:
                out_high = trend_reversed[i + 2]
            out_list.append(out_low)
        out_list.reverse()
        return out_list


class MultiScaleSeasonMixing(nn.Module):
    """Bottom-up mixing season pattern (verbatim semantics)."""

    def __init__(self, seq_len, sampling_layers):
        super().__init__()
        self.down_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        seq_len // 2**i, seq_len // 2 ** (i + 1)
                    ),
                    nn.GELU(),
                    nn.Linear(
                        seq_len // 2 ** (i + 1), seq_len // 2 ** (i + 1)
                    ),
                )
                for i in range(sampling_layers)
            ]
        )

    def forward(self, season_list):
        if len(season_list) == 1:
            out_low = out_high = season_list[0]
        else:
            out_high = season_list[0]
            out_low = season_list[1]
        out_list = [out_high]
        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](
                out_high.permute(0, 2, 1)
            ).permute(0, 2, 1)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_list.append(out_high)
        return out_list


class DeMultiBlock(nn.Module):
    def __init__(self, d_model, seq_len, sampling_layers):
        super().__init__()
        self.chan2pre = nn.ModuleList(
            [nn.Linear(seq_len // 2**i, seq_len) for i in range(sampling_layers + 1)]
        )
        self.act = nn.GELU()
        self.pre2pre = nn.ModuleList(
            [nn.Linear(seq_len, seq_len) for _ in range(sampling_layers + 1)]
        )
        self.project = nn.Linear(d_model * (sampling_layers + 1), d_model)

    def forward(self, scales):
        out_list = []
        for i, x in enumerate(scales):
            out = self.pre2pre[i](self.act(self.chan2pre[i](x.transpose(1, 2))))
            out_list.append(out.transpose(1, 2))
        return self.project(torch.cat(out_list, dim=-1))


class MultiBlock(nn.Module):
    def __init__(
        self, seq_len, d_model, freq, dropout, sampling_layers, d_ff, time_mix=True
    ):
        super().__init__()
        self.sampling_layers = sampling_layers
        self.time_mix = time_mix
        self.embedding_t = DataEmbedding(1, d_model, "fixed", freq, dropout)
        self.embedding_s = DataEmbedding(1, d_model, "fixed", freq, dropout)
        self.line_bedding_t = nn.Linear(1, d_model)
        self.line_bedding_s = nn.Linear(1, d_model)
        self.act = nn.GELU()
        self.trend_mixing = MultiScaleTrendMixing(seq_len, sampling_layers)
        self.season_mixing = MultiScaleSeasonMixing(seq_len, sampling_layers)
        self.down_pool = nn.AvgPool1d(2)
        self.project = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.d_model_to_one = nn.Linear(d_model, 1)

    def _multi_scale_inputs(self, trend, season):
        trend_scales = [trend]
        season_scales = [season]
        for _ in range(self.sampling_layers):
            trend = self.down_pool(trend.permute(0, 2, 1)).permute(0, 2, 1)
            season = self.down_pool(season.permute(0, 2, 1)).permute(0, 2, 1)
            trend_scales.append(trend)
            season_scales.append(season)
        return trend_scales, season_scales

    def forward(self, target, x_mark):
        trend, season = _wavelet_split(target)
        if self.time_mix:
            trend = self.embedding_t(trend, x_mark)
            season = self.embedding_s(season, x_mark)
        else:
            trend = self.act(self.line_bedding_t(trend))
            season = self.act(self.line_bedding_s(season))
        trend_scales, season_scales = self._multi_scale_inputs(trend, season)
        trend_scales = self.trend_mixing(trend_scales)
        season_scales = self.season_mixing(season_scales)
        out_list = [
            out_season + out_trend
            for out_season, out_trend in zip(season_scales, trend_scales)
        ]
        residual = self.norm(self.project(target.unsqueeze(-1)))
        out_list[0] = out_list[0] + residual
        result = self.d_model_to_one(out_list[0])
        return out_list, result


class SeqToPre(nn.Module):
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.linear1 = nn.Linear(seq_len, pred_len)
        self.linear2 = nn.Linear(pred_len, pred_len)
        self.norm = nn.LayerNorm(pred_len)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.norm(self.linear1(x)))
        x = self.act(self.norm(self.linear2(x))).transpose(1, 2)
        return x


class DWTFormer(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        c_in,
        c_out,
        d_model=32,
        d_ff=1024,
        dropout=0.09,
        sampling_layers=2,
        freq="t",
    ):
        super().__init__()
        self.pred_len = pred_len
        self.multi_block_one = MultiBlock(
            seq_len, d_model, freq, dropout, sampling_layers, d_ff, time_mix=True
        )
        self.multi_block_two = MultiBlock(
            seq_len, d_model, freq, dropout, sampling_layers, d_ff, time_mix=False
        )
        self.enc_embedding = DataEmbeddingInverted(
            seq_len, TRANSFORMER_DIM, dropout=0.1
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor=5,
                            attention_dropout=0.1,
                            output_attention=True,
                        ),
                        TRANSFORMER_DIM,
                        8,
                    ),
                    d_model=TRANSFORMER_DIM,
                    d_ff=TRANSFORMER_DIM,
                    dropout=0.1,
                    activation="gelu",
                )
                for _ in range(2)
            ],
            norm_layer=nn.LayerNorm(TRANSFORMER_DIM),
        )
        self.demulti = DeMultiBlock(d_model, seq_len, sampling_layers)
        self.seq2pre = SeqToPre(seq_len, pred_len)
        fusion_dim = c_in + MARK_DIMS[freq] + d_model
        self.line = nn.Linear(fusion_dim, d_ff)
        self.fusion_norm = nn.LayerNorm(d_ff)
        self.act = nn.GELU()
        self.projection = nn.Linear(TRANSFORMER_DIM, pred_len)
        self.projection2 = nn.Linear(d_ff, c_out)

    def forward(self, x, x_mark):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        target = x[:, :, -1]
        out_one_list, out_one = self.multi_block_one(target, x_mark)
        out_two_list, _ = self.multi_block_two(out_one.squeeze(-1).detach(), x_mark)
        out_list = [one + two for one, two in zip(out_one_list, out_two_list)]
        wave_out = self.seq2pre(self.demulti(out_list).transpose(1, 2))

        enc_out, _ = self.encoder(self.enc_embedding(x, x_mark))
        transformer_out = self.projection(enc_out).permute(0, 2, 1)

        fused = torch.cat([wave_out, transformer_out], dim=-1)
        fused = self.act(self.fusion_norm(self.line(fused)))
        dec_out = self.projection2(fused)

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # 15-minute protocol => published run.py setting freq='t' (5 marks).
        self.freq = "t"
        self.mark_dim = MARK_DIMS[self.freq]
        channels = int(configs.enc_in)
        self.model = DWTFormer(
            seq_len=int(configs.seq_len),
            pred_len=int(configs.pred_len),
            c_in=channels,
            c_out=channels,
            d_model=int(getattr(configs, "dwt_d_model", 32)),
            d_ff=int(getattr(configs, "dwt_d_ff", 1024)),
            dropout=float(getattr(configs, "dwt_dropout", 0.09)),
            sampling_layers=int(getattr(configs, "dwt_sampling_layers", 2)),
            freq=self.freq,
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask
        # The data bundle already carries the target as its last channel.
        marks = x_enc.new_zeros((x_enc.shape[0], x_enc.shape[1], self.mark_dim))
        dec_out = self.model(x_enc, marks)
        return dec_out[:, :, -1:]
