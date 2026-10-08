"""
CoRe-Sleep model for SHHS sleep staging.
Combines positional encodings (SHHS_pos), transformer building blocks, SleepEnc encoder,
and Sleep_CoRe multimodal fusion model — all in one file.

Adapted from ~/Documents/CoRe-Sleep/ for the EEGBenchmarks plugin interface:
  - __init__(self, args, encs=[], **kwargs)
  - forward(self, data, label=None, return_features=False, **kwargs)
    returns {"preds": {"combined", "c", "g"}, "features": {...}, "losses": {"align": ...}}
"""

__all__ = [
    "Sleep_CoRe",
    "SleepEnc",
    "tf_encoder",
    "tf_layer",
    "positionwise_FC",
    "multihead_attention",
    "scaled_dot_product_att",
    "My_TransformerEncoder",
    "modtype_embedding",
    "pos_encoding_sinusoidal",
    "learned_pos_embedding",
    "sin_pos",
    "relative_pos_embedding",
]

import torch.nn as nn
import torch
import numpy as np
import einops
import copy
import math
from torch import Tensor
from typing import Optional
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Positional encodings  (from SHHS_pos.py)
# ---------------------------------------------------------------------------

class pos_encoding_sinusoidal(nn.Module):

    def __init__(self, d_hid, n_position=400):
        super(pos_encoding_sinusoidal, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

    def forward_concat(self, x):
        pos = self.pos_table[:, :x.size(1)].detach()
        x = torch.cat([x, pos.repeat(x.shape[0], 1, 1)], dim=2)
        return x


class learned_pos_embedding(nn.Module):
    def __init__(self, max_pos, dim):
        super().__init__()
        self.pos_inner_tokens = nn.Parameter(torch.randn(max_pos, dim), requires_grad=True)
        self.pos_outer_tokens = nn.Parameter(torch.randn(max_pos, dim), requires_grad=True)

    def forward_outer(self, data):
        """data: [batch, outer, inner, mod, ch, features]"""
        data_shape = data.shape
        data = einops.rearrange(data, "b outer inner mod ch k -> (b inner mod ch) outer k")
        data = data + self.pos_inner_tokens[:data.shape[1], :].squeeze()
        data = einops.rearrange(data, "(b inner mod ch) outer k -> b outer inner mod ch k",
                                b=data_shape[0], inner=data_shape[2], mod=data_shape[3], ch=data_shape[4])
        return data

    def forward_inner(self, data):
        """data: [batch, outer, inner, mod, ch, features]"""
        data_shape = data.shape
        data = einops.rearrange(data, "b outer inner mod ch k -> (b outer mod ch) inner k")
        data = data + self.pos_outer_tokens[:data.shape[1], :].squeeze()
        data = einops.rearrange(data, "(b outer mod ch) inner k -> b outer inner mod ch k",
                                b=data_shape[0], outer=data_shape[1], mod=data_shape[3], ch=data_shape[4])
        return data


class sin_pos(nn.Module):
    def __init__(self, dmodel, npoints=400):
        super().__init__()
        self.pos = pos_encoding_sinusoidal(dmodel, n_position=npoints)

    def forward_inner(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = (
            x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4])
        x = einops.rearrange(x, "b outer inner mod ch k -> (b outer mod ch) inner k")
        x = self.pos(x)
        x = einops.rearrange(x, "(b outer mod ch) inner k -> b outer inner mod ch k",
                              b=self.batch, outer=self.outer, mod=self.mod, ch=self.ch)
        return x

    def forward_outer(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = (
            x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4])
        x = einops.rearrange(x, "b outer inner mod ch k -> (b inner mod ch) outer k")
        x = self.pos(x)
        x = einops.rearrange(x, "(b inner mod ch) outer k -> b outer inner mod ch k",
                              b=self.batch, inner=self.inner, mod=self.mod, ch=self.ch)
        return x

    def forward_time_inner(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.time, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        x = einops.rearrange(x, "b outer time ch k -> (b outer ch) time k")
        x = self.pos(x)
        x = einops.rearrange(x, "(b outer ch) time k -> b outer time ch k",
                              b=self.batch, outer=self.outer, ch=self.ch)
        return x


class relative_pos_embedding(nn.Module):
    def __init__(self, tokens, dim_head, max_tokens=2000, heads=None):
        super().__init__()
        scale = dim_head ** -0.5
        self.shared_heads = heads if heads is not None else True
        if self.shared_heads:
            self.rel_pos_emb = nn.Parameter(torch.randn(heads, 2 * tokens - 1, dim_head) * scale)
        else:
            self.rel_pos_emb = nn.Parameter(torch.randn(2 * tokens - 1, dim_head) * scale)

        indices = torch.arange(-tokens, tokens + 1)
        self.indices_ext = torch.cat([
            indices[0].repeat(int((max_tokens - 2 * tokens + 1))),
            indices,
            indices[-1].repeat(int((max_tokens - 2 * tokens + 1)))
        ], dim=0)
        self.indices_ext = self.indices_ext.unsqueeze(dim=0).repeat(max_tokens, 1)
        for i in range(max_tokens):
            self.indices_ext[i] = torch.roll(self.indices_ext[i], shifts=i)
        self.indices_ext = self.indices_ext[:max_tokens - tokens + 2, max_tokens - tokens + 1:]

    def forward(self, q):
        if self.shared_heads:
            this_indices_ext = self.indices_ext[:q.shape[2], :q.shape[2]]
            emb = torch.einsum('b h t d, h r t d -> b h t r', q, self.rel_pos_emb[:, this_indices_ext])
        else:
            this_indices_ext = self.indices_ext[:q.shape[2], q.shape[2]]
            emb = torch.einsum('b h t d, r t d -> b h t r', q, self.rel_pos_emb[this_indices_ext])
        return emb


# ---------------------------------------------------------------------------
# Transformer building blocks  (from CoRe_model.py)
# ---------------------------------------------------------------------------

class scaled_dot_product_att(nn.Module):
    def __init__(self, rpos=False, d_head=16, max_len=7, heads=8):
        super().__init__()
        self.rpos = rpos
        self.heads = heads
        if rpos:
            self.k_rpos = relative_pos_embedding(tokens=max_len, dim_head=d_head, heads=heads)
            self.v_rpos = relative_pos_embedding(tokens=max_len, dim_head=d_head, heads=heads)

    def forward(self, query, key, value):
        query = einops.rearrange(query, "seq b f -> b seq f")
        key   = einops.rearrange(key,   "seq b f -> b seq f")
        value = einops.rearrange(value, "seq b f -> b seq f")
        dk = query.size()[-1]

        if self.rpos:
            rel_key = einops.rearrange(key, "(b h) seq f -> b h seq f",
                                       b=int(key.shape[0] / self.heads), h=self.heads)
            rel_key = self.k_rpos(rel_key)
            rel_key = einops.rearrange(rel_key, "b h seq f -> (b h) seq f")
            scores = (query.matmul(key.transpose(-2, -1)) + rel_key) / math.sqrt(dk)
        else:
            scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)

        attention = nn.functional.softmax(scores, dim=-1)
        attn_output = torch.einsum('b i j , b j d -> b i d', attention, value)
        attn_output = einops.rearrange(attn_output, "b seq f -> seq b f")
        return attn_output


class multihead_attention(nn.Module):
    def __init__(self, in_features, heads, bias=True, dim_proj=128, rpos=False):
        super(multihead_attention, self).__init__()
        self.activation = None
        if in_features % heads != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(
                in_features, heads))
        self.in_features = in_features
        self.head_num = heads
        self.bias = bias
        self.linear_q = nn.Linear(in_features, dim_proj, bias)
        self.linear_k = nn.Linear(in_features, dim_proj, bias)
        self.linear_v = nn.Linear(in_features, dim_proj, bias)
        self.linear_o = nn.Linear(dim_proj, in_features, False)
        self.scaled_dotproduct_attention = scaled_dot_product_att(
            rpos=rpos, d_head=int(dim_proj / heads), heads=heads)

    def forward(self, q, v, k):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        y = self.scaled_dotproduct_attention(q, k, v)
        y = self._reshape_from_batches(y)
        y = self.linear_o(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        seq_len, batch_size, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return einops.rearrange(x, "seq b (h sub_dim) -> seq (b h) sub_dim",
                                h=self.head_num, sub_dim=sub_dim)

    def _reshape_from_batches(self, x):
        seq_len, batch_size, in_feature = x.size()
        batch_size //= self.head_num
        return einops.rearrange(x, "seq (b h) sub_dim -> seq b (h sub_dim)",
                                h=self.head_num, b=batch_size)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation)


class positionwise_FC(nn.Module):
    def __init__(self, d_model, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_in = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout_out = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src: Tensor) -> Tensor:
        src_fc = self.linear2(self.dropout_in(self.activation(self.linear1(src))))
        src_att = self.norm(src + self.dropout_out(src_fc))
        return src_att


class tf_layer(nn.Module):
    def __init__(self, d_model, heads, CA_flag=False, CA_CA_flag=False, rpos=False,
                 dim_feedforward=1024, dim_proj=128, dropout=0.1, activation="relu"):
        super().__init__()
        self.CA_flag = CA_flag
        self.CA_CA_flag = CA_CA_flag
        self.self_attn = multihead_attention(d_model, heads, dim_proj=dim_proj, rpos=rpos)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if self.CA_flag:
            self.CA = multihead_attention(d_model, heads=heads, dim_proj=dim_proj, rpos=rpos)
            self.norm_CA = nn.LayerNorm(d_model)
            self.dropout_CA = nn.Dropout(dropout)
        if self.CA_CA_flag:
            self.CA_CA = multihead_attention(d_model, heads=heads, dim_proj=dim_proj, rpos=rpos)
            self.norm_CA_CA = nn.LayerNorm(d_model)
            self.dropout_CA_CA = nn.Dropout(dropout)
        self.fc_SA = positionwise_FC(d_model=d_model, dropout=dropout,
                                     dim_feedforward=dim_feedforward, activation=activation)

    def forward(self, src: Tensor,
                crossatt_src: Optional[Tensor] = None,
                crossatt_src_2: Optional[Tensor] = None,
                **kwargs) -> Tensor:
        src_att = self.self_attn(src, src, src, **kwargs)
        src_att = self.norm(src + self.dropout(src_att))
        if crossatt_src is not None:
            src_ca = self.CA(crossatt_src, src_att, src_att, **kwargs)
            src_att = self.norm_CA(src_att + self.dropout_CA(src_ca))
        if crossatt_src_2 is not None:
            src_ca = self.CA_CA(crossatt_src_2, src_att, src_att, **kwargs)
            src_att = self.norm_CA_CA(src_att + self.dropout_CA_CA(src_ca))
        src_att = self.fc_SA(src_att)
        return src_att


class My_TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(My_TransformerEncoder, self).__init__()

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor,
                src_ca: Optional[Tensor] = None,
                src_ca_ca: Optional[Tensor] = None,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                **kwargs) -> Tensor:
        return_layer = kwargs.get("return_layer", "last")
        ca_type = kwargs.get("ca_type", None)

        output = src
        output_list = []
        for li, mod in enumerate(self.layers):
            if src_ca is not None and src_ca_ca is not None:
                this_ca = src_ca[li] if ca_type == "full" else src_ca
                this_ca_ca = src_ca_ca[li] if ca_type == "full" else src_ca_ca
                output = mod(output, crossatt_src=this_ca, crossatt_src_1=this_ca_ca, **kwargs)
            elif src_ca is not None:
                this_ca = src_ca[li] if ca_type == "full" else src_ca
                output = mod(output, crossatt_src=this_ca, **kwargs)
            elif src_ca_ca is not None:
                this_ca_ca = src_ca_ca[li] if ca_type == "full" else src_ca_ca
                output = mod(output, crossatt_ca_src=this_ca_ca, **kwargs)
            else:
                output = mod(output, **kwargs)
            if return_layer != "last":
                output_list.append(output)

        if return_layer == "all":
            output = torch.cat([i.unsqueeze(dim=0) for i in output_list])
        return output


class tf_encoder(nn.Module):
    def __init__(self, dmodel, num_layers=4, heads=8, **kwargs):
        super().__init__()
        enc = tf_layer(dmodel, heads=heads, **kwargs)
        self.tf = My_TransformerEncoder(enc, num_layers=num_layers)

    def forward(self, x, **kwargs):
        return self.forward_sa(x, **kwargs)

    def forward_inner(self, x, x_ca=None, x_ca_ca=None, **kwargs):
        x_shape = x.shape
        batch, outer, inner, mod, ch, features = (
            x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5])
        x = einops.rearrange(x, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        if x_ca is not None:
            x_ca = einops.rearrange(x_ca, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        if x_ca_ca is not None:
            x_ca_ca = einops.rearrange(x_ca_ca, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        x = self.tf(x, src_ca=x_ca, src_ca_ca=x_ca_ca, **kwargs)
        x = einops.rearrange(x, "(inner mod ch) (b outer) k -> b outer inner mod ch k",
                              outer=outer, mod=mod, ch=ch, b=batch)
        return x

    def forward_outer(self, x, x_ca=None, x_ca_ca=None, **kwargs):
        x_shape = x.shape
        batch, outer, inner, mod, ch, features = (
            x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5])
        x = einops.rearrange(x, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        if x_ca is not None:
            x_ca = einops.rearrange(x_ca, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        if x_ca_ca is not None:
            x_ca_ca = einops.rearrange(x_ca_ca, "b outer inner mod ch k -> (outer mod) (b inner ch) k")
        x = self.tf(x, src_ca=x_ca, src_ca_ca=x_ca_ca, **kwargs)
        x = einops.rearrange(x, "(outer mod) (b inner ch) k -> b outer inner mod ch k",
                              outer=outer, mod=mod, ch=ch, b=batch)
        return x


class modtype_embedding(nn.Module):
    def __init__(self, num_modalities, dim):
        super().__init__()
        self.mod_tokens = nn.Parameter(torch.randn(num_modalities, dim), requires_grad=True)

    def forward(self, data, mod_num):
        return data + self.mod_tokens[mod_num]


# ---------------------------------------------------------------------------
# SleepEnc encoder  (from CoRe_model.py)
# ---------------------------------------------------------------------------

class SleepEnc(nn.Module):
    def __init__(self, args, encs=[]):
        super().__init__()
        self.args = args
        self.pos = args.get("pos", True)

        d_model = args.dmodel
        rpos = args.get("rpos", False)
        heads = args.get("heads", 8)
        dropout = args.get("dropout", 0.1)
        num_layers = args.get("num_layers", 4)
        dim_proj = args.get("dim_proj", d_model)
        CA_flag = args.get("CA_flag", False)

        self.inner_tf = tf_encoder(CA_flag=CA_flag, dmodel=d_model, num_layers=num_layers,
                                   heads=heads, dim_proj=dim_proj, rpos=rpos, dropout=dropout)
        self.outer_tf = tf_encoder(CA_flag=CA_flag, dmodel=d_model, num_layers=num_layers,
                                   heads=heads, dim_proj=dim_proj, rpos=rpos, dropout=dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        if self.pos == "trained":
            self.pos_emb = learned_pos_embedding(max_pos=200, dim=d_model)
        elif self.pos == "sinusoidal":
            self.pos_emb = sin_pos(d_model)

        self.fc = nn.Linear(d_model, 5)

    def forward(self, x, extract_norm=False, **kwargs):
        if self.args.modality == "eeg":
            x_d = x["stft_eeg"]
        elif self.args.modality == "eog":
            x_d = x["stft_eog"]
        else:
            raise ValueError("Modality not supported")

        if "stft_eeg" in x and "stft_eog" in x:
            x_d = einops.rearrange(x_d[:, :, :, :, 1:, :],
                                   "b outer mod ch f inner -> b outer inner mod ch f")
        else:
            x_d = einops.rearrange(x_d[:, 1:, :, :, :, :],
                                   "b f mod ch inner outer -> b outer inner mod ch f")

        x_d = x_d.float()

        if self.pos:
            x_d = self.pos_emb.forward_inner(x_d)

        cls_token_d = self.cls_token.repeat(x_d.shape[0], x_d.shape[1], 1, 1, x_d.shape[3], 1)
        x_d = torch.cat([cls_token_d, x_d], dim=2)

        x_feat = self.inner_tf.forward_inner(x_d)[:, :, :1]

        if self.pos:
            x_feat = self.pos_emb.forward_outer(x_feat)
        x_feat = self.outer_tf.forward_outer(x_feat)

        aggregated_x = einops.rearrange(x_feat,
                                        "b outer mod ch inner f -> (b outer inner mod ch) f")
        preds = self.fc(aggregated_x)

        return {"preds": {"combined": preds}, "features": {"combined": aggregated_x}}


# ---------------------------------------------------------------------------
# Sleep_CoRe — main multimodal fusion model  (from CoRe_model.py)
# Adapted for EEGBenchmarks: __init__(args, encs, **kwargs) and
#   forward(data, label=None, return_features=False, **kwargs)
# ---------------------------------------------------------------------------

class Sleep_CoRe(nn.Module):
    def __init__(self, args, encs=[], **kwargs):
        super().__init__()
        self.args = args

        d_model = args.d_model
        self.pos = args.get("pos", True)
        self.outer_rep = args.get("outer_rep", False)
        self.skip_percentile = args.get("skip_percentile", False)
        self.mod_token = args.get("mod_token", False)
        self.align_inner = args.get("align_inner", False)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        if self.mod_token:
            self.modtype_token = modtype_embedding(num_modalities=2, dim=d_model)

        self.mm_fc = nn.Linear(d_model, 5)

    def prepare_modalities(self, x, skip_modality="full"):
        xeeg = None
        if skip_modality != "eeg":
            xeeg = x["stft_eeg"][:, :, :, :, 1:, :]
            xeeg = xeeg.float()
            xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
            if type(skip_modality) == dict and "stft_eeg" in skip_modality:
                xeeg_shape = xeeg.shape
                xeeg = xeeg[~skip_modality["stft_eeg"].bool()]
                xeeg = einops.rearrange(xeeg, "(b outer) i m c f -> b outer i m c f",
                                        outer=xeeg_shape[1],
                                        b=int(xeeg.shape[0] / xeeg_shape[1]))
            if self.mod_token:
                xeeg = self.modtype_token(data=xeeg, mod_num=0)
            if self.pos:
                xeeg = self.enc_0.pos_emb.forward_inner(xeeg)
            cls_token_eeg = self.enc_0.cls_token.repeat(
                xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
            xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        xeog = None
        if skip_modality != "eog":
            xeog = x["stft_eog"][:, :, :, :, 1:, :]
            xeog = xeog.float()
            xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")
            if type(skip_modality) == dict and "stft_eog" in skip_modality:
                xeog_shape = xeog.shape
                xeog = xeog[~skip_modality["stft_eog"].bool()]
                xeog = einops.rearrange(xeog, "(b outer) i m c f -> b outer i m c f",
                                        outer=xeog_shape[1],
                                        b=int(xeog.shape[0] / xeog_shape[1]))
            if self.mod_token:
                xeog = self.modtype_token(data=xeog, mod_num=1)
            if self.pos:
                xeog = self.enc_1.pos_emb.forward_inner(xeog)
            cls_token_eog = self.enc_1.cls_token.repeat(
                xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
            xeog = torch.cat([cls_token_eog, xeog], dim=2)

        return xeeg, xeog

    def forward(self, data, label=None, return_features=False, **kwargs):
        """
        EEGBenchmarks-compatible forward.

        Args:
            data: dict with keys "stft_eeg" and "stft_eog"
                  shape: [batch, outer_seq, mod, ch, freq+1, inner]
            label: ignored (loss computed externally by Trainer)
            return_features: ignored (features always returned)
            **kwargs: may contain "skip_view" (dict of skip masks per modality)

        Returns:
            {
                "preds":    {"combined": Tensor, "c": Tensor, "g": Tensor},
                "features": {"combined": Tensor, "eeg": Tensor, "eog": Tensor, ...},
                "losses":   {"align": Tensor}   (only when both modalities present)
            }
        """
        skip_modality = kwargs.get("skip_view", "full")

        xeeg, xeog = self.prepare_modalities(data, skip_modality)

        output = {"preds": {}, "features": {}}
        output = self.forward_sole(xeeg=xeeg, xeog=xeog, output=output,
                                   skip_modality=skip_modality,
                                   align_inner=self.align_inner, **kwargs)
        if (xeeg is not None and xeog is not None
                and "inner_eeg" in output["features"]
                and "inner_eog" in output["features"]):
            output = self.forward_common(xeeg=xeeg, xeog=xeog, output=output,
                                         skip_modality=skip_modality, **kwargs)

        if skip_modality != "eeg" and skip_modality != "eog":
            output = self.get_matches(xeeg, output, skip_modality, **kwargs)

        if skip_modality != "eog" and skip_modality != "eeg":
            output["preds"]["combined"] = self.mm_fc(output["features"]["combined"])
        if skip_modality != "eeg":
            output["preds"]["c"] = self.enc_0.fc(output["features"]["eeg"])
        if skip_modality != "eog":
            output["preds"]["g"] = self.enc_1.fc(output["features"]["eog"])

        return output

    def _keep_common(self, x, common_idx, skip_idx):
        if len(x.shape) == 6:
            output = x[einops.rearrange(~common_idx[~skip_idx.bool()], "(b outer) -> b outer",
                                        b=x.shape[0], outer=x.shape[1])]
            output = einops.rearrange(output, "(b outer) i m c f -> b outer i m c f",
                                      outer=common_idx.shape[1],
                                      b=int(output.shape[0] / common_idx.shape[1]))
        elif len(x.shape) == 3:
            output = x[einops.rearrange(~common_idx[~skip_idx.bool()], "(b outer) -> b outer",
                                        b=x.shape[0], outer=x.shape[1])]
            output = einops.rearrange(output, "(b outer) f -> b outer f",
                                      outer=common_idx.shape[1],
                                      b=int(output.shape[0] / common_idx.shape[1]))
        elif len(x.shape) == 2:
            x = x.unsqueeze(dim=1)
            output = x[einops.rearrange(~common_idx[~skip_idx.bool()], "(b outer) -> b outer",
                                        b=x.shape[0], outer=x.shape[1])]
            output = einops.rearrange(output, "(b outer) f -> b outer f",
                                      outer=common_idx.shape[1],
                                      b=int(output.shape[0] / common_idx.shape[1]))
        return output

    def forward_common(self, xeeg, xeog, output, **kwargs):
        skip_modality = kwargs["skip_modality"]

        xeeg_common_i = output["features"]["inner_eeg"]
        xeog_common_i = output["features"]["inner_eog"]

        if xeeg_common_i.shape[1] > 1 and xeog_common_i.shape[1] > 1:
            xeeg_common_outer = output["features"]["eeg"]
            xeog_common_outer = output["features"]["eog"]

        if (type(skip_modality) == dict
                and "stft_eeg" in skip_modality
                and "stft_eog" in skip_modality):
            common_kept_idx = torch.logical_or(
                skip_modality["stft_eeg"].bool(), skip_modality["stft_eog"].bool())
            if "xeeg_common_outer" in locals() and "xeog_common_outer" in locals():
                xeeg_common_outer = self._keep_common(
                    xeeg_common_outer, common_kept_idx, skip_modality["stft_eeg"])
                xeog_common_outer = self._keep_common(
                    xeog_common_outer, common_kept_idx, skip_modality["stft_eog"])
            xeeg = self._keep_common(xeeg, common_kept_idx, skip_modality["stft_eeg"])
            xeog = self._keep_common(xeog, common_kept_idx, skip_modality["stft_eog"])

        xeeg_ca_common = self.enc_0.inner_tf.forward_inner(xeeg, xeog_common_i)[:, :, :1]
        xeog_ca_common = self.enc_1.inner_tf.forward_inner(xeog, xeeg_common_i)[:, :, :1]

        if "xeeg_common_outer" in locals() and "xeog_common_outer" in locals():
            if self.pos:
                xeeg_ca_common = self.enc_0.pos_emb.forward_outer(xeeg_ca_common)
                xeog_ca_common = self.enc_1.pos_emb.forward_outer(xeog_ca_common)
            xeeg_ca_common_outer = self.enc_0.outer_tf.forward_outer(
                xeeg_ca_common, xeog_common_outer)
            xeog_ca_common_outer = self.enc_1.outer_tf.forward_outer(
                xeog_ca_common, xeeg_common_outer)
            x_common = xeeg_ca_common_outer + xeog_ca_common_outer
        else:
            x_common = xeeg_ca_common + xeog_ca_common

        output["features"]["combined"] = x_common
        output["features"]["eeg"] = einops.rearrange(
            output["features"]["eeg"],
            "b outer mod ch inner f -> (b outer inner mod ch) f")
        output["features"]["eog"] = einops.rearrange(
            output["features"]["eog"],
            "b outer mod ch inner f -> (b outer inner mod ch) f")
        output["features"]["combined"] = einops.rearrange(
            output["features"]["combined"],
            "b outer mod ch inner f -> (b outer inner mod ch) f")
        return output

    def get_matches(self, x, output, skip_modality, **kwargs):
        xeeg_match_sq = output["features"]["inner_eeg"][:, :, :1].squeeze()
        xeog_match_sq = output["features"]["inner_eog"][:, :, :1].squeeze()

        if (type(skip_modality) == dict
                and "stft_eeg" in skip_modality
                and "stft_eog" in skip_modality):
            common_kept_idx = torch.logical_or(
                skip_modality["stft_eeg"].bool(), skip_modality["stft_eog"].bool())
            xeeg_match_sq = self._keep_common(
                xeeg_match_sq, common_kept_idx, skip_modality["stft_eeg"])
            xeog_match_sq = self._keep_common(
                xeog_match_sq, common_kept_idx, skip_modality["stft_eog"])
            output["features"]["inner_eeg"] = self._keep_common(
                output["features"]["inner_eeg"], common_kept_idx, skip_modality["stft_eeg"])
            output["features"]["inner_eog"] = self._keep_common(
                output["features"]["inner_eog"], common_kept_idx, skip_modality["stft_eog"])

        if (len(xeeg_match_sq.shape) == 3 and len(xeog_match_sq.shape) == 3
                and xeeg_match_sq.shape[0] > 0 and xeog_match_sq.shape[0] > 0):
            if self.args.get("big_al", False):
                xeeg_n = einops.rearrange(xeeg_match_sq, "b o f -> (b o) f")
                xeog_n = einops.rearrange(xeog_match_sq, "b o f -> (b o) f")
                xeeg_n = xeeg_n / xeeg_n.norm(dim=1, keepdim=True)
                xeog_n = xeog_n / xeog_n.norm(dim=1, keepdim=True)
                x_match_eeg = torch.matmul(xeeg_n, xeog_n.t())
                x_match_eog = x_match_eeg.permute(1, 0)
            else:
                xeeg_n = xeeg_match_sq / xeeg_match_sq.norm(dim=1, keepdim=True)
                xeog_n = xeog_match_sq / xeog_match_sq.norm(dim=1, keepdim=True)
                x_match_eeg = torch.einsum("b o f, b m f -> b o m", xeeg_n, xeog_n)
                x_match_eog = x_match_eeg.permute(0, 2, 1)
            target = (torch.arange(x_match_eeg.shape[1])
                      .to(x_match_eeg.device).unsqueeze(0)
                      .repeat(x_match_eeg.shape[0], 1))
            output["losses"] = {
                "align": (F.cross_entropy(x_match_eeg, target)
                          + F.cross_entropy(x_match_eog, target))
            }
            output["losses"]["align"] = (
                output["losses"]["align"]
                * self.args.multi_loss.multi_supervised_w.get("align", 0))

        elif (len(xeeg_match_sq.shape) == 2 and len(xeog_match_sq.shape) == 2):
            if self.args.get("big_al", False):
                xeeg_n = einops.rearrange(xeeg_match_sq, "b o f -> (b o) f")
                xeog_n = einops.rearrange(xeog_match_sq, "b o f -> (b o) f")
                xeeg_n = xeeg_n / xeeg_n.norm(dim=1, keepdim=True)
                xeog_n = xeog_n / xeog_n.norm(dim=1, keepdim=True)
                x_match_eeg = torch.matmul(xeeg_n, xeog_n.t())
                x_match_eog = x_match_eeg.permute(1, 0)
            else:
                xeeg_n = xeeg_match_sq / xeeg_match_sq.norm(dim=1, keepdim=True)
                xeog_n = xeog_match_sq / xeog_match_sq.norm(dim=1, keepdim=True)
                x_match_eeg = torch.einsum("o f, m f -> o m", xeeg_n, xeog_n)
                x_match_eog = x_match_eeg.permute(1, 0)
            output["losses"] = {
                "align": (
                    F.cross_entropy(x_match_eeg,
                                    torch.arange(x_match_eeg.shape[0]).to(x_match_eeg.device))
                    + F.cross_entropy(x_match_eog,
                                      torch.arange(x_match_eog.shape[0]).to(x_match_eog.device)))
            }
            output["losses"]["align"] = (
                output["losses"]["align"]
                * self.args.multi_loss.multi_supervised_w.get("align", 0))

        return output

    def forward_sole(self, xeeg, xeog, output, skip_modality, return_matches=False, **kwargs):
        if xeeg is not None and skip_modality != "eeg" and xeeg.shape[0] > 0:
            xeeg_sole = self.enc_0.inner_tf.forward_inner(xeeg)
            output["features"]["inner_eeg"] = xeeg_sole
            xeeg_cls_sole = xeeg_sole[:, :, :1]
            if xeeg_cls_sole.shape[1] > 1:
                xeeg_outer_sole = self.enc_0.outer_tf.forward_outer(xeeg_cls_sole)
                output["features"]["eeg"] = xeeg_outer_sole
            else:
                output["features"]["eeg"] = xeeg_cls_sole

        if xeog is not None and skip_modality != "eog" and xeog.shape[0] > 0:
            xeog_sole = self.enc_1.inner_tf.forward_inner(xeog)
            output["features"]["inner_eog"] = xeog_sole
            xeog_cls_sole = xeog_sole[:, :, :1]
            if xeog_cls_sole.shape[1] > 1:
                xeog_outer_sole = self.enc_1.outer_tf.forward_outer(xeog_cls_sole)
                output["features"]["eog"] = xeog_outer_sole
            else:
                output["features"]["eog"] = xeog_cls_sole

        return output
