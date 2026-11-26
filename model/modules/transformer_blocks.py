"""
    This code is based on:
    [1] Towards An End-to-End Framework for Video Inpainting, CVPR 2022
        https://github.com/MCG-NKU/E2FGVI
    [2] MambaIR: A Simple Baseline for Image Restoration with State-Space Model, ECCV 2024
        https://github.com/csguoh/MambaIR
"""

from functools import reduce
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.modules.mambair_arch import SS2D

class SoftSplit(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding,
                 t2t_param):
        super(SoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.t2t = nn.Unfold(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)

        self.t2t_param = t2t_param

    def forward(self, x, b, output_size):
        f_h = int((output_size[0] + 2 * self.t2t_param['padding'][0] -
                   (self.t2t_param['kernel_size'][0] - 1) - 1) /
                  self.t2t_param['stride'][0] + 1)

        f_w = int((output_size[1] + 2 * self.t2t_param['padding'][1] -
                   (self.t2t_param['kernel_size'][1] - 1) - 1) /
                  self.t2t_param['stride'][1] + 1)
        feat = self.t2t(x)
        feat = feat.permute(0, 2, 1)
        feat = self.embedding(feat)

        feat = feat.view(b, -1, f_h, f_w, feat.size(2))
        return feat


class SoftComp(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super(SoftComp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_conv = nn.Conv2d(channel,
                                   channel,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

    def forward(self, x, t, output_size):
        b_, _, _, _, c_ = x.shape
        x = x.view(b_, -1, c_)
        feat = self.embedding(x)
        b, _, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = F.fold(feat,
                      output_size=output_size,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding)
        feat = self.bias_conv(feat)
        return feat


class HybridSpatialTemporalLayer(nn.Module):
    """ Hybrid Spatial-Temporal Layer """
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.num_classes = 5
        self.extra_norm = norm_layer(dim)
        self.gate_fc = nn.Sequential(
            nn.Linear(self.num_classes, dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, 1),
            nn.Tanh()
        )
        self.norm1 = norm_layer(dim)
        self.d_state = 16
        self.selectiveScan = SS2D(d_model=dim, d_state=self.d_state, expand=2)

        self.norm2 = norm_layer(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, x, classes):
        B, T, H, W, C = x.shape

        xt = rearrange(x, 'b t h w m -> (b h w) t m', b=B, h=H, w=W, t=T)
        res_temporal = self.temporal_attn(self.temporal_norm1(xt))
        res_temporal = rearrange(res_temporal, '(b h w) t m -> b t h w m', b=B, h=H, w=W, t=T)
        xt = x + res_temporal

        # CDGMamba
        xs = xt
        xs = rearrange(self.extra_norm(xs), 'b t h w m -> (b t) (h w) m', b=B, h=H, w=W, t=T)
        one_hot = F.one_hot(classes, num_classes=self.num_classes).float()
        gates = self.gate_fc(one_hot)
        fused_tokens = xs * gates + xs
        fused_tokens = rearrange(fused_tokens, '(b t) (h w) m -> (b t) h w m', b=B, h=H, w=W, t=T)
        res_spatial = self.selectiveScan(self.norm1(fused_tokens))
        res_spatial = rearrange(res_spatial, '(b t) h w m -> b t h w m', b=B, h=H, w=W, t=T)
        x = xt + res_spatial

        x = x.view(-1, H * W, C)
        x = x + self.mlp(self.norm2(x), T, H, W)
        x = x.reshape(B, T, H, W, C)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim = 768):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, t, H, W):
        BT, N, C = x.shape
        B = BT // t
        assert H * W == N
        x = x.view(B, t, H, W, C).permute(0, 4, 1, 2, 3)
        x = self.dwconv(x)
        x = x.contiguous().permute(0, 2, 3, 4, 1).reshape(BT, N, C)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwc = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, t, H, W):
        x = self.fc1(x)
        x = self.dwc(x, t, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



