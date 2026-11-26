import torch
import torch.nn as nn
from functools import partial
from torch.nn import functional as F

class PyramidTemporalRegroupDetection(nn.Module):
    """        Pyramid Temporal-Regroup Detection Block    """
    def __init__(self, embed_dims=[32, 64, 128, 256], norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 1, 1, 1],
                 extra_norm=True):
        super(PyramidTemporalRegroupDetection, self).__init__()
        self.depths = depths

        self.extra_norm = extra_norm
        if self.extra_norm:
            self.norm_list = nn.ModuleList()
            for dim in embed_dims:
                self.norm_list.append(norm_layer(dim))

        self.patch_embeddings = nn.ModuleList()
        for i in range(len(depths)):
            if i == 0:
                self.patch_embeddings.append(
                    nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, stride=1, padding=1))
            else:
                self.patch_embeddings.append(
                    nn.Conv2d(embed_dims[i - 1], embed_dims[i], kernel_size=3, stride=2, padding=1))

        self.pos_block = nn.ModuleList([TsmConv(embed_dim, embed_dim) for embed_dim in embed_dims])
        self.decoder = SegHead(in_channels=embed_dims, embedding_dim=256)

    def forward_one_step(self, cur, ref):
        x = torch.cat([cur, ref], dim=1)
        b, t, c, h, w = x.shape

        outputs = []
        for i in range(len(self.depths)):
            x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
            x = self.patch_embeddings[i](x)
            x = x.view(b, t, x.shape[-3], x.shape[-2], x.shape[-1])
            x = self.pos_block[i](x)
            if self.extra_norm:
                x = self.norm_list[i](x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
            outputs.append(x)

        return outputs

    def forward(self, x, num_ref=5):
        b, T, c, h, w = x.shape
        ref_idx = T - num_ref
        indices = torch.tensor([ref_idx, ref_idx + num_ref // 2, ref_idx + num_ref - 1], device=x.device)
        ref = x.index_select(dim=1, index=indices)
        num_levels = len(self.depths)

        outputs_per_level = [[] for _ in range(num_levels)]
        for t in range(T):
            cur = x[:, t:t+1]
            feat_scales = self.forward_one_step(cur, ref)

            for lvl, feat in enumerate(feat_scales):
                feat_cur = feat[:, 0].contiguous()
                outputs_per_level[lvl].append(feat_cur)

        out = [torch.stack(feats, dim=1) for feats in outputs_per_level]
        out = self.decoder(out)
        return out


class TsmConv(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(TsmConv, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x):
        b, t, c, h, w = x.shape
        cnn_feat = tsm(x).view(-1, c, h, w)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.view(b, t, c, h, w)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


def tsm(tensor, version='circulant'):
    shape = B, T, C, H, W = tensor.shape
    split_size = C // 4

    pre_tensor, post_tensor, peri_tensor = tensor.split(
        [split_size, split_size, C - 2 * split_size],
        dim=2
    )
    if version == 'zero':
        pre_tensor = F.pad(pre_tensor, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1, ...]
        post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:, ...]
    elif version == 'circulant':
        pre_tensor = torch.cat((pre_tensor[:, -1:, ...],
                                pre_tensor[:, :-1, ...]), dim=1)
        post_tensor = torch.cat((post_tensor[:, 1:, ...],
                                 post_tensor[:, :1, ...]), dim=1)
    else:
        raise ValueError('Unknown TSM version: {}'.format(version))
    return torch.cat((pre_tensor, post_tensor, peri_tensor), dim=2).view(shape)


class SegHead(nn.Module):
    def __init__(self, in_channels, embedding_dim=768, dropout=0.1, interpolate_mode='bilinear', align_corners=False):
        super(SegHead, self).__init__()

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.dropout = nn.Dropout2d(dropout)
        self.interpolate_mode = interpolate_mode
        self.align_corners = align_corners

        self.linear_c4 = MLPLayer(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLPLayer(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLPLayer(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLPLayer(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim*4, out_channels=embedding_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=embedding_dim, out_channels=c1_in_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        c1, c2, c3, c4 = x
        n, t, _, _, _ = c4.shape
        c1 = c1.view(n * t, -1, c1.shape[3], c1.shape[4])
        c2 = c2.view(n * t, -1, c2.shape[3], c2.shape[4])
        c3 = c3.view(n * t, -1, c3.shape[3], c3.shape[4])
        c4 = c4.view(n * t, -1, c4.shape[3], c4.shape[4])

        _c4 = self.linear_c4(c4).permute(0,2,1).contiguous().reshape(n * t, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode=self.interpolate_mode, align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).permute(0,2,1).contiguous().reshape(n * t, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode=self.interpolate_mode, align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0,2,1).contiguous().reshape(n * t, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode=self.interpolate_mode, align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0,2,1).contiguous().reshape(n * t, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        return _c


class MLPLayer(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x




