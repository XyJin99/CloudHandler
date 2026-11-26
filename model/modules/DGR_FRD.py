import torch
import torch.nn as nn
from model.modules.transformer_blocks import SoftSplit, HybridSpatialTemporalLayer, SoftComp


class STMambaBlock(nn.Module):
    """ CR branch: Detection-Guided Removal Block
            CD branch: Feedback Residual Detection Block """
    def __init__(self, dim, depths=8):
        super().__init__()
        self.dim = dim
        self.hidden_dim = dim * 2
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        t2t_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding
        }
        self.ss = SoftSplit(dim, self.hidden_dim, kernel_size, stride, padding, t2t_param=t2t_params)
        self.sc = SoftComp(dim, self.hidden_dim, kernel_size, stride, padding)

        self.depths = depths
        self.blocks = nn.ModuleList([
            HybridSpatialTemporalLayer(dim=self.hidden_dim, num_heads=4, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm) for _
            in range(depths)
        ])
        self.cd_add = FeedbackResidualDetection(dim, scales=(3, 5))

    def cd_router(self, cd_masks, kernel_size=(7, 7), stride=(3, 3), padding=(3, 3)):
        device = cd_masks.device
        B, T, _, H, W = cd_masks.shape
        ks_area = kernel_size[0] * kernel_size[1]

        unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding).to(device)
        unfold_mask = unfold(cd_masks.view(B * T, 1, H, W).float())
        K, F = unfold_mask.shape[1], unfold_mask.shape[2]
        unfold_mask = unfold_mask.view(B, T, K, F)

        self_clear_ratio = (unfold_mask == 0).sum(dim=2) / ks_area

        mask_self = torch.eye(T, device=device).bool().unsqueeze(0).unsqueeze(-1)
        ref_masks = unfold_mask.unsqueeze(1).expand(-1, T, -1, -1, -1)
        ref_masks = ref_masks.masked_fill(mask_self.unsqueeze(3), 1)
        ref_clear_ratio = (ref_masks == 0).sum(dim=3).max(dim=2).values / ks_area

        fully_recoverable = (ref_clear_ratio == 1.0)
        partially_recoverable = (ref_clear_ratio > 0) & (~fully_recoverable)
        fully_unrecoverable = (ref_clear_ratio == 0)
        self_more = self_clear_ratio >= ref_clear_ratio

        token_class = torch.full((B, T, F), -1, dtype=torch.long, device=device)
        token_class[fully_recoverable & self_more] = 0
        token_class[fully_recoverable & (~self_more)] = 1
        token_class[partially_recoverable & self_more] = 2
        token_class[partially_recoverable & (~self_more)] = 3
        token_class[fully_unrecoverable] = 4

        token_class = token_class.view(B * T, F)
        return token_class

    def forward(self, cr_feats, cd_masks=None):
        b, t, c, h, w = cr_feats.shape
        fold_output_size = (h, w)
        gate_feat = cr_feats * (1 - cd_masks)

        token_classes = self.cd_router(cd_masks)
        trans_feat = self.ss(gate_feat.view(b * t, -1, h, w), b, fold_output_size)
        for i in range(self.depths):
            trans_feat = self.blocks[i](trans_feat, token_classes)
        trans_feat = self.sc(trans_feat, t, fold_output_size)
        trans_feat = trans_feat.view(b, t, -1, h, w)
        out_feats = trans_feat + gate_feat

        cd_out = self.cd_add(cr_feats.view(-1, c, h, w), out_feats.view(-1, c, h, w))

        return out_feats, cd_out


class FeedbackResidualDetection(nn.Module):
    """ Feedback Residual Detection Block """
    def __init__(self, channels, scales=(3, 5)):
        super(FeedbackResidualDetection, self).__init__()
        self.scales = scales
        self.branchs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=ks, padding=ks//2, groups=channels),
                nn.LeakyReLU(0.2),
            ) for ks in scales
        ])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels//8, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels//8, len(scales), kernel_size=1, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, feat_in, feat_out):
        feat_in = self.norm1(feat_in.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feat_out = self.norm2(feat_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        diff_feats = feat_out - feat_in
        feats = [branch(diff_feats) for branch in self.branchs]
        fused_feats = sum(feats)
        weights = self.softmax(self.fc(self.global_pool(fused_feats))).view(-1, len(self.scales), 1, 1, 1)
        feats = torch.stack(feats, dim=1)
        fused_feats = (feats * weights).sum(dim=1)
        return fused_feats
