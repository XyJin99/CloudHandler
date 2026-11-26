import torch
import torch.nn as nn
from timm.layers import trunc_normal_
import math
from model.modules.spectral_norm import spectral_norm as _spectral_norm
from torch.nn import functional as F

from model.modules.PTD import PyramidTemporalRegroupDetection as PTD
from model.modules.DGR_FRD_m import STMambaBlock


class InpaintGenerator(nn.Module):
    def __init__(self, channels=64):
        """ CloudHandler Generator """
        super(InpaintGenerator, self).__init__()
        self.cd_enc = Encoder(channels=channels)
        self.CDB0 = PTD(embed_dims=[channels, channels * 2, channels * 4, channels * 8])
        self.cd_gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, 1, 1, 1, 0),
        )
        self.cd_fuse = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0)
        self.cd_project = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.cd_saliency_pred = nn.Conv2d(channels // 4, 1, kernel_size=1)
        for module in self.get_cd_modules():
            module.apply(_init_weights)

        self.cr_enc = Encoder(channels=channels)
        self.HCRB = STMambaBlock(channels, depths=8)
        self.cr_removal_out = nn.Sequential(
            deconv(channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        for module in self.get_cr_modules():
            module.apply(_init_weights)

    def get_cr_modules(self):
        return [self.cr_enc, self.HCRB, self.cr_removal_out]

    def get_cd_modules(self):
        return [self.cd_enc, self.CDB0, self.cd_gate, self.cd_fuse, self.cd_project, self.cd_saliency_pred]

    def get_cr_params(self):
        return [p for layer in self.get_cr_modules() for p in layer.parameters()]

    def get_cd_params(self):
        return [p for layer in self.get_cd_modules() for p in layer.parameters()]

    def forward(self, frames, num_ref=5):
        b, t, c, h, w = frames.shape
        cr_feats = self.cr_enc(frames.view(b * t, -1, h, w))
        cr_feats = cr_feats.view(b, t, -1, cr_feats.shape[2], cr_feats.shape[3])
        _, _, c, fh, fw = cr_feats.shape

        cd_feats = self.cd_enc(frames.view(b * t, -1, h, w))
        cd_feats = cd_feats.view(b, t, -1, cd_feats.shape[2], cd_feats.shape[3])
        cd_feats = self.CDB0(cd_feats, num_ref=num_ref)
        gate_masks = self.cd_gate(cd_feats)

        masks = torch.sigmoid(gate_masks)
        cd_masks = (masks >= 0.5).float()
        cd_masks = cd_masks.view(b, t, -1, fh, fw)
        cr_feats, cd_res_feats = self.HCRB(cr_feats, cd_masks)

        cd_out = self.cd_fuse(cd_feats + cd_res_feats * (1 - cd_masks.view(b * t, -1, fh, fw)))

        _c = F.interpolate(cd_out, size=cd_out.shape[-1] * 2, mode='bilinear', align_corners=False)
        _c = self.cd_project(_c)
        _c = F.interpolate(_c, size=_c.shape[-1] * 2, mode='bilinear', align_corners=False)
        cd_out = self.cd_saliency_pred(_c)

        cr_out = cr_feats.view(-1, cr_feats.shape[2], fh, fw)
        cr_out = torch.tanh(self.cr_removal_out(cr_out))
        return cd_out, cr_out, gate_masks

def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


class Encoder(nn.Module):
    def __init__(self, channels=64):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.neck = nn.Sequential(
            nn.Conv2d(128, channels, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        bt, c, _, _ = x.size()
        out = x
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = out
                _, _, h, w = x0.size()
            if i > 8 and i % 2 == 0:
                g = self.group[(i - 8) // 2]
                x = x0.view(bt, g, -1, h, w)
                o = out.view(bt, g, -1, h, w)
                out = torch.cat([x, o], 2).view(bt, -1, h, w)
            out = layer(out)
        out = self.neck(out)
        return out


class deconv(nn.Module):
    def __init__(self, input_channel, output_channel,  kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,  output_channel, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x,  scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv(x)

# ######################################################################
#  Discriminator for Temporal Patch GAN
# ######################################################################

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            'Network [%s] was created. Total number of parameters: %.1f million. '
            'To see the architecture, do print(network).' %
            (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1
                                           or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' %
                        init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class Discriminator(BaseNetwork):
    def __init__(self,
                 in_channels=3,
                 use_sigmoid=False,
                 use_spectral_norm=True,
                 init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels,
                          out_channels=nf * 1,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=1,
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 1,
                          nf * 2,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 2,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4,
                      nf * 4,
                      kernel_size=(3, 5, 5),
                      stride=(1, 2, 2),
                      padding=(1, 2, 2)))

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        xs_t = torch.transpose(xs, 1, 2)
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)
        return out


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module


if __name__ == '__main__':
    import os
    import torch.backends.cudnn as cudnn
    import numpy as np
    import random


    def set_seed(seed=10):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        if not torch.cuda.is_available():
            raise Exception('No Gpu found, please run with gpu')
        else:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            cudnn.benchmark = True

    set_seed(seed=1234)
    input = torch.rand(2, 8, 3, 512, 512).cuda()
    modelB = InpaintGenerator().cuda()
    cd_out, cr_out, sup_cd = modelB(input, num_ref=5)
    print(cd_out.shape, cr_out.shape, sup_cd.shape)



