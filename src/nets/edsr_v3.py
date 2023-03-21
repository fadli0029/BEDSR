import torch.nn as nn
from nets.common_layers import *

class Net(nn.Module):
    def __init__(
            self, 
            scale=2,
            kernel_size=3,
            n_feats=256,
            n_resblocks=32,
            res_scale=0.1,
            rgb_range=255.0,
            n_colors=3,
            conv=default_conv
        ):
        super(Net, self).__init__()

        # Downsampling scale of LR images.
        self.scale = scale

        # Neural network architecture settings.
        self.kernel_size = kernel_size
        self.n_feats = n_feats
        self.n_resblocks = n_resblocks
        self.res_scale = res_scale
        self.rgb_range = rgb_range
        self.n_colors = n_colors

        # Mean shifts layers
        self.sub_mean = MeanShift(self.rgb_range)
        self.add_mean = MeanShift(self.rgb_range, sign=1)

        self.act = nn.ReLU(True)

        # Head module
        m_head = [conv(self.n_colors, self.n_feats, self.kernel_size)]

        # Body module
        m_body = [
            ResBlock(
                conv=conv, 
                n_feats=self.n_feats, 
                kernel_size=self.kernel_size, 
                res_scale=self.res_scale,
                act=self.act
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(self.n_feats, self.n_feats, self.kernel_size))

        # Tail module
        m_tail = [
            conv(n_feats, n_feats * self.scale ** 2, 3),
            nn.PixelShuffle(self.scale),
            conv(self.n_feats, self.n_colors, self.kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
