# --------------------------------------------------------
# IFNet
# Written by Jiaheng Wang
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class Net(nn.Sequential):
    def __init__(self, config: dict):
        super(Net, self).__init__()
        F, C, T = config['batch_shape']

        self.backbone = IFNet(in_planes=C, out_planes=64, kernel_size=63, patch_size=125)

        self.backbone.cuda().eval()
        with torch.no_grad():
            dummy_input = torch.randn(2, F, C, T, device='cuda')
            output_shape = self.backbone(dummy_input).shape
            in_channels = output_shape[1]
            kernel_size = output_shape[2:]
        self.backbone.train()

        self.classifier = Classifier(in_channels, config['n_class'], kernel_size)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class Classifier(nn.Module):
    def __init__(self, in_channels, n_class, kernel_size):
        super(Classifier, self).__init__()

        self.dense = nn.Conv2d(in_channels=in_channels, out_channels=n_class, kernel_size=kernel_size)

    def forward(self, x):
        x = self.dense(x)
        return x.squeeze(3).squeeze(2)


class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        nn.Module.__init__(self)
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class LogPowerLayer(nn.Module):
    def __init__(self, dim):
        super(LogPowerLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(torch.mean(x ** 2, dim=self.dim), 1e-4, 1e4))
        # return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=False), 1e-4, 1e4))


class InterFre(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        out = sum(x)
        out = F.gelu(out)
        return out


class Stem(nn.Module):
    def __init__(self, in_planes: int, out_planes=64, kernel_size=63, patch_size=125):
        nn.Module.__init__(self)
        self.radix = 1
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = out_planes * self.radix
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.sconv = Conv(nn.Conv1d(self.in_planes, self.mid_planes, 1, bias=False, groups=self.radix),
                          bn=nn.BatchNorm1d(self.mid_planes), activation=None)

        self.tconv = nn.ModuleList()
        for _ in range(self.radix):
            self.tconv.append(Conv(nn.Conv1d(self.out_planes, self.out_planes, kernel_size, 1, groups=self.out_planes,
                                             padding=kernel_size // 2, bias=False, ),
                                   bn=nn.BatchNorm1d(self.out_planes), activation=None))
            kernel_size //= 2

        self.interFre = InterFre()

        self.power = LogPowerLayer(dim=3)
        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        N, C, T = x.shape
        out = self.sconv(x)

        out = torch.split(out, self.out_planes, dim=1)
        out = [m(x) for x, m in zip(out, self.tconv)]

        out = self.interFre(out)
        out = out.reshape(N, self.out_planes, T // self.patch_size, self.patch_size)
        out = self.power(out)
        out = self.dp(out)
        return out


class IFNet(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size, patch_size):
        r'''Interactive Frequency Convolutional Neural Network V2

        :param in_planes: Number of input EEG channels
        :param out_planes: Number of output feature dimensions
        :param kernel_size: Temporal convolution kernel size
        :param radix:   Number of input frequency bands
        :param patch_size: Temporal pooling size
        :param time_points: Input window length
        :param num_classes: Number of classes
        '''
        nn.Module.__init__(self)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stem = Stem(self.in_planes, self.out_planes, kernel_size, patch_size=patch_size)

        # self.fc = nn.Sequential(
        #     LinearWithConstraint(out_planes * (time_points // patch_size), num_classes, doWeightNorm=True),
        # )
        # print(f'fc layer feature dims:{self.fc[-1].weight.shape}')
        self.apply(self.initParms)

    def initParms(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        # x = x[:, :, :750]
        x = self.stem(x)
        # out = self.fc(out.flatten(1))
        # x = torch.flatten(out, start_dim=1)
        return x.unsqueeze(2)
