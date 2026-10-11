import math

import torch
import torch.nn as nn

from model.layers import Conv2dWithConstraint


class Net(nn.Module):
    def __init__(self, config: dict):
        super(Net, self).__init__()
        F, C, T = config['batch_shape']
        self.backbone = ADFCNN(num_channels=C)

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


class ADFCNN(nn.Module):
    def __init__(self, num_channels: int, F1=8, D=1, F2='auto',
                 pool_mode='mean', drop_out=0.25):
        super(ADFCNN, self).__init__()

        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        if F2 == 'auto':
            F2 = F1 * D

        # Spectral
        self.spectral_1 = nn.Sequential(
            Conv2dWithConstraint(1, F1, kernel_size=[1, 125], padding='same', max_norm=2.),
            nn.BatchNorm2d(F1),
        )
        self.spectral_2 = nn.Sequential(
            Conv2dWithConstraint(1, F1, kernel_size=[1, 30], padding='same', max_norm=2.),
            nn.BatchNorm2d(F1),
        )

        # Spatial
        self.spatial_1 = nn.Sequential(
            Conv2dWithConstraint(F2, F2, (num_channels, 1), padding=0, groups=F2, bias=False, max_norm=2.),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.Dropout(drop_out),
            Conv2dWithConstraint(F2, F2, kernel_size=[1, 1], padding='valid',
                                 max_norm=2.),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            pooling_layer((1, 32), stride=32),
            nn.Dropout(drop_out),
        )

        self.spatial_2 = nn.Sequential(
            Conv2dWithConstraint(F2, F2, kernel_size=[num_channels, 1], padding='valid',
                                 max_norm=2.),
            nn.BatchNorm2d(F2),
            ActSquare(),
            pooling_layer((1, 75), stride=25),
            ActLog(),
            nn.Dropout(drop_out),
        )

        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(drop_out)
        self.w_q = nn.Linear(F2, F2)
        self.w_k = nn.Linear(F2, F2)
        self.w_v = nn.Linear(F2, F2)

    def forward(self, x):
        x_1 = self.spectral_1(x)
        x_2 = self.spectral_2(x)

        x_filter_1 = self.spatial_1(x_1)
        x_filter_2 = self.spatial_2(x_2)
        x_noattention = torch.cat((x_filter_1, x_filter_2), 3)
        B2, C2, H2, W2 = x_noattention.shape
        x_attention = x_noattention.reshape(B2, C2, H2 * W2).permute(0, 2, 1)  #### the last one is channel

        B, N, C = x_attention.shape

        q = self.w_q(x_attention).permute(0, 2, 1)
        k = self.w_k(x_attention).permute(0, 2, 1)
        v = self.w_v(x_attention).permute(0, 2, 1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        d_k = q.size(-1)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        # -------------------
        attn = attn.softmax(dim=-1)

        x = (attn @ v).reshape(B, N, C)

        x_attention = x_attention + self.drop(x)
        x_attention = x_attention.reshape(B2, H2, W2, C2).permute(0, 3, 1, 2)
        x = self.drop(x_attention)
        return x


class ActSquare(nn.Module):
    def __init__(self):
        super(ActSquare, self).__init__()
        pass

    def forward(self, x):
        return torch.square(x)


class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))
