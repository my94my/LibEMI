import torch
import torch.nn as nn

from model.layers import Conv2dWithConstraint


class Net(nn.Sequential):
    def __init__(self, config: dict):
        super(Net, self).__init__()
        F, C, T = config['batch_shape']
        self.backbone = EEGNet(n_channel=C,
                               )

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


class EEGNet(nn.Module):
    def __init__(self,
                 n_channel: int,
                 F1=8, D=2, F2='auto', T1=64, T2=16, P1=4, P2=8, pool_mode='mean',
                 drop_out=0.25):
        super(EEGNet, self).__init__()

        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        if F2 == 'auto':
            F2 = F1 * D

        # Spectral
        self.spectral = nn.Sequential(
            nn.Conv2d(1, F1, (1, T1), padding=(0, T1 // 2), bias=False),
            nn.BatchNorm2d(F1))

        # Spatial
        self.spatial = nn.Sequential(
            Conv2dWithConstraint(F1, F1 * D, (n_channel, 1), padding=0, groups=F1, bias=False, max_norm=1),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            pooling_layer((1, P1), stride=4),
            nn.Dropout(drop_out)
        )
        # Temporal
        self.temporal = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, T2), padding=(0, T2 // 2), groups=F1 * D),
            nn.Conv2d(F2, F2, 1, stride=1, bias=False, padding=0),
            nn.BatchNorm2d(F2),
            # ActSquare(),
            nn.ELU(),
            pooling_layer((1, P2), stride=8),
            # ActLog(),
            nn.Dropout(drop_out)
        )
        self.flatten = nn.Flatten()
        self.bn = nn.BatchNorm2d(F2)

    def forward(self, x):
        x = self.spectral(x)
        x = self.spatial(x)
        x = self.temporal(x)
        # x = self.flatten(x)

        return x


if __name__ == '__main__':
    m = Net(num_channels=22, num_classes=4, sampling_rate=2)
    a = torch.rand((16, 1, 22, 750))
    b = m(a)
    print(b.shape)
