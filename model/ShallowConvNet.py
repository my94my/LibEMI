import torch
import torch.nn as nn

from model.layers import Conv2dWithConstraint


class Net(nn.Module):
    def __init__(self, config: dict):
        super(Net, self).__init__()
        F, C, T = config['batch_shape']
        self.backbone = ShallowConvNet(n_channel=C,
                                       sampling_rate=config['sampling_rate'],
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


class ShallowConvNet(nn.Module):
    def __init__(
            self,
            n_channel: int,
            sampling_rate: int,
            F1=40,
            T1=25,
            F2=40,
            P1_T=75,
            P1_S=15,
            drop_out=0.25,
            pool_mode='mean',
    ):
        super(ShallowConvNet, self).__init__()

        kernel_size = int(sampling_rate * 0.12)
        pooling_size = 0.3
        hop_size = 0.7
        pooling_kernel_size = int(sampling_rate * pooling_size)
        pooling_stride_size = int(sampling_rate * pooling_size * (1 - hop_size))

        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        self.net = nn.Sequential(
            Conv2dWithConstraint(1, F1, (1, kernel_size), padding='same', max_norm=2.),
            Conv2dWithConstraint(F1, F2, (n_channel, 1), padding='valid', max_norm=2.),
            nn.BatchNorm2d(F2),
            ActSquare(),
            pooling_layer((1, pooling_kernel_size), (1, pooling_stride_size)),
            ActLog(),
            nn.Dropout(drop_out),
            # nn.Flatten(),
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.net(x)
        # output = self.flatten(x)
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
