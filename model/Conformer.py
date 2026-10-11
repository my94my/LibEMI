import torch
import torch.nn as nn

from model.layers import Conv2dWithConstraint


class Net(nn.Module):
    def __init__(self, config: dict):
        super(Net, self).__init__()
        F, C, T = config['batch_shape']
        self.backbone = Conformer(num_channels=C,
                                  sampling_rate=config['sampling_rate'])

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


class Conformer(nn.Module):
    def __init__(
            self,
            num_channels: int,
            sampling_rate: int,
            F1=40,
            F2=40,
            drop_out=0.25,
            pool_mode='mean',
    ):
        super(Conformer, self).__init__()

        kernel_size = int(sampling_rate * 0.12)
        pooling_size = 0.3
        hop_size = 0.7
        pooling_kernel_size = int(sampling_rate * pooling_size)
        pooling_stride_size = int(sampling_rate * pooling_size * (1 - hop_size))

        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        self.net = nn.Sequential(
            Conv2dWithConstraint(1, F1, (1, kernel_size), padding='same', max_norm=2.),
            Conv2dWithConstraint(F1, F2, (num_channels, 1), padding='valid', max_norm=2.),
            nn.BatchNorm2d(F2),
            ActSquare(),
            pooling_layer((1, pooling_kernel_size), (1, pooling_stride_size)),
            ActLog(),
            nn.Dropout(drop_out),
            # nn.Flatten(),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=F2, nhead=8, dim_feedforward=64),
            num_layers=2
        )

    def forward(self, x):
        x = self.net(x)
        x = torch.squeeze(x, dim=2)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        B2, W2, C2 = x.shape
        H2 = 1
        x = x.reshape(B2, H2, W2, C2).permute(0, 3, 1, 2)
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
