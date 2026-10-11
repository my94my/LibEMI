import torch
import torch.nn as nn

from model.layers import Conv2dWithConstraint


class Net(nn.Sequential):
    def __init__(self, config: dict):
        super(Net, self).__init__()
        F, C, T = config['batch_shape']
        self.backbone = DeepConvNet(n_channel=C)

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


class DeepConvNet(nn.Module):
    def __init__(
            self,
            n_channel: int,
            first_conv_length=10,
            block_out_channels=None,
            pool_size=3,
    ):
        super(DeepConvNet, self).__init__()
        if block_out_channels is None:
            block_out_channels = [25, 25, 50, 100, 200]
        self.first_conv_block = nn.Sequential(
            Conv2dWithConstraint(1, block_out_channels[0], kernel_size=(1, first_conv_length), max_norm=2),
            Conv2dWithConstraint(block_out_channels[0], block_out_channels[1], kernel_size=(n_channel, 1),
                                 bias=False,
                                 max_norm=2),
            nn.BatchNorm2d(block_out_channels[1]),
            nn.ELU(),
            nn.MaxPool2d((1, pool_size))
        )
        self.deep_block = nn.ModuleList(
            [self.default_block(block_out_channels[i - 1], block_out_channels[i], first_conv_length, pool_size) for i in
             range(2, 5)]
        )
        self.flatten = nn.Flatten()

    def default_block(self, in_channels, out_channels, T, P):
        default_block = nn.Sequential(
            nn.Dropout(0.5),
            Conv2dWithConstraint(in_channels, out_channels, (1, T), bias=False, max_norm=2),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.MaxPool2d((1, P))
        )

        return default_block

    def forward(self, x):
        out = self.first_conv_block(x)
        for block in self.deep_block:
            out = block(out)
        return out


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
