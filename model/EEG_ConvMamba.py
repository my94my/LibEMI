import torch
import torch.nn as nn


class Net(nn.Sequential):
    def __init__(self, config: dict):
        super(Net, self).__init__()
        F, C, T = config['batch_shape']
        self.backbone = EEGConvMamba(n_channel=C,
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


class EEGConvMamba(nn.Module):
    """EEG-ConvMamba 2a feature backbone for the existing Conv2d classifier.

    Uses the authors' four temporal branches (kernel 15), 40-channel spatial
    filter, two temporal pools and four bidirectional V2 Mamba blocks. The
    original 1000->256->32->class FC head is deliberately left to Classifier.
    Requires the authors' mamba_ssm fork (bimamba_type='V2').
    """

    def __init__(self, n_channel: int, F1=10, F2=40, depth=4, drop_out=0.5):
        super(EEGConvMamba, self).__init__()
        from mamba_ssm.modules.mamba_simple import Mamba

        if F2 != 4 * F1:
            raise ValueError('F2 must be four times F1 for the four branches')

        self.temporal = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, F1, kernel_size=(1, 15)),
                nn.BatchNorm2d(F1),
            ) for _ in range(4)
        ])
        self.spatial = nn.Sequential(
            nn.Conv2d(F2, F2, kernel_size=(n_channel, 1)),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 75), (1, 12)),
            nn.Dropout(drop_out),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(F2, F2, kernel_size=(1, 1)),
            nn.AvgPool2d((1, 3)),
            nn.Dropout(drop_out),
        )

        # The authors configure four VisionMamba blocks with V2 bidirectional
        # mixers and RMS normalization. Preserve that mixer configuration while
        # exposing the sequence as a feature map for the unchanged Classifier.
        self.norms = nn.ModuleList([_RMSNorm(F2) for _ in range(depth)])
        self.mixers = nn.ModuleList([
            Mamba(d_model=F2, layer_idx=i, bimamba_type='V2',
                  if_devide_out=True)
            for i in range(depth)
        ])
        self.drop_paths = nn.ModuleList([
            _DropPath(0.3 * i / max(depth - 1, 1)) for i in range(depth)
        ])
        self.final_drop_path = _DropPath(0.3)
        self.final_norm = _RMSNorm(F2)

    def forward(self, x):
        if x.shape[1] != 1:
            raise ValueError('EEG-ConvMamba expects F=1 in batch_shape')
        x = torch.cat([branch(x) for branch in self.temporal], dim=1)
        x = self.projection(self.spatial(x))
        x = x.squeeze(2).transpose(1, 2)  # (batch, tokens, 40)
        hidden, residual = x, None
        for norm, mixer, drop_path in zip(self.norms, self.mixers, self.drop_paths):
            residual = hidden if residual is None else residual + drop_path(hidden)
            hidden = mixer(norm(residual))
        x = self.final_norm(residual + self.final_drop_path(hidden))
        return x.transpose(1, 2).unsqueeze(2)  # (batch, 40, 1, tokens)


class _RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class _DropPath(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1.0 - self.probability
        mask = (torch.rand((x.shape[0],) + (1,) * (x.ndim - 1),
                           device=x.device) < keep).to(x.dtype)
        return x * mask / keep


if __name__ == '__main__':
    m = Net(num_channels=22, num_classes=4, sampling_rate=2)
    a = torch.rand((16, 1, 22, 750))
    b = m(a)
    print(b.shape)
