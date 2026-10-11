import math

import torch
import torch.nn as nn


class Net(nn.Sequential):
    def __init__(self, config: dict):
        super(Net, self).__init__()
        F, C, T = config['batch_shape']
        self.backbone = TFCABiLSTM(n_channel=C)

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


class TimeFrequencyPlanes(nn.Module):
    """Full FFT amplitude/phase as in TFCA; normalization is trial-local."""

    def forward(self, x):
        # FFT in float32 also supports AMP and non-power-of-two trial lengths.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        spectrum = torch.fft.fft(work, dim=-1)
        planes = torch.cat((work, spectrum.abs(), spectrum.angle()), dim=1)
        mean = planes.mean(dim=(-2, -1), keepdim=True)
        variance = planes.var(dim=(-2, -1), keepdim=True, unbiased=False)
        planes = (planes - mean) * torch.rsqrt(variance + 1e-6)
        return planes.to(dtype=x.dtype)


class SELayer(nn.Module):

    def __init__(self, channel, reduction=2, batch_first=True):
        super(SELayer, self).__init__()

        self.batch_first = batch_first
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        input.size == output.size 
        '''
        if not self.batch_first:
            x = x.permute(1, 0, 2)

        b, c, _, = x.size()
        y = self.avg_pool(x).view(b, c)  # size = (batch,channel)

        y = self.fc(y).view(b, c, 1)  # size = (batch,channel,1,1)
        out = x * y.expand_as(x)  # size = (batch,channel,w,h)

        if not self.batch_first:
            out = out.permute(1, 0, 2)  # size = (channel,batch,w,h)

        return out


class PatchEmbeddingCNN(nn.Module):
    def __init__(self, f1=20, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.3,
                 number_channel=22, emb_size=40):
        super().__init__()
        f2 = D * f1
        self.features = TimeFrequencyPlanes()

        self.cnn_cat = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(3, f1, (1, kernel_size), (1, 1), padding='same', bias=False),
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False),  #
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 1
            nn.AvgPool2d((1, pooling_size1)),
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # average pooling 2 to adjust the length of feature into transformer encoder
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),
        )
        self.se1d = SELayer(f2, reduction=2)

    def forward(self, x):
        x = self.features(x)
        x = self.cnn_cat(x).squeeze(2)
        x = self.se1d(x)
        return x.transpose(1, 2)


class MultiHeadAttention(nn.Module):
    """Native-PyTorch equivalent of the TFCA attention, including its scale."""

    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        if emb_size % num_heads:
            raise ValueError('emb_size must be divisible by num_heads')
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        batch, length, width = x.shape

        def split(layer):
            return layer(x).reshape(batch, length, self.num_heads,
                                    width // self.num_heads).transpose(1, 2)

        q, k, v = split(self.queries), split(self.keys), split(self.values)
        attention = (q @ k.transpose(-2, -1)) / math.sqrt(self.emb_size)
        attention = self.att_drop(attention.softmax(dim=-1))
        out = (attention @ v).transpose(1, 2).reshape(batch, length, width)
        return self.projection(out)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)

        out = self.layernorm(self.drop(res) + x_input)
        return out


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                MultiHeadAttention(emb_size, num_heads, drop_p),
            ), emb_size, drop_p),
            ResidualAdd(nn.Sequential(
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
            ), emb_size, drop_p)

        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])


class PositionalEncoding(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        count = x.shape[1]
        if count <= self.encoding.shape[1]:
            positions = self.encoding[:, :count]
        else:
            positions = torch.nn.functional.interpolate(
                self.encoding.transpose(1, 2), size=count,
                mode='linear', align_corners=False,
            ).transpose(1, 2)
        return self.dropout(x + positions.to(dtype=x.dtype))


class ResidualBiLSTM(nn.Module):
    """ConvoReleNet-inspired sequence refinement; keep feature width at 40."""

    def __init__(self, feature_dim=40, dropout=0.1):
        super().__init__()
        if feature_dim % 2:
            raise ValueError('feature_dim must be even')
        self.rnn = nn.LSTM(feature_dim, feature_dim // 2, num_layers=1,
                           batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        refinement, _ = self.rnn(x)
        return self.norm(x + self.dropout(refinement))


class TFCABiLSTM(nn.Module):
    def __init__(self, n_channel, use_bilstm=True):
        super().__init__()
        self.n_channel = n_channel
        self.emb_size = 40
        self.cnn = PatchEmbeddingCNN(number_channel=n_channel)
        self.position = PositionalEncoding(self.emb_size, 100, dropout=0.1)
        self.trans = TransformerEncoder(heads=4, depth=6, emb_size=self.emb_size)
        self.refinement = ResidualBiLSTM(self.emb_size) if use_bilstm else nn.Identity()
        self.classification_dropout = nn.Dropout(0.5)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != self.n_channel:
            raise ValueError(f'Expected [B,1,{self.n_channel},T], got {tuple(x.shape)}')
        if x.shape[-1] < 64:
            raise ValueError('T must be >= 64 for the two temporal pools')
        cnn = self.cnn(x) * math.sqrt(self.emb_size)
        features = cnn + self.trans(self.position(cnn))
        features = self.refinement(features)
        features = self.classification_dropout(features)
        return features.transpose(1, 2).unsqueeze(2)


if __name__ == '__main__':
    for channels, classes in ((22, 4), (3, 2)):
        config = {'batch_shape': (1, channels, 1000), 'n_class': classes}
        model = Net(config).cuda().eval()
        x = torch.randn(2, *config['batch_shape'], device='cuda')
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, classes)
        assert torch.isfinite(logits).all()
        print(f'C={channels}: {tuple(logits.shape)}')
