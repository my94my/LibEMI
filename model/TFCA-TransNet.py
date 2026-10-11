import math

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from torch import nn


class Net(nn.Module):
    def __init__(self, config: dict):
        super(Net, self).__init__()
        F, C, T = config['batch_shape']
        self.backbone = EEGTransformer(num_channels=C,
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


class EEGTransformer(nn.Module):
    def __init__(self, heads=4,
                 emb_size=40,
                 depth=6,
                 eeg1_f1=20,
                 eeg1_kernel_size=64,
                 eeg1_D=2,
                 eeg1_pooling_size1=8,
                 eeg1_pooling_size2=8,
                 eeg1_dropout_rate=0.3,
                 num_channels=22,
                 **kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.flatten = nn.Flatten()

        self.cnn = BranchEEGNetTransformer(heads, depth, emb_size, number_channel=num_channels,
                                           f1=eeg1_f1,
                                           kernel_size=eeg1_kernel_size,
                                           D=eeg1_D,
                                           pooling_size1=eeg1_pooling_size1,
                                           pooling_size2=eeg1_pooling_size2,
                                           dropout_rate=eeg1_dropout_rate)

        self.position = PositioinalEncoding(emb_size, 100, dropout=0.1)
        self.trans = TransformerEncoder(heads, depth, emb_size)

        self.flatten = nn.Flatten()

    def forward(self, x):
        # FFT
        fft_result = torch.fft.fft(x, dim=-1)  # FFT along time axis
        amplitude = torch.abs(fft_result)  # amplitude
        phase = torch.angle(fft_result)  # phase

        x = torch.cat([x, amplitude, phase], dim=-1)

        cnn = self.cnn(x)
        # add label
        cnn = cnn * math.sqrt(self.emb_size)

        features = self.position(cnn)
        features = self.trans(features)

        out = cnn + features

        # features = self.cnn_output(features) #if
        # out = self.flatten(out)

        return out.unsqueeze(2)


class PatchEmbeddingCNN(nn.Module):
    def __init__(self, f1=16, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.3,
                 number_channel=22, emb_size=40):
        super().__init__()
        f2 = D * f1

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
        self.se1d = SELayer(40, reduction=2)

        self.projection = nn.Sequential(
            Rearrange('b e (h) -> b (h) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        _, _, _, T = x.shape
        n_split = T // 3
        time_feature = x[:, :, :, :n_split]  # time feature
        magnitude_feature = x[:, :, :, n_split:n_split * 2]  # magnitude feature
        phase_feature = x[:, :, :, n_split * 2:]  # phase feature
        # 2d
        x = torch.cat([time_feature, magnitude_feature, phase_feature], dim=1)
        x = self.cnn_cat(x)
        x = x.squeeze(2)

        x = self.se1d(x)
        x = self.projection(x)

        return x


# Channel Attention

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


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


# PointWise FFN
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


class BranchEEGNetTransformer(nn.Sequential):
    def __init__(self, heads=4,
                 depth=6,
                 emb_size=40,
                 number_channel=22,
                 f1=20,
                 kernel_size=64,
                 D=2,
                 pooling_size1=8,
                 pooling_size2=8,
                 dropout_rate=0.3,
                 **kwargs):
        super().__init__(
            PatchEmbeddingCNN(f1=f1,
                              kernel_size=kernel_size,
                              D=D,
                              pooling_size1=pooling_size1,
                              pooling_size2=pooling_size2,
                              dropout_rate=dropout_rate,
                              number_channel=number_channel,
                              emb_size=emb_size),
        )


class PositioinalEncoding(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))

    def forward(self, x):  # x-> [batch, embedding, length]
        x = x + self.encoding[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)
