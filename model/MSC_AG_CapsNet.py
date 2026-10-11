import math

import torch
from torch import nn
from torch.nn import functional as functional


class MultiScaleConv(nn.Module):
    """Parallel temporal/spatial filters, fusion and temporal refinement."""

    def __init__(self, in_features, n_channels, branch_filters=8,
                 depth_multiplier=2, feature_dim=32, dropout=0.25):
        super().__init__()
        branch_dim = branch_filters * depth_multiplier
        self.branches = nn.ModuleList()
        for kernel in (16, 32, 64):
            # Explicit asymmetric padding preserves T for even kernel sizes.
            self.branches.append(nn.Sequential(
                nn.ZeroPad2d((kernel // 2 - 1, kernel // 2, 0, 0)),
                nn.Conv2d(in_features, branch_filters, (1, kernel), bias=False),
                nn.BatchNorm2d(branch_filters),
                nn.Conv2d(branch_filters, branch_dim, (n_channels, 1),
                          groups=branch_filters, bias=False),
                nn.BatchNorm2d(branch_dim),
            ))

        self.fusion = nn.Sequential(
            nn.Conv2d(3 * branch_dim, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(feature_dim, feature_dim, (1, 16), bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ELU(),
            nn.AvgPool2d((1, 7)),
            nn.Dropout(dropout),
        )
        self.out_features = feature_dim

    def forward(self, x):
        # Each EEG channel (and each input feature plane) is standardized over
        # time within its trial. This has no train/test running statistics.
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(variance + 1e-6)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.fusion(x).squeeze(2).transpose(1, 2)  # (B, N, d)


class AttentionCapsuleClassifier(nn.Module):
    """Shared self-attention and position/class-specific capsule transforms."""

    def __init__(self, n_tokens, feature_dim, n_class, capsule_dim):
        super().__init__()
        self.n_tokens = n_tokens
        self.n_class = n_class
        self.query = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key = nn.Linear(feature_dim, feature_dim, bias=False)
        # W_ij in equation (10): separate transform for token i and class j.
        self.vote_weight = nn.Parameter(
            torch.empty(n_tokens, n_class, feature_dim, capsule_dim)
        )
        nn.init.xavier_uniform_(self.vote_weight.view(n_tokens * n_class,
                                                      feature_dim, capsule_dim))

    def forward(self, tokens):
        if tokens.shape[1] != self.n_tokens:
            raise ValueError(
                f"Expected {self.n_tokens} tokens, got {tokens.shape[1]}; "
                "use the T specified in config['batch_shape']."
            )
        attention = torch.softmax(
            self.query(tokens) @ self.key(tokens).transpose(-2, -1)
            / math.sqrt(tokens.shape[-1]), dim=-1
        )  # (B, N, N)
        votes = torch.einsum('bnd,nkdh->bnkh', tokens, self.vote_weight)
        combined = torch.einsum('bij,bjkh->bikh', attention, votes).mean(dim=1)
        squared_norm = combined.square().sum(dim=-1, keepdim=True)
        # Length of the squash(s) capsule is ||s||^2 / (1 + ||s||^2).
        lengths = squared_norm.squeeze(-1) / (1.0 + squared_norm.squeeze(-1))
        return lengths


class Net(nn.Sequential):
    """Drop-in model: Net({'batch_shape': (F, C, T), 'n_class': K}).

    Optional config keys: branch_filters=8, depth_multiplier=2,
    feature_dim=32, capsule_dim=16 for four classes / 24 for two,
    dropout=0.25. Create on CPU and call model.to(device) as usual.
    """

    def __init__(self, config: dict):
        super().__init__()
        in_features, n_channels, n_times = config['batch_shape']
        n_class = config['n_class']
        if min(in_features, n_channels, n_class) < 1 or n_times < 56:
            raise ValueError("batch_shape must be positive with T >= 56; n_class >= 1")

        self.backbone = MultiScaleConv(
            in_features, n_channels,
            branch_filters=config.get('branch_filters', 8),
            depth_multiplier=config.get('depth_multiplier', 2),
            feature_dim=config.get('feature_dim', 32),
            dropout=config.get('dropout', 0.25),
        )
        # Both temporal convolutions preserve length; pool(8), then pool(7).
        n_tokens = (n_times // 8) // 7
        self.classifier = AttentionCapsuleClassifier(
            n_tokens, self.backbone.out_features, n_class,
            config.get('capsule_dim', 24 if n_class == 2 else 16),
        )
        self.input_shape = (in_features, n_channels, n_times)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"Expected input (batch, {', '.join(map(str, self.input_shape))}), "
                f"got {tuple(x.shape)}"
            )
        return self.classifier(self.backbone(x))


class MarginLoss(nn.Module):
    """Paper's capsule margin loss; labels are integer class indices."""

    def __init__(self, positive=0.9, negative=0.1, absent_weight=0.5):
        super().__init__()
        self.positive = positive
        self.negative = negative
        self.absent_weight = absent_weight

    def forward(self, lengths, labels):
        targets = functional.one_hot(labels.long(), lengths.shape[1]).to(
            device=lengths.device, dtype=lengths.dtype
        )
        loss = (targets * functional.relu(self.positive - lengths).square()
                + self.absent_weight * (1 - targets)
                * functional.relu(lengths - self.negative).square())
        return loss.sum(dim=1).mean()


if __name__ == '__main__':
    config = {'batch_shape': (1, 22, 1125), 'n_class': 4}
    model = Net(config)
    x = torch.randn(2, *config['batch_shape'])
    output = model(x)
    loss = MarginLoss()(output, torch.tensor([0, 2]))
    loss.backward()
    print('scores:', output.shape, 'loss:', float(loss.detach()))
