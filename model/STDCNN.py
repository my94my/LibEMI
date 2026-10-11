import torch
import torch.nn as nn

from model.layers import Conv2dWithConstraint


class Net(nn.Module):
    def __init__(self, config: dict):
        super(Net, self).__init__()
        F, C, T = config['batch_shape']
        self.backbone = STDCNN(num_channels=C)

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


# SSM State-Space Model (Temporal Dynamics)
class StateSpaceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop=0.1):
        super(StateSpaceModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # learnable state transition matrix A
        self.A = nn.Parameter(torch.eye(hidden_dim, hidden_dim))
        # learnable input matrix B
        self.B = nn.Parameter(torch.zeros(input_dim, hidden_dim))
        # learnable output matrix C
        self.C = nn.Parameter(torch.eye(hidden_dim, output_dim))

        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        # Apply state-space model (Kalman filtering)
        b, l, d = x.shape  # b: batch, l: sequence length, d: input dimension

        # Process inputs and compute state updates
        state = torch.zeros(b, self.hidden_dim).to(x.device)  # Initial state
        states_list = []

        for t in range(l):
            # Compute the predicted state
            state = torch.matmul(state, self.A) + torch.matmul(x[:, t, :], self.B.T)
            # Compute the output using the observation matrix
            output = torch.matmul(state, self.C.T)  # Observation
            states_list.append(output)

        # Stack all states and outputs
        output_tensor = torch.stack(states_list, dim=1)
        output_tensor = self.dropout(output_tensor)
        return output_tensor


# STDCNN Backbone with Innovations
class STDCNN(nn.Module):
    def __init__(self,
                 num_channels=22,
                 F1=8, D=2, F2='auto', pool_mode='mean',
                 drop_out=0.25):
        super(STDCNN, self).__init__()

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

        # Spatial 1
        self.spatial_1 = nn.Sequential(
            Conv2dWithConstraint(F1, F2, (num_channels, 1), padding=0, groups=F1, bias=False, max_norm=2.),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.Dropout(drop_out),
            Conv2dWithConstraint(F2, F2, kernel_size=[1, 1], padding='valid', max_norm=2.),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            pooling_layer((1, 32), stride=32),
            nn.Dropout(drop_out),
        )

        # Spatial 2
        self.spatial_2 = nn.Sequential(
            Conv2dWithConstraint(F1, F2, kernel_size=[num_channels, 1], padding='valid', max_norm=2.),
            nn.BatchNorm2d(F2),
            ActSquare(),
            pooling_layer((1, 75), stride=25),
            ActLog(),
            nn.Dropout(drop_out),
        )

        # SSM State-Space Model (Temporal Dynamics)
        self.SSM = StateSpaceModel(input_dim=F2, hidden_dim=F2, output_dim=F2)

        # Transformer Encoder Layer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=F2, nhead=8, dim_feedforward=64),
            num_layers=2
        )
        self.classifier = nn.Linear(16 * 1 * 69, 4)

    def forward(self, x):
        x_1 = self.spectral_1(x)
        x_2 = self.spectral_2(x)
        # Extract spatial features
        x_filter_1 = self.spatial_1(x_1)
        x_filter_2 = self.spatial_2(x_2)

        # Combine the spatial features
        x_noattention = torch.cat((x_filter_1, x_filter_2), 3)
        B2, C2, H2, W2 = x_noattention.shape

        # Flatten for attention mechanism
        x_attention = x_noattention.reshape(B2, C2, H2 * W2).permute(0, 2, 1)

        # Apply SSM (Temporal Modeling)
        x_attention_SSM = self.SSM(x_attention)
        # Transformer Encoder Layer
        x_attention = self.transformer(x_attention_SSM)
        x_attention = x_attention.reshape(B2, W2, 1, C2).permute(0, 3, 2, 1)

        return x_attention


# Main Network with all Innovations
class NetInnovative(nn.Module):
    def __init__(self, num_classes=4, num_channels=22):
        super(NetInnovative, self).__init__()

        self.backbone = STDCNN(num_channels=num_channels)

        self.classifier = classifier(num_classes)

    def forward(self, x):
        # Apply STDCNN backbone and temporal modeling
        x = self.backbone(x)

        # Classification
        x = self.classifier(x)

        return x


# Activation functions
class ActSquare(nn.Module):
    def __init__(self):
        super(ActSquare, self).__init__()

    def forward(self, x):
        return torch.square(x)


class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))


if __name__ == "__main__":
    backbone = STDCNN(num_channels=22)

    classifier = classifier(4)

    x = torch.randn(16, 1, 22, 1000)
    x = backbone(x)
    x = classifier(x)
    print(x.shape)
