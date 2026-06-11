import torch
import torch.nn as nn

from src.models.hda_1dcnn import DepthwiseSeparableConv1d


class VanillaMLP(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class Standard1DCNN(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.classifier(self.net(x).squeeze(-1))


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, num_classes=2, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


class MobileNet1D(nn.Module):
    def __init__(self, input_dim, num_classes=2, shared_dim=64):
        super().__init__()
        self.proj = nn.Linear(input_dim, shared_dim)
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            DepthwiseSeparableConv1d(16, 32, 3, 1),
            nn.MaxPool1d(2),
            DepthwiseSeparableConv1d(32, 64, 3, 1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.proj(x).unsqueeze(1)
        return self.classifier(self.backbone(x).squeeze(-1))


class LightTransformer(nn.Module):
    def __init__(self, input_dim, num_classes=2, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.proj(x).unsqueeze(1)
        x = self.encoder(x)
        return self.classifier(x[:, 0, :])


BASELINE_MODELS = {
    "MLP": VanillaMLP,
    "Standard1DCNN": Standard1DCNN,
    "LSTM": LSTMBaseline,
    "MobileNet1D": MobileNet1D,
    "LightTransformer": LightTransformer,
}
