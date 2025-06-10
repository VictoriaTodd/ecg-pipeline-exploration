import torch
import torch.nn as nn

class ToyCNN(nn.Module):
    def __init__(self, num_classes=11, input_channels=1):
        super(ToyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Output shape: (batch_size, 64, 1)
            nn.Flatten(),             # Shape: (batch_size, 64)
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch_size, seq_len)
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.classifier(x)
        return x
