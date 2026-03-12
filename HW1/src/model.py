import torch
from torch import nn

from src.melbanks import LogMelFilterBanks


class SpeechCommandModel(nn.Module):
    def __init__(self, n_mels: int = 80, groups: int = 1):
        """
        n_mels  — число mel-фильтров
        groups  — параметр групповой свёртки
        """

        super().__init__()

        # feature extraction
        self.mel = LogMelFilterBanks(n_mels=n_mels)

        # количество каналов в CNN
        c1, c2, c3 = 32, 64, 64

        # проверка корректности групповой свёртки
        assert n_mels % groups == 0
        assert c1 % groups == 0
        assert c2 % groups == 0
        assert c3 % groups == 0

        # CNN encoder
        self.encoder = nn.Sequential(

            nn.Conv1d(
                in_channels=n_mels,
                out_channels=c1,
                kernel_size=5,
                padding=2,
                groups=groups
            ),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=c1,
                out_channels=c2,
                kernel_size=5,
                padding=2,
                groups=groups
            ),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=c2,
                out_channels=c3,
                kernel_size=3,
                padding=1,
                groups=groups
            ),
            nn.BatchNorm1d(c3),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        # classifier
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x):

        # x shape
        # (batch, time)

        x = self.mel(x)

        # (batch, n_mels, frames)

        x = self.encoder(x)

        # (batch, channels, 1)

        x = self.head(x)

        # (batch, 2)

        return x