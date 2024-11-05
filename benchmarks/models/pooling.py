# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Pooling layers."""

import torch
from torch import nn


__all__ = ["LinearPooling"]


class LinearPooling(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.mlp = nn.Linear(num_channels, 1)

    def forward(self, x):
        # (B, N, K, H)
        x = x.movedim(-1, -2)
        # (B, N, H, K)
        x = self.mlp(x)[..., 0]
        return x


if __name__ == "__main__":
    x = torch.randn(2, 100, 3, 256)
    model = LinearPooling(3)
    y = model(x)
    print(x.shape)
    print(y.shape)
