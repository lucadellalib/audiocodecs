# ==============================================================================
# Copyright 2025 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Pooling layers."""

import torch
from torch import nn


__all__ = ["AttentionalPooling", "LinearPooling"]


class AttentionalPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim = hidden_dim or input_dim
        self.attn_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=False),
            nn.Softmax(dim=-2),
        )
        self.attn = None

    def forward(self, x):
        # (B, N, K, H)
        attn = self.attn_mlp(x).squeeze(dim=-1)  # (B, N, K)
        x = attn.unsqueeze(dim=-2) @ x  # (B, N, 1, K) @ (B, N, K, H) = (B, N, 1, H)
        x = x.movedim(-1, -2).squeeze(dim=-1)  # (B, N, H)
        self.attn = attn.detach()
        return x


class LinearPooling(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        if num_channels == 1:
            self.mlp = nn.Identity()
        else:
            self.mlp = nn.Linear(num_channels, 1, bias=False)

    def forward(self, x):
        # (B, N, K, H)
        x = x.movedim(-1, -2)
        # (B, N, H)
        x = self.mlp(x)[..., 0]
        return x


class WeightedPooling(nn.Module):
    def __init__(self, num_channels, channel_idx=None):
        super().__init__()
        self.num_channels = num_channels
        self.channel_idx = channel_idx
        if channel_idx is None:
            self.weight = nn.Parameter(torch.ones(num_channels))

    def forward(self, x):
        if self.channel_idx is not None:
            return x[..., self.channel_idx, :]
        # (B, N, H, K)
        x = x.movedim(-1, -2)
        weight = nn.functional.softmax(self.weight)
        # (B, N, H)
        x = (x * weight).sum(dim=-1)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 100, 3, 256)
    model = LinearPooling(3)
    y = model(x)
    print(x.shape)
    print(y.shape)

    x = torch.randn(2, 100, 3, 256)
    model = WeightedPooling(3)
    y = model(x)
    print(x.shape)
    print(y.shape)

    x = torch.randn(2, 100, 3, 256)
    model = AttentionalPooling(256, 256)
    y = model(x)
    print(y.shape)
    print(model.attn)
