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


__all__ = ["LinearPooling"]


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


if __name__ == "__main__":
    x = torch.randn(2, 100, 3, 256)
    model = LinearPooling(3)
    y = model(x)
    print(x.shape)
    print(y.shape)
