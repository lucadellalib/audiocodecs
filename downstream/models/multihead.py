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

"""Multi-head layers."""

import torch


__all__ = [
    "MultiHeadEmbedding",
    "MultiHeadLinear",
]


class MultiHeadEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_codebooks,
        padding_idx=False,
        **kwargs,
    ):
        if isinstance(vocab_size, (list, tuple)):
            assert len(vocab_size) == num_codebooks, [len(vocab_size), num_codebooks]
            num_embeddings = torch.tensor(vocab_size).sum().item()
            self.offsets = torch.tensor([0] + vocab_size[:-1]).cumsum(dim=-1)
        else:
            num_embeddings = vocab_size * num_codebooks
            self.offsets = torch.arange(0, num_embeddings, vocab_size)
        if padding_idx:
            padding_idx = num_embeddings
            num_embeddings += 1
        else:
            padding_idx = None
        super().__init__(num_embeddings, embedding_dim, padding_idx, **kwargs)
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks

    def forward(self, input):
        offsets = self.offsets.to(input)
        output = input + offsets
        if self.padding_idx is not None:
            output[input == self.vocab_size] = self.padding_idx
        output = super().forward(output)
        return output


class MultiHeadLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        num_codebooks,
        **kwargs,
    ):
        if isinstance(out_features, (list, tuple)):
            assert len(out_features) == num_codebooks, [
                len(out_features),
                num_codebooks,
            ]
            total_out_features = torch.tensor(out_features).sum().item()
        else:
            total_out_features = out_features * num_codebooks
        super().__init__(in_features, total_out_features, **kwargs)
        self.num_codebooks = num_codebooks

    def forward(self, input):
        input_shape = input.shape
        output = super().forward(input)
        output = output.reshape(*input_shape[:-1], self.num_codebooks, -1)
        return output


if __name__ == "__main__":
    B = 2
    T = 10
    H = 64
    C = [512, 1024, 512, 1024]
    K = 4

    embedding = MultiHeadEmbedding(C, H, K, padding_idx=True)
    linear = MultiHeadLinear(H, C, K)

    input = torch.randint(0, min(C) + 1, size=(B, T, K))
    output = embedding(input)
    print(output.shape)

    output = output.sum(dim=-2)
    output = linear(output)
    print(output.shape)
