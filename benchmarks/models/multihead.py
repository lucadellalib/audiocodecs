# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
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
    C = 512
    K = 4

    embedding = MultiHeadEmbedding(C, H, K, padding_idx=True)
    linear = MultiHeadLinear(H, C, K)

    input = torch.randint(0, C + 1, size=(B, T, K))
    output = embedding(input)
    print(output.shape)

    output = output.sum(dim=-2)
    output = linear(output)
    print(output.shape)
