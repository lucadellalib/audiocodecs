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

"""Codebook utilization."""

import math

import torch
from speechbrain.utils.metric_stats import MetricStats


__all__ = ["CodebookUtil"]


class CodebookUtil(MetricStats):
    def __init__(self, num_codebooks, vocab_size):
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.vocab_sizes = [vocab_size] * num_codebooks

        self.toks_count_per_codebook = [torch.zeros(v) for v in self.vocab_sizes]
        self.total_toks = 0
        self.clear()

    @torch.no_grad()
    def append(self, hyp_toks, lens=None):
        assert hyp_toks.ndim == 3
        assert hyp_toks.shape[0] == 1, "Batch size must be 1"

        # [B, N, K]
        for k in range(hyp_toks.shape[-1]):
            idxes, counts = hyp_toks[..., k].unique(return_counts=True)
            idxes, counts = idxes.cpu(), counts.cpu()
            self.toks_count_per_codebook[k][idxes] += counts
        self.total_toks += hyp_toks.shape[:2].numel()

    def summarize(self, field=None):
        codebook_util_per_codebook = []
        norm_entropy_per_codebook = []

        for counts, vocab_size in zip(self.toks_count_per_codebook, self.vocab_sizes):
            probs = counts / self.total_toks
            valid_mask = probs > 0
            valid_probs = probs[valid_mask]

            entropy = -(valid_probs * valid_probs.log2()).sum()
            valid_vocab_size = valid_mask.sum()

            if valid_vocab_size > 1:
                codebook_util = valid_vocab_size / vocab_size
                norm_entropy_valid = entropy / math.log2(valid_vocab_size)
            else:
                codebook_util = 0
                norm_entropy_valid = 0.0

            codebook_util_per_codebook.append(codebook_util)
            norm_entropy_per_codebook.append(norm_entropy_valid)

        codebook_util = sum(codebook_util_per_codebook) / len(
            codebook_util_per_codebook
        )
        norm_entropy = sum(norm_entropy_per_codebook) / len(norm_entropy_per_codebook)

        self.summary = {}
        self.summary["codebook_util"] = round(
            100 * torch.tensor(codebook_util).item(), 2
        )
        self.summary["norm_entropy"] = round(100 * torch.tensor(norm_entropy).item(), 2)

        if field is not None:
            return self.summary[field]
        else:
            return self.summary


if __name__ == "__main__":
    batch_size = 1
    seq_length = 10
    num_codebook = 8
    vocab_size = 1024
    hyp_toks = torch.randint(0, vocab_size, size=(batch_size, seq_length, num_codebook))
    norm_entropy = CodebookUtil(num_codebook, vocab_size)
    norm_entropy.append(hyp_toks)
    print(norm_entropy.summarize("codebook_util"))
    print(norm_entropy.summarize("norm_entropy"))
