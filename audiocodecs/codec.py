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

"""Codec interface."""

from abc import ABC, abstractmethod

import torch
import torchaudio


__all__ = ["Codec"]


# B: batch size
# T: sequence length in the time domain
# N: sequence length in the token domain
# C: vocabulary size (assuming that each codebook has the same number of tokens)
# K: number of codebooks
class Codec(torch.nn.Module, ABC):
    _MODES = ["encode", "decode", "reconstruct"]

    def __init__(self, sample_rate, orig_sample_rate, mode="reconstruct"):
        super().__init__()
        if mode not in self._MODES:
            raise ValueError(f"`mode` ({mode}) must be one of {self._MODES}")
        self.sample_rate = sample_rate
        self.orig_sample_rate = orig_sample_rate
        self.mode = mode
        self._logits = None

    def forward(self, input, length=None):
        if self.mode == "encode":
            toks = self.sig_to_toks(input, length)
            return toks
        if self.mode == "decode":
            sig = self.toks_to_sig(input, length)
            return sig
        if self.mode == "reconstruct":
            toks = self.sig_to_toks(input, length)
            sig = self.toks_to_sig(toks, length)
            return sig

    def sig_to_toks(self, sig, length=None):
        # sig: [B, T]
        sig = torchaudio.functional.resample(
            sig,
            self.sample_rate,
            self.orig_sample_rate,
        )
        if length is None:
            length = torch.ones(len(sig), device=sig.device)
        return self._sig_to_toks(sig, length)

    def sig_to_feats(self, sig, length=None):
        # sig: [B, T]
        sig = torchaudio.functional.resample(
            sig,
            self.sample_rate,
            self.orig_sample_rate,
        )
        if length is None:
            length = torch.ones(len(sig), device=sig.device)
        return self._sig_to_feats(sig, length)

    def sig_to_qfeats(self, sig, length=None):
        # sig: [B, T]
        sig = torchaudio.functional.resample(
            sig,
            self.sample_rate,
            self.orig_sample_rate,
        )
        if length is None:
            length = torch.ones(len(sig), device=sig.device)
        return self._sig_to_qfeats(sig, length)

    def toks_to_sig(self, toks, length=None):
        # toks: [B, N, K]
        if length is None:
            length = torch.ones(len(toks), device=toks.device)
        sig = self._toks_to_sig(toks, length)
        sig = torchaudio.functional.resample(
            sig,
            self.orig_sample_rate,
            self.sample_rate,
        )
        return sig

    def toks_to_qfeats(self, toks, length=None):
        # toks: [B, N, K]
        if length is None:
            length = torch.ones(len(toks), device=toks.device)
        qfeats = self._toks_to_qfeats(toks, length)
        return qfeats

    def feats_to_sig(self, feats, length=None):
        # toks: [B, N, H]
        if length is None:
            length = torch.ones(len(feats), device=feats.device)
        sig = self._feats_to_sig(feats, length)
        sig = torchaudio.functional.resample(
            sig,
            self.orig_sample_rate,
            self.sample_rate,
        )
        return sig

    def resample(self, toks, p=0.2, temp=1.0, top_k=None, top_p=None):
        # toks: [B, N, K]
        if p <= 0.0:
            return toks
        orig_toks = toks = toks.clone()
        num_codebooks = toks.shape[-1]
        toks = toks.flatten(end_dim=-2).T  # [K, BN]
        logits = self.logits()  # [K, C, C]
        vocab_size = logits.shape[-1]
        selected_logits = logits.gather(
            1, toks[..., None].expand(-1, -1, vocab_size)
        )  # [K, BN, C]
        selected_logits = selected_logits.flatten(end_dim=-2)  # [KBN, C]
        selected_probs = (selected_logits / temp).softmax(dim=-1)  # [KBN, C]
        if top_k is None and top_p is None:
            samples = selected_probs.multinomial(num_samples=1)  # [KBN]
        elif top_k is not None and top_p is None:
            samples = self._sample_top_k(selected_probs, top_k)  # [KBN]
        elif top_k is None and top_p is not None:
            samples = self._sample_top_p(selected_probs, top_p)  # [KBN]
        else:
            raise NotImplementedError
        samples = samples.reshape(num_codebooks, -1)  # [K, BN]
        samples = samples.T  # [BN, K]
        mask = torch.rand(orig_toks.shape) < p  # [B, N, K]
        orig_toks[mask] = samples.reshape_as(orig_toks)[mask]
        # toks: [B, N, K]
        return orig_toks

    @torch.no_grad()
    def logits(self):
        if self._logits is None:
            # Codebook pairwise logits
            embs = self.embs()  # [K, C, H]
            logits = -torch.cdist(embs, embs)  # [K, C, C]
            mask = torch.eye(logits.shape[-1]).bool().expand(len(logits), -1, -1)
            logits[mask] = -float("inf")
            self._logits = logits
        return self._logits.clone()

    def _sample_top_k(self, probs, k):
        # [B, C]
        probs, idx = probs.topk(k, dim=-1)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        samples = probs.multinomial(num_samples=1)
        samples = idx.gather(-1, samples)
        # [B]
        return samples[:, 0]

    def _sample_top_p(self, probs, p):
        # [B, C]
        probs, idx = probs.sort(dim=-1, descending=True)
        probs_sum = probs.cumsum(dim=-1)
        mask = probs_sum - probs > p
        probs[mask] = 0.0
        probs = probs / probs.sum(dim=-1, keepdim=True)
        samples = torch.multinomial(probs, num_samples=1)
        samples = idx.gather(-1, samples)
        # [B]
        return samples[:, 0]

    @abstractmethod
    def embs(self):
        raise NotImplementedError

    @abstractmethod
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        raise NotImplementedError

    @abstractmethod
    def _sig_to_feats(self, sig, length):
        # sig: [B, T]
        raise NotImplementedError

    @abstractmethod
    def _sig_to_qfeats(self, sig, length):
        # sig: [B, T]
        raise NotImplementedError

    @abstractmethod
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K]
        raise NotImplementedError

    # Optional
    def _toks_to_qfeats(self, toks, length):
        # toks: [B, N, K]
        raise NotImplementedError

    # Optional
    def _feats_to_sig(self, feats, length):
        # feats: [B, N, H]
        raise NotImplementedError
