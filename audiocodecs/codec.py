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
        self._dist = None

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
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K]
        raise NotImplementedError
