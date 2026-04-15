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

"""MagiCodec (see https://arxiv.org/abs/2506.00385)."""

import os
import sys

import torch

from audiocodecs.codec import Codec


__all__ = ["MagiCodec"]


class MagiCodec(Codec):
    def __init__(
        self,
        sample_rate,
        mode="reconstruct",
        num_codebooks=1,
    ):
        try:
            # Workaround to avoid name collisions with installed modules
            root_dir = os.path.dirname(os.path.realpath(__file__))
            sys_path = [x for x in sys.path]
            sys.path = [x for x in sys.path if root_dir not in x]
            from magicodec.generator import Generator

            sys.path = sys_path
        except ImportError:
            raise ImportError(
                "`pip install git+https://github.com/lucadellalib/MagiCodec.git` to use this module"
            )

        super().__init__(sample_rate, 16000, mode)
        assert num_codebooks == 1
        self.num_codebooks = num_codebooks
        self.vocab_size = 131072

        self.model = Generator.from_pretrained()
        if mode == "encode":
            self.model.decoder = None
        elif mode == "decode":
            self.model.encoder = None

    # override
    @torch.no_grad()
    def embs(self):
        embs = self.model.quantizer.codebook.weight.clone()
        embs = embs[None]  # [K, C, H]
        return embs

    # override
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        toks = self.model.sig_to_toks(sig)
        toks = toks[..., None]  # [B, N, 1]
        return toks

    # override
    def _sig_to_feats(self, sig, length):
        # sig: [B, T]
        feats = self.model.sig_to_feats(sig)
        return feats

    # override
    def _sig_to_qfeats(self, sig, length):
        # sig: [B, T]
        qfeats = self.model.sig_to_qfeats(sig)
        return qfeats

    # override
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K]
        sig = self.model.toks_to_sig(toks[..., 0])
        return sig

    # override
    def _toks_to_qfeats(self, toks, length):
        # toks: [B, N, K=1]
        qfeats = self.model.toks_to_qfeats(toks[..., 0])
        return qfeats

    # override
    def _feats_to_sig(self, feats, length):
        sig = self.model.feats_to_sig(feats)[:, 0]
        return sig


# Test
if __name__ == "__main__":
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 10000
    batch_size = 2
    num_codebooks = 1

    for mode in ["encode", "decode", "reconstruct"]:
        codec = (
            MagiCodec(
                sample_rate,
                mode=mode,
                num_codebooks=num_codebooks,
            )
            .eval()
            .to(device)
        )
        input = (
            torch.zeros(batch_size, 10, num_codebooks).long()
            if mode == "decode"
            else torch.randn(batch_size, sample_rate)
        ).to(device)
        with torch.no_grad():
            output = codec(input)
            print(output.shape)
            embs = codec.embs()
            print(embs.shape)
            if mode in ["encode", "reconstruct"]:
                output = codec.sig_to_feats(input)
                print(output.shape)
                output = codec.sig_to_qfeats(input)
                print(output.shape)

    sig, sample_rate = torchaudio.load("example.wav")
    # Move to device as StableCodec does not support CPU
    sig = sig.to(device)
    codec = MagiCodec(sample_rate, num_codebooks=num_codebooks).eval().to(device)
    with torch.no_grad():
        rec_sig = codec(sig)
    torchaudio.save("reconstruction.wav", rec_sig.cpu(), sample_rate)
