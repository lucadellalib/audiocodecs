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

"""PAST (see https://arxiv.org/abs/2505.14470)."""

import os
import sys

import torch

from audiocodecs.codec import Codec


__all__ = ["PAST"]


class PAST(Codec):
    MODEL_NAMES = ["PAST", "PAST_streamable"]

    def __init__(
        self,
        sample_rate,
        mode="reconstruct",
        num_codebooks=8,
        model_name="PAST_streamable",
    ):
        try:
            # Workaround to avoid name collisions with installed modules
            root_dir = os.path.dirname(os.path.realpath(__file__))
            sys_path = [x for x in sys.path]
            sys.path = [x for x in sys.path if root_dir not in x]
            from past.models.past_model import PastModel

            sys.path = sys_path
        except ImportError:
            raise ImportError(
                "`pip install git+https://github.com/lucadellalib/PAST.git` to use this module"
            )

        super().__init__(sample_rate, 16000, mode)
        self.num_codebooks = num_codebooks
        self.vocab_size = 1024

        self.model = PastModel.from_pretrained(model_name)
        self.model.quantizer.n_q = num_codebooks
        if mode == "encode":
            self.model.decoder = None
        elif mode == "decode":
            self.model.encoder = None

    # override
    @torch.no_grad()
    def embs(self):
        layers = self.model.quantizer.vq.layers[: self.num_codebooks]
        embs = [layer.codebook for layer in layers]
        embs = torch.stack(embs)  # [K, C, H]
        return embs

    # override
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        toks = self.model.encode(sig[:, None])[0]
        toks = toks.movedim(-1, -2)
        return toks

    # override
    def _sig_to_feats(self, sig, length):
        # sig: [B, T]
        sig = sig[:, None]
        x = self.model.preprocess(sig)[0]
        feats = self.model.encoder(x)
        feats = feats.movedim(-1, -2)
        return feats

    # override
    def _sig_to_qfeats(self, sig, length):
        # sig: [B, T]
        toks = self.model.encode(sig[:, None])[0]
        qfeats = self.model.quantizer.decode(toks)
        qfeats = qfeats.movedim(-1, -2)
        return qfeats

    # override
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K]
        output = self.model.decode(toks.movedim(-1, -2))
        sig = output[:, 0]  # [B, T]
        return sig

    # override
    def _toks_to_qfeats(self, toks, length):
        # toks: [B, N, K]
        qfeats = self.model.quantizer.decode(toks.movedim(-1, -2))
        qfeats = qfeats.movedim(-1, -2)
        return qfeats


# Test
if __name__ == "__main__":
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 10000
    batch_size = 2
    num_codebooks = 8

    for mode in ["encode", "decode", "reconstruct"]:
        codec = (
            PAST(
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
    codec = PAST(sample_rate, num_codebooks=num_codebooks).eval()
    with torch.no_grad():
        rec_sig = codec(sig)
    torchaudio.save("reconstruction.wav", rec_sig, sample_rate)
