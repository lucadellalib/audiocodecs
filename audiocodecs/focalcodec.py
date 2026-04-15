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

"""FocalCodec (see https://arxiv.org/abs/2502.04465)."""

import torch
import torchaudio

from audiocodecs.codec import Codec


__all__ = ["FocalCodec"]


class FocalCodec(Codec):
    CONFIGS = [
        "lucadellalib/focalcodec_50hz",
        "lucadellalib/focalcodec_50hz_2k_causal",
        "lucadellalib/focalcodec_50hz_4k_causal",
        "lucadellalib/focalcodec_50hz_65k_causal",
        "lucadellalib/focalcodec_25hz",
        "lucadellalib/focalcodec_12_5hz",
    ]

    def __init__(
        self,
        sample_rate,
        num_codebooks=1,
        vocab_size=8192,
        mode="reconstruct",
        config="lucadellalib/focalcodec_50hz",
    ):
        try:
            import safetensors

        except ImportError:
            raise ImportError("`pip install safetensors` to use this module")

        super().__init__(sample_rate, 16000, mode)
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

        self.model = torch.hub.load(
            "lucadellalib/focalcodec",
            "focalcodec",
            config=config,
        )
        assert self.model.sample_rate_input == 16000
        assert num_codebooks == 1
        assert vocab_size == self.model.codebook.shape[0]

        self.sample_rate_input = self.model.sample_rate_input
        self.sample_rate_output = self.model.sample_rate_output

        if mode == "encode":
            self.model.decoder = None
        elif mode == "decode":
            self.model.encoder = None
            self.model.compressor = None

    # override
    @torch.no_grad()
    def embs(self):
        embs = self.model.codebook[None]
        return embs

    # override
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        toks = self.model.sig_to_toks(sig)  # [B, N]
        return toks[..., None]  # [B, N, 1]

    # override
    def _sig_to_feats(self, sig, length):
        # sig: [B, T]
        feats = self.model.sig_to_feats(sig)  # [B, N, H]
        return feats

    # override
    def _sig_to_qfeats(self, sig, length):
        # sig: [B, T]
        qfeats = self.model.sig_to_qfeats(sig)  # [B, N, H]
        return qfeats

    # override
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K=1]
        toks = toks[..., 0]
        sig = self.model.toks_to_sig(toks)
        if self.sample_rate_output != self.orig_sample_rate:
            sig = torchaudio.functional.resample(
                sig, self.sample_rate_output, self.orig_sample_rate
            )
        return sig

    # override
    def _toks_to_qfeats(self, toks, length):
        # toks: [B, N, K=1]
        qfeats = self.model.toks_to_qfeats(toks[..., 0])
        return qfeats

    # override
    def _feats_to_sig(self, feats, length):
        sig = self.model.feats_to_sig(feats)
        if self.sample_rate_output != self.orig_sample_rate:
            sig = torchaudio.functional.resample(
                sig, self.sample_rate_output, self.orig_sample_rate
            )
        return sig


# Test
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 10000
    batch_size = 2

    for mode in ["encode", "decode", "reconstruct"]:
        codec = FocalCodec(sample_rate, mode=mode).eval().to(device)
        input = (
            torch.zeros(batch_size, 10, 1).long()
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
    codec = FocalCodec(sample_rate).eval()
    with torch.no_grad():
        rec_sig = codec(sig)
    torchaudio.save("reconstruction.wav", rec_sig, sample_rate)
