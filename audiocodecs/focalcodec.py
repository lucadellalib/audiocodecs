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

"""FocalCodec."""

import torch

from audiocodecs.codec import Codec


__all__ = ["FocalCodec"]


class FocalCodec(Codec):
    CONFIGS = [
        "lucadellalib/focalcodec_50hz",
        "lucadellalib/focalcodec_25hz",
        "lucadellalib/focalcodec_12_5hz",
    ]

    def __init__(
        self,
        sample_rate,
        num_codebooks=1,
        vocab_size=8192,
        mode="reconstruct",
        config="lucadellalib/focalcodec/LibriTTS960_50Hz",
    ):
        try:
            import safetensors

        except ImportError:
            raise ImportError("`pip install safetensors` to use this module")

        super().__init__(sample_rate, 16000, mode)
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        assert num_codebooks == 1, num_codebooks
        assert vocab_size == 8192, vocab_size

        self.model = torch.hub.load(
            "lucadellalib/focalcodec", "focalcodec", config=config
        )

        if mode == "encode":
            self.model.decompressor = None
            self.model.decoder = None
        elif mode == "decode":
            self.model.encoder = None
            self.model.compressor = None

    # override
    @torch.no_grad()
    def embs(self):
        embs = self.model.codebook
        return embs

    # override
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        toks = self.model.sig_to_toks(sig, length)  # [B, N]
        return toks[..., None]  # [B, N, 1]

    # override
    def _sig_to_feats(self, sig, length):
        # sig: [B, T]
        feats = self.model.sig_to_lats(sig, length)  # [B, N, H]
        return feats

    # override
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K=1]
        toks = toks[..., 0]
        sig = self.model.toks_to_sig(toks)
        return sig


# Test
if __name__ == "__main__":
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 10000
    batch_size = 2

    for mode in ["encode", "decode", "reconstruct"]:
        codec = (
            FocalCodec(
                sample_rate,
                mode=mode,
            )
            .eval()
            .to(device)
        )
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

    sig, sample_rate = torchaudio.load("example.wav")
    codec = FocalCodec(sample_rate).eval()
    with torch.no_grad():
        rec_sig = codec(sig)
    torchaudio.save("reconstruction.wav", rec_sig, sample_rate)
