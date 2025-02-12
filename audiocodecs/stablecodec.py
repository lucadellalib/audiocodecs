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

"""Stable Codec (see https://arxiv.org/abs/2411.19842)."""

import os
import sys

import torch

from audiocodecs.codec import Codec


__all__ = ["StableCodec"]


class StableCodec(Codec):
    SOURCES = ["stabilityai/stable-codec-speech-16k"]
    NUM_CODEBOOKS = [1, 2, 4]
    VOCAB_SIZES = [46656, 15625, 729]
    CONFIGS = {
        (1, 46656): "1x46656_400bps",
        (2, 15625): "2x15625_700bps",
        (4, 729): "4x729_1000bps",
    }

    def __init__(
        self,
        sample_rate,
        mode="reconstruct",
        source="stabilityai/stable-codec-speech-16k",
        num_codebooks=2,
        vocab_size=15625,
    ):
        try:
            # Workaround to avoid name collisions with installed modules
            root_dir = os.path.dirname(os.path.realpath(__file__))
            sys_path = [x for x in sys.path]
            sys.path = [x for x in sys.path if root_dir not in x]
            import stable_codec

            sys.path = sys_path
        except ImportError:
            raise ImportError(
                "pip install git+https://github.com/lucadellalib/stable-codec.git@main#egg=stable_codec` to use this module"
            )

        super().__init__(sample_rate, 16000, mode)
        assert num_codebooks in self.NUM_CODEBOOKS
        assert vocab_size in self.VOCAB_SIZES
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

        self.model = stable_codec.StableCodec(pretrained_model=source)
        self.model.set_posthoc_bottleneck(self.CONFIGS[(num_codebooks, vocab_size)])

        if mode == "encode":
            self.model.decoder = None
        elif mode == "decode":
            self.model.encoder = None

    # override
    @torch.no_grad()
    def embs(self):
        device = next(iter(self.model.state_dict().values())).device
        toks = torch.arange(self.vocab_size, device=device)
        toks = (
            toks[:, None, None].expand(-1, -1, self.num_codebooks).clone()
        )  # [C, 1, K]
        embs = [
            quantizer.indices_to_codes(toks[..., k, None])
            for k, quantizer in enumerate(self.model.residual_fsq.quantizers)
        ]
        embs = torch.cat(embs, dim=1)
        embs = embs.movedim(0, 1)  # [K, C, H]
        return embs

    # override
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        # Skip normalize: it slows down inference and I did not observe any improvement on LibriSpeech test resynthesis
        # sig = torch.cat([self.model.volume_norm(x[None]) for x in sig])
        # Length must be multiple of window_size
        window_size = 320
        if sig.shape[-1] % window_size != 0:
            sig = torch.nn.functional.pad(
                sig, [0, window_size - sig.shape[-1] % window_size]
            )
        _, toks = self.model.encode(
            sig[:, None], posthoc_bottleneck=True
        )  # K x [B, N, 1]
        toks = torch.cat(toks, dim=-1)  # [B, N, K]
        return toks

    # override
    def _sig_to_feats(self, sig, length):
        # sig: [B, T]
        pre_bottleneck_latents, _ = self.model.encode(
            sig[:, None], posthoc_bottleneck=True
        )  # [B, H, N]
        pre_bottleneck_latents = pre_bottleneck_latents.movedim(-1, -2)  # [B, N, H]
        return pre_bottleneck_latents

    # override
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K]
        sig = self.model.decode(
            toks[..., None, :].unbind(dim=-1), posthoc_bottleneck=True
        )
        sig = sig[:, 0]  # [B, T]
        return sig


# Test
if __name__ == "__main__":
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 10000
    batch_size = 2

    for mode in ["encode", "decode", "reconstruct"]:
        for num_codebooks, vocab_size in zip(
            StableCodec.NUM_CODEBOOKS, StableCodec.VOCAB_SIZES
        ):
            codec = (
                StableCodec(
                    sample_rate,
                    mode=mode,
                    num_codebooks=num_codebooks,
                    vocab_size=vocab_size,
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

    sig, sample_rate = torchaudio.load("example.wav")
    # Move to device as StableCodec does not support CPU
    sig = sig.to(device)
    codec = StableCodec(sample_rate).eval().to(device)
    with torch.no_grad():
        rec_sig = codec(sig)
    torchaudio.save("reconstruction.wav", rec_sig.cpu(), sample_rate)
