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

"""NanoCodec (see https://arxiv.org/abs/2508.05835)."""

import logging

import torch

from audiocodecs.codec import Codec


__all__ = ["NanoCodec"]


class NanoCodec(Codec):
    MODEL_NAMES = [
        "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps",
        "nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps",
        "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps",
    ]

    def __init__(
        self,
        sample_rate,
        mode="reconstruct",
        num_codebooks=4,
        model_name="nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps",
    ):
        try:
            from nemo.collections.tts.models import AudioCodecModel
        except ImportError:
            raise ImportError("`pip install nemo_toolkit[all]` to use this module")

        logging.getLogger("nemo_logger").setLevel(logging.ERROR)

        super().__init__(sample_rate, 22050, mode)
        self.num_codebooks = num_codebooks
        self.vocab_size = 4096
        self.model_name = model_name

        if model_name == "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps":
            assert num_codebooks == 4

        self.model = AudioCodecModel.from_pretrained(model_name).eval()
        if mode == "encode":
            self.model.audio_decoder = None
        elif mode == "decode":
            self.model.audio_encoder = None

    # override
    @torch.no_grad()
    def embs(self):
        device = next(iter(self.model.state_dict().values())).device
        toks = torch.arange(self.vocab_size, device=device)
        toks = (
            toks[:, None, None].expand(-1, -1, self.num_codebooks).clone()
        )  # [C, 1, K]
        embs = [
            quantizer.decode(
                indices=toks[..., k, None].movedim(-1, 0),
                input_len=torch.ones(len(toks), device=device),
            )
            for k, quantizer in enumerate(self.model.vector_quantizer.fsqs)
        ]
        embs = torch.cat(embs, dim=-1)
        embs = embs.permute(2, 0, 1)  # [K, C, H]
        return embs

    # override
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        abs_length = (sig.shape[-1] * length).round().long()
        encoded_tokens, _ = self.model.encode(audio=sig, audio_len=abs_length)
        encoded_tokens = encoded_tokens.movedim(-1, -2)  # [B, N, K]
        return encoded_tokens

    # override
    def _sig_to_feats(self, sig, length):
        # sig: [B, T]
        abs_length = (sig.shape[-1] * length).round().long()
        encoded, _ = self.model.encode_audio(audio=sig, audio_len=abs_length)
        return encoded.movedim(-1, -2)

    # override
    def _sig_to_qfeats(self, sig, length):
        # sig: [B, T]
        toks = self._sig_to_toks(sig, length)
        qfeats = self._toks_to_qfeats(toks, length)
        return qfeats

    # override
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K]
        abs_lens = toks.shape[-2] * length
        toks = toks.movedim(-1, -2)
        output, _ = self.model.decode(tokens=toks, tokens_len=abs_lens)
        return output

    # override
    def _toks_to_qfeats(self, toks, length):
        # toks: [B, N, K]
        abs_lens = toks.shape[-2] * length
        toks = toks.movedim(-1, -2)
        dequantized = self.model.dequantize(tokens=toks, tokens_len=abs_lens)
        return dequantized.movedim(-1, -2)


# Test
if __name__ == "__main__":
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 10000
    batch_size = 2
    num_codebooks = 4

    for mode in ["encode", "decode", "reconstruct"]:
        codec = (
            NanoCodec(sample_rate, mode=mode, num_codebooks=num_codebooks)
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
    # Move to device as NanoCodec does not support CPU (?)
    sig = sig.to(device)
    codec = NanoCodec(sample_rate, num_codebooks=num_codebooks).eval().to(device)
    with torch.no_grad():
        rec_sig = codec(sig).cpu()
    torchaudio.save("reconstruction.wav", rec_sig, sample_rate)
