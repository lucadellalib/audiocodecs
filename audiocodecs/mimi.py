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

"""Mimi (see https://kyutai.org/Moshi.pdf)."""

import torch

from audiocodecs.codec import Codec


__all__ = ["Mimi"]


class Mimi(Codec):
    def __init__(
        self,
        sample_rate,
        mode="reconstruct",
        num_codebooks=8,
        latent=True,
    ):
        try:
            from transformers import MimiModel
        except ImportError:
            raise ImportError("`pip install transformers>=4.45.1` to use this module")

        super().__init__(sample_rate, 24000, mode)
        self.num_codebooks = num_codebooks
        self.vocab_size = 2048
        self.latent = latent

        self.model = MimiModel.from_pretrained("kyutai/mimi")
        if mode == "encode":
            self.model.decoder = None
            self.model.decoder_transformer = None
        elif mode == "decode":
            self.model.encoder = None
            self.model.encoder_transformer = None

    # override
    @torch.no_grad()
    def embs(self):
        if self.latent:
            semantic_layers = (
                self.model.quantizer.semantic_residual_vector_quantizer.layers
            )
            acoustic_layers = (
                self.model.quantizer.acoustic_residual_vector_quantizer.layers
            )
            layers = (semantic_layers + acoustic_layers)[: self.num_codebooks]
            embs = [layer.codebook.embed for layer in layers]
            embs = torch.stack(embs)  # [K, C, H]
            return embs
        semantic_layers = self.model.quantizer.semantic_residual_vector_quantizer.layers
        acoustic_layers = self.model.quantizer.acoustic_residual_vector_quantizer.layers
        layers = (semantic_layers + acoustic_layers)[: self.num_codebooks]
        embs = [layer.codebook.embed for layer in layers]
        embs = torch.stack(embs)  # [K, C, H]
        embs = embs[..., None]
        embs_semantic = (
            self.model.quantizer.semantic_residual_vector_quantizer.output_proj(
                embs[0]
            )[..., 0]
        )
        if self.num_codebooks > 1:
            embs_acoustic = (
                self.model.quantizer.acoustic_residual_vector_quantizer.output_proj(
                    embs[1:].flatten(end_dim=1)
                )
            )
            embs_acoustic = embs_acoustic.reshape(
                self.num_codebooks - 1, self.vocab_size, -1
            )
            embs = torch.cat([embs_semantic[None], embs_acoustic])
        else:
            embs = embs_semantic[None]
        return embs

    # override
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        abs_lens = sig.shape[-1] * length
        max_len = abs_lens.max().long().item()
        padding_mask = (
            torch.arange(
                max_len,
                device=length.device,
                dtype=length.dtype,
            )[None]
            < abs_lens[:, None]
        )
        output = self.model.encode(
            sig[:, None], padding_mask[:, None], num_quantizers=self.num_codebooks
        )
        toks = output.audio_codes.movedim(-1, -2)  # [B, N, K]
        return toks

    # override
    def _sig_to_feats(self, sig, length):
        # sig: [B, T]
        if self.latent:
            input_values = sig[:, None]
            embeddings = self.model.encoder(input_values)
            encoder_outputs = self.model.encoder_transformer(embeddings.transpose(1, 2))
            embeddings = encoder_outputs[0].transpose(1, 2)
            embeddings = self.model.downsample(embeddings)
            embeddings = (
                self.model.quantizer.semantic_residual_vector_quantizer.input_proj(
                    embeddings
                )
            )
            feats = embeddings.movedim(-1, -2)
            return feats
        input_values = sig[:, None]
        embeddings = self.model.encoder(input_values)
        encoder_outputs = self.model.encoder_transformer(embeddings.transpose(1, 2))
        embeddings = encoder_outputs[0].transpose(1, 2)
        embeddings = self.model.downsample(embeddings)
        feats = embeddings.movedim(-1, -2)
        return feats

    # override
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K]
        output = self.model.decode(toks.movedim(-1, -2))
        sig = output.audio_values[:, 0]  # [B, T]
        return sig


# Test
if __name__ == "__main__":
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 10000
    batch_size = 2
    num_codebooks = 8

    for mode in ["encode", "decode", "reconstruct"]:
        codec = (
            Mimi(sample_rate, mode=mode, num_codebooks=num_codebooks).eval().to(device)
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
    codec = Mimi(sample_rate, num_codebooks=num_codebooks).eval()
    with torch.no_grad():
        rec_sig = codec(sig)
    torchaudio.save("reconstruction.wav", rec_sig, sample_rate)
