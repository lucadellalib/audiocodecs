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

"""DAC (see https://arxiv.org/abs/2306.06546)."""

import os
import sys

import torch

from audiocodecs.codec import Codec


__all__ = ["DAC"]


class DAC(Codec):
    def __init__(
        self,
        sample_rate,
        orig_sample_rate=16000,
        mode="reconstruct",
        num_codebooks=8,
        latent=False,
    ):
        try:
            # Workaround to avoid name collisions with installed modules
            root_dir = os.path.dirname(os.path.realpath(__file__))
            sys_path = [x for x in sys.path]
            sys.path = [x for x in sys.path if root_dir not in x]
            import dac

            sys.path = sys_path
        except ImportError:
            raise ImportError("`pip install descript-audio-codec` to use this module")

        super().__init__(sample_rate, orig_sample_rate, mode)
        self.num_codebooks = num_codebooks
        self.vocab_size = 1024
        self.latent = latent

        tag = int(orig_sample_rate / 1000)
        model_path = str(dac.utils.download(model_type=f"{tag}khz"))
        self.model = dac.DAC.load(model_path)

        if mode == "encode":
            self.model.decoder = None
        elif mode == "decode":
            self.model.encoder = None

    # override
    @torch.no_grad()
    def embs(self):
        if self.latent:
            embs = [
                quantizer.codebook.weight
                for quantizer in self.model.quantizer.quantizers
            ]
            embs = embs[: self.num_codebooks]
            embs = torch.stack(embs)  # [K, C, H]
            return embs
        # See https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/nn/quantize.py#L200
        device = next(iter(self.model.state_dict().values())).device
        toks = torch.arange(self.vocab_size, device=device)
        toks = (
            toks[:, None, None].expand(-1, self.num_codebooks, -1).clone()
        )  # [C, K, 1]
        z_q, z_p, _ = self.model.quantizer.from_codes(toks)
        z_ps = z_p.split(z_p.shape[1] // toks.shape[1], dim=1)  # [C, D, 1] * K
        z_qs = []
        for i, z_p_i in enumerate(z_ps):
            z_q_i = self.model.quantizer.quantizers[i].out_proj(z_p_i)  # [C, H, 1]
            z_qs.append(z_q_i)
        # assert (z_q == sum(z_qs)).all()
        # Embeddings pre-projections: size = 8
        # Embeddings post-projections: size = 1024
        embs = torch.stack(z_qs)[..., 0]  # [K, C, H]
        return embs

    # override
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        _, toks, *_ = self.model.encode(
            sig[:, None], n_quantizers=self.num_codebooks
        )  # [B, K, N]
        toks = toks.movedim(-1, -2)  # [B, N, K]
        return toks

    # override
    def _sig_to_feats(self, sig, length):
        # sig: [B, T]
        if self.latent:
            feats = self.model.encoder(sig[:, None])  # [B, H, N]
            feats = self.model.quantizer.quantizers[0].in_proj(feats)
            feats = feats.movedim(-1, -2)
            return feats
        feats = self.model.encoder(sig[:, None])  # [B, H, N]
        feats = feats.movedim(-1, -2)
        return feats

    # override
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K]
        qfeats, _, _ = self.model.quantizer.from_codes(
            toks.movedim(-1, -2)  # [B, K, N]
        )
        sig = self.model.decode(qfeats)[:, 0]  # [B, T]
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
            DAC(
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

    sig, sample_rate = torchaudio.load("example.wav")
    codec = DAC(sample_rate, num_codebooks=num_codebooks).eval()
    with torch.no_grad():
        rec_sig = codec(sig)
    torchaudio.save("reconstruction.wav", rec_sig, sample_rate)
