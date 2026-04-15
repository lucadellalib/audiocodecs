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

"""X-Codec 2.0 (see https://arxiv.org/abs/2502.04128)."""

import os
import sys

import torch

from audiocodecs.codec import Codec


__all__ = ["XCodec2"]


class XCodec2(Codec):
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
            from xcodec2.modeling_xcodec2 import XCodec2Model

            sys.path = sys_path
        except ImportError:
            raise ImportError(
                "`pip install git+https://github.com/lucadellalib/X-Codec-2.0.git` to use this module"
            )

        super().__init__(sample_rate, 16000, mode)
        assert num_codebooks == 1
        self.num_codebooks = num_codebooks
        self.vocab_size = 65536

        self.model = XCodec2Model.from_pretrained("HKUST-Audio/xcodec2")
        if mode == "encode":
            self.model.generator.backbone = None
            self.model.generator.head = None
        elif mode == "decode":
            self.model.semantic_model = None
            self.model.SemanticEncoder_module = None
            self.model.CodecEnc = None

    # override
    @torch.no_grad()
    def embs(self):
        device = next(self.model.parameters()).device
        toks = torch.arange(0, self.vocab_size, device=device)
        embs = self.model.generator.quantizer.layers[0]._indices_to_codes(toks)
        embs = embs[None]  # [K, C, H]
        return embs

    # override
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        toks = self.model.encode_code(sig)
        toks = toks.movedim(-1, -2)  # [B, N, 1]
        return toks

    # override
    def _sig_to_feats(self, sig, length):
        # sig: [B, T]
        feats = self.model.encode_feats(sig)
        feats = feats.movedim(-1, -2)
        return feats

    # override
    def _sig_to_qfeats(self, sig, length):
        # sig: [B, T]
        concat_emb = self.model.encode_feats(sig)
        _, vq_code, _ = self.model.generator(concat_emb, vq=True)
        vq_post_emb = self.model.generator.quantizer.get_output_from_indices(
            vq_code.transpose(1, 2)
        )
        return vq_post_emb

    # override
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K]
        toks = toks.movedim(-1, -2)  # [B, 1, N]
        output = self.model.decode_code(toks)
        sig = output[:, 0]  # [B, T]
        return sig

    # override
    def _toks_to_qfeats(self, toks, length):
        # toks: [B, N, K=1]
        vq_post_emb = self.model.generator.quantizer.get_output_from_indices(toks)
        return vq_post_emb

    # override
    def _feats_to_sig(self, feats, length):
        vq_post_emb = self.model.fc_post_a(feats)
        sig = self.model.generator(vq_post_emb, vq=False)[0][:, 0]
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
            XCodec2(
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
    codec = XCodec2(sample_rate, num_codebooks=num_codebooks).eval()
    with torch.no_grad():
        rec_sig = codec(sig)
    torchaudio.save("reconstruction.wav", rec_sig, sample_rate)
