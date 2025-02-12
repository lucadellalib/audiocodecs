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

"""BigCodec (see https://arxiv.org/abs/2409.05377)."""

import os
import sys

import torch
from huggingface_hub import snapshot_download

from audiocodecs.codec import Codec


__all__ = ["BigCodec"]


class BigCodec(Codec):
    SOURCES = ["Alethia/BigCodec"]
    CHECKPOINTS = ["bigcodec.pt"]

    def __init__(
        self,
        sample_rate,
        mode="reconstruct",
        source="Alethia/BigCodec",
        checkpoint="bigcodec.pt",
        latent=True,
    ):
        try:
            # Workaround to avoid name collisions with installed modules
            root_dir = os.path.dirname(os.path.realpath(__file__))
            sys_path = [x for x in sys.path]
            sys.path = [x for x in sys.path if root_dir not in x]
            import bigcodec

            sys.path = sys_path
        except ImportError:
            raise ImportError(
                "`pip install git+https://github.com/lucadellalib/BigCodec.git@main` to use this module"
            )

        super().__init__(sample_rate, 16000, mode)
        self.num_codebooks = 1
        self.vocab_size = 8192
        self.latent = latent

        path = snapshot_download(repo_id=source)
        checkpoint_path = os.path.join(path, checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.encoder = bigcodec.CodecEncoder()
        self.encoder.load_state_dict(checkpoint["CodecEnc"])
        self.decoder = bigcodec.CodecDecoder()
        self.decoder.load_state_dict(checkpoint["generator"])
        self.quantizer = self.decoder.quantizer

        if mode == "encode":
            self.decoder = None
        elif mode == "decode":
            self.encoder = None

    # override
    @torch.no_grad()
    def embs(self):
        if self.latent:
            embs = self.quantizer.get_emb()[0]
            embs = embs[None]  # [K=1, C, H]
            return embs
        embs = self.quantizer.get_emb()[0]
        embs = embs[None]
        embs = self.quantizer.layers[0].out_proj(embs)
        return embs

    # override
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        feats = self.encoder(sig[:, None])
        _, toks, _ = self.quantizer(feats)
        toks = toks[0, :, :, None]  # [B, N, K=1]
        return toks

    # override
    def _sig_to_feats(self, sig, length):
        # sig: [B, T]
        if self.latent:
            feats = self.encoder(sig[:, None])  # [B, H, N]
            feats = feats.movedim(-1, -2)
            feats = self.quantizer.layers[0].in_proj(feats)
            return feats
        feats = self.encoder(sig[:, None])  # [B, H, N]
        feats = feats.movedim(-1, -2)
        return feats

    # override
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K=1]
        qfeats = self.quantizer.vq2emb(toks)
        sig = self.decoder(qfeats.movedim(-1, -2), vq=False)[:, 0]  # [B, T]
        return sig


# Test
if __name__ == "__main__":
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 10000
    batch_size = 2

    for mode in ["encode", "decode", "reconstruct"]:
        codec = BigCodec(sample_rate, mode=mode).eval().to(device)
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
    codec = BigCodec(sample_rate).eval()
    with torch.no_grad():
        rec_sig = codec(sig)
    torchaudio.save("reconstruction.wav", rec_sig, sample_rate)
