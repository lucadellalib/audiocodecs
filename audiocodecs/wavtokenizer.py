# ==============================================================================
# Copyright 2024 Luca Della Libera.
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

"""WavTokenizer (see https://arxiv.org/abs/2408.16532)."""

import os
import sys

import torch
from huggingface_hub import snapshot_download

from audiocodecs.codec import Codec


__all__ = ["WavTokenizer"]


class WavTokenizer(Codec):
    def __init__(
        self,
        sample_rate,
        mode="reconstruct",
        source="novateur/WavTokenizer-medium-speech-75token",
        config="wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        checkpoint="wavtokenizer_medium_speech_320_24k_v2.ckpt",
    ):
        try:
            # Workaround to avoid name collisions with installed modules
            root_dir = os.path.dirname(os.path.realpath(__file__))
            sys_path = [x for x in sys.path]
            sys.path = [x for x in sys.path if root_dir not in x]
            import wavtokenizer

            sys.path = sys_path
        except ImportError:
            raise ImportError(
                "`pip install git+https://github.com/lucadellalib/WavTokenizer.git@main` to use this module"
            )

        super().__init__(sample_rate, 24000, mode)
        self.num_codebooks = 1
        self.vocab_size = 4096

        path = snapshot_download(repo_id=source)
        config_path = os.path.join(path, config)
        checkpoint_path = os.path.join(path, checkpoint)
        self.model = wavtokenizer.WavTokenizer.from_pretrained0802(
            config_path, checkpoint_path
        )

        if mode == "encode":
            self.model.feature_extractor.encodec.decoder = None
            self.model.head = None
        elif mode == "decode":
            self.model.feature_extractor.encodec.encoder = None

    # override
    @torch.no_grad()
    def embs(self):
        embs = self.model.feature_extractor.encodec.quantizer.vq.layers[0].codebook
        embs = embs[None]  # [K, C, H]
        return embs

    # override
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        _, toks = self.model.encode(sig, bandwidth_id=0)
        toks = toks.movedim(0, -1)  # [B, N, K]
        return toks

    # override
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K]
        feats = self.model.codes_to_features(toks.movedim(-1, 0))
        sig = self.model.decode(
            feats, bandwidth_id=torch.tensor(0, device=toks.device)
        )  # [B, T]
        return sig


# Test
if __name__ == "__main__":
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 10000
    batch_size = 2

    for mode in ["encode", "decode", "reconstruct"]:
        codec = WavTokenizer(sample_rate, mode=mode).eval().to(device)
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

    sig, sample_rate = torchaudio.load("example.wav")
    codec = WavTokenizer(sample_rate).eval()
    with torch.no_grad():
        rec_sig = codec(sig)
    torchaudio.save("reconstruction.wav", rec_sig, sample_rate)
