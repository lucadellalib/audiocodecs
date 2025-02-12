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

"""WavLM + K-means (see https://arxiv.org/abs/2312.09747)."""

import torch

from audiocodecs.codec import Codec


__all__ = ["WavLMKmeans"]


class WavLMKmeans(Codec):
    LAYER_IDS = [(6,), (1, 3, 6)]

    def __init__(self, sample_rate, mode="reconstruct", layer_ids=(6,)):
        try:
            import speechbrain
        except ImportError:
            raise ImportError(
                "`pip install git+https://github.com/lucadellalib/speechbrain@50ffdc772c0d977390025ee7787735db9b92488c#egg=speechbrain` to use this module"
            )

        super().__init__(sample_rate, 16000, mode)
        self.layer_ids = layer_ids
        self.vocab_size = 512

        self.model = torch.hub.load(
            repo_or_dir="lucadellalib/discrete-wavlm-codec",
            model="discrete_wavlm_large",
            layer_ids=layer_ids,
        )
        if mode == "encode":
            self.model.dequantizer = None
            self.model.vocoder = None
        elif mode == "decode":
            self.model.encoder = None

    # override
    @torch.no_grad()
    def embs(self):
        device = next(iter(self.model.state_dict().values())).device
        toks = torch.arange(self.vocab_size, device=device)
        toks = toks[:, None].expand(-1, len(self.layer_ids)).clone()  # [C, K]
        embs = self.model.toks_to_qfeats(toks)  # [C, H, K]
        embs = embs.movedim(-1, 0)  # [K, C, H]
        return embs

    # override
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        feats = self.model.sig_to_feats(sig)
        toks = self.model.feats_to_toks(feats)  # [B, N, K]
        return toks

    # override
    def _sig_to_feats(self, sig, length):
        # sig: [B, T]
        feats = self.model.sig_to_feats(sig).mean(dim=-1)  # [B, N, H, K]
        return feats

    # override
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K]
        qfeats = self.model.toks_to_qfeats(toks)
        feats = self.model.qfeats_to_feats(qfeats)
        sig = self.model.feats_to_sig(feats)[:, 0]  # [B, T]
        return sig


# Test
if __name__ == "__main__":
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 10000
    batch_size = 2
    layer_ids = [6]

    for mode in ["encode", "decode", "reconstruct"]:
        codec = (
            WavLMKmeans(sample_rate, mode=mode, layer_ids=layer_ids).eval().to(device)
        )
        input = (
            torch.zeros(batch_size, 10, len(layer_ids)).long()
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
    codec = WavLMKmeans(sample_rate, layer_ids=layer_ids).eval()
    with torch.no_grad():
        rec_sig = codec(sig)
    torchaudio.save("reconstruction.wav", rec_sig, sample_rate)
