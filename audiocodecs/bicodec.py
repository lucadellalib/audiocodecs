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

"""BiCodec (see https://arxiv.org/abs/2503.01710)."""

import torch

from audiocodecs.codec import Codec


__all__ = ["BiCodec"]


class BiCodec(Codec):
    MODEL_NAMES = ["SparkAudio/Spark-TTS-0.5B"]

    def __init__(
        self,
        sample_rate,
        mode="reconstruct",
        num_codebooks=1,
        model_name="SparkAudio/Spark-TTS-0.5B",
    ):
        try:
            # Workaround to avoid name collisions with installed modules
            from sparktts.models.bicodec import BiCodec as BiCodecOrig
        except ImportError:
            raise ImportError(
                "`pip install git+https://github.com/lucadellalib/Spark-TTS.git` to use this module"
            )

        super().__init__(sample_rate, 16000, mode)
        assert num_codebooks == 1
        self.num_codebooks = num_codebooks
        self.vocab_size = 8192

        self.model = BiCodecOrig.load_from_checkpoint(model_name)
        if mode == "encode":
            self.model.decoder = None
            self.model.prenet = None
        elif mode == "decode":
            self.model.encoder = None
            self.model.mel_transformer = None
            self.model.feature_extractor = None

    # override
    @torch.no_grad()
    def embs(self):
        device = next(self.model.parameters()).device
        embs = self.model.combined_codebook().clone().to(device)
        embs = embs[None]  # [K, C, H]
        return embs

    # override
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        semantic_tokens, global_tokens = self.model.tokenize(sig[:, None])
        assert global_tokens.shape[-1] == 32
        toks = torch.cat([global_tokens[:, 0], semantic_tokens], dim=-1)
        toks = toks[..., None]  # [B, N, 1]
        return toks

    # override
    def _sig_to_feats(self, sig, length):
        # sig: [B, T]
        feats = self.model.extract_combined_feats(sig[:, None])
        feats = feats.movedim(-1, -2)
        return feats

    # override
    def _sig_to_qfeats(self, sig, length):
        # sig: [B, T]
        semantic_tokens, global_tokens = self.model.tokenize(sig[:, None])
        assert global_tokens.shape[-1] == 32
        z_q = self.model.quantizer.detokenize(semantic_tokens)
        d_vector = self.model.speaker_encoder.detokenize(global_tokens)
        d_vector = d_vector[..., None].expand(-1, -1, z_q.shape[-1]).clone()
        qfeats = torch.cat([z_q, d_vector], dim=1).movedim(-1, -2)
        return qfeats

    # override
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K]
        toks = toks[..., 0]
        assert toks.shape[-1] >= 32
        global_tokens = toks[..., :32]
        semantic_tokens = toks[..., 32:]
        global_tokens = global_tokens[:, None]
        output = self.model.detokenize(semantic_tokens, global_tokens)
        sig = output[:, 0]  # [B, T]
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
            BiCodec(
                sample_rate,
                mode=mode,
                num_codebooks=num_codebooks,
            )
            .eval()
            .to(device)
        )
        input = (
            torch.zeros(batch_size, 32 + 10, num_codebooks).long()
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
    codec = BiCodec(sample_rate, num_codebooks=num_codebooks).eval()
    with torch.no_grad():
        rec_sig = codec(sig)
    torchaudio.save("reconstruction.wav", rec_sig, sample_rate)
