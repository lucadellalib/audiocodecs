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

"""Cosine similarity between speaker embeddings."""

import torch
import torchaudio
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.metric_stats import MetricStats
from transformers import AutoModelForAudioXVector


__all__ = ["SpkSimECAPATDNN", "SpkSimWavLM"]


SAMPLE_RATE = 16000


class SpkSimECAPATDNN(MetricStats):
    def __init__(
        self, model_hub, sample_rate, save_path=HUGGINGFACE_HUB_CACHE, model=None
    ):
        self.sample_rate = sample_rate
        self.model = model
        if model is None:
            self.model = SpeakerRecognition.from_hparams(
                model_hub, savedir=save_path
            ).cpu()
        self.clear()

    @torch.no_grad()
    def append(self, ids, hyp_sig, ref_sig, lens=None):
        assert hyp_sig.shape == ref_sig.shape
        assert hyp_sig.ndim == 2

        # Concatenate
        sig = torch.cat([hyp_sig, ref_sig])
        if lens is not None:
            lens = torch.cat([lens, lens])

        # Resample
        sig = torchaudio.functional.resample(sig, self.sample_rate, SAMPLE_RATE)

        self.model.device = hyp_sig.device
        self.model.to(hyp_sig.device)
        self.model.eval()

        # Forward
        embs = self.model.encode_batch(sig, lens, normalize=False)
        hyp_embs, ref_embs = embs.split([len(hyp_sig), len(ref_sig)])
        scores = self.model.similarity(hyp_embs, ref_embs)[:, 0]

        self.ids += ids
        self.scores += scores.cpu().tolist()


class SpkSimWavLM(MetricStats):
    def __init__(
        self, model_hub, sample_rate, save_path=HUGGINGFACE_HUB_CACHE, model=None
    ):
        self.sample_rate = sample_rate
        self.model = model
        if model is None:
            self.model = AutoModelForAudioXVector.from_pretrained(
                model_hub, cache_dir=save_path
            )
        self.clear()

    @torch.no_grad()
    def append(self, ids, hyp_sig, ref_sig, lens=None):
        assert hyp_sig.shape == ref_sig.shape
        assert hyp_sig.ndim == 2

        # Concatenate
        sig = torch.cat([hyp_sig, ref_sig])
        if lens is not None:
            lens = torch.cat([lens, lens])

        # Resample
        sig = torchaudio.functional.resample(sig, self.sample_rate, SAMPLE_RATE)
        if sig.shape[-1] < 4880:
            sig = torch.nn.functional.pad(
                sig, [0, 4880 - sig.shape[-1]], mode="replicate"
            )

        self.model.to(hyp_sig.device)
        self.model.eval()

        # Attention mask
        attention_mask = None
        if lens is not None:
            abs_length = lens * sig.shape[-1]
            attention_mask = length_to_mask(
                abs_length.int()
            ).long()  # 0 for masked tokens

        # Forward
        embs = self.model(
            input_values=sig,
            attention_mask=attention_mask,
            output_attentions=False,
        ).embeddings

        hyp_embs, ref_embs = embs.split([len(hyp_sig), len(ref_sig)])
        scores = torch.nn.functional.cosine_similarity(hyp_embs, ref_embs, dim=-1)

        self.ids += ids
        self.scores += scores.cpu().tolist()


if __name__ == "__main__":
    sample_rate = 24000
    ids = ["A", "B"]
    hyp_sig = torch.randn(2, 2 * sample_rate)
    ref_sig = torch.randn(2, 2 * sample_rate)

    spk_sim = SpkSimECAPATDNN("speechbrain/spkrec-ecapa-voxceleb", sample_rate)
    spk_sim.append(ids, hyp_sig, ref_sig)
    print(spk_sim.summarize("average"))

    spk_sim = SpkSimWavLM("microsoft/wavlm-base-sv", sample_rate)
    spk_sim.append(ids, hyp_sig, ref_sig)
    print(spk_sim.summarize("average"))
