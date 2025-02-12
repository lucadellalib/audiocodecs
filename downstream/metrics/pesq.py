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

"""Perceptual evaluation of speech quality (PESQ) (see https://en.wikipedia.org/wiki/Perceptual_Evaluation_of_Speech_Quality)."""

import os
import sys

import torch
import torchaudio
from speechbrain.utils.metric_stats import MetricStats
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality


__all__ = ["PESQ"]


SAMPLE_RATE = 16000


class PESQ(MetricStats):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.clear()

    @torch.no_grad()
    def append(self, ids, hyp_sig, ref_sig, lens=None):
        assert hyp_sig.shape == ref_sig.shape
        assert hyp_sig.ndim == 2

        # Resample
        hyp_sig = torchaudio.functional.resample(hyp_sig, self.sample_rate, SAMPLE_RATE)
        ref_sig = torchaudio.functional.resample(ref_sig, self.sample_rate, SAMPLE_RATE)

        # Workaround to avoid name collisions with installed modules
        root_dir = os.path.dirname(os.path.realpath(__file__))
        sys_path = [x for x in sys.path]
        sys.path = [x for x in sys.path if root_dir not in x]
        scores = [
            perceptual_evaluation_speech_quality(hyp, ref, SAMPLE_RATE, "wb").cpu()
            for hyp, ref in zip(hyp_sig, ref_sig)
        ]
        sys.path = sys_path

        self.ids += ids
        self.scores += scores


if __name__ == "__main__":
    sample_rate = 24000
    ids = ["A", "B"]
    hyp_sig = torch.randn(2, 2 * sample_rate)
    ref_sig = torch.randn(2, 2 * sample_rate)

    pesq = PESQ(sample_rate)
    pesq.append(ids, hyp_sig, ref_sig)
    print(pesq.summarize("average"))
