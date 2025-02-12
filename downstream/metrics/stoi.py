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

"""Short-time objective intelligibility (STOI) (see https://ieeexplore.ieee.org/abstract/document/5495701)."""

import torch
import torchaudio
from speechbrain.utils.metric_stats import MetricStats
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility


__all__ = ["STOI"]


SAMPLE_RATE = 16000


class STOI(MetricStats):
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

        scores = [
            short_time_objective_intelligibility(
                hyp.cpu(), ref.cpu(), SAMPLE_RATE
            ).float()
            for hyp, ref in zip(hyp_sig, ref_sig)
        ]

        self.ids += ids
        self.scores += scores


if __name__ == "__main__":
    sample_rate = 24000
    ids = ["A", "B"]
    hyp_sig = torch.randn(2, 2 * sample_rate)
    ref_sig = torch.randn(2, 2 * sample_rate)

    stoi = STOI(sample_rate)
    stoi.append(ids, hyp_sig, ref_sig)
    print(stoi.summarize("average"))
