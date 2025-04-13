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

"""STFT distance."""

import torch
import torchaudio
from speechbrain.utils.metric_stats import MetricStats


__all__ = ["STFTDistance"]


SAMPLE_RATE = 16000


class STFTDistance(MetricStats):
    def __init__(self, sample_rate, n_fft=1024, hop_length=320):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.clear()

    @torch.no_grad()
    def append(self, ids, hyp_sig, ref_sig, lens=None):
        assert hyp_sig.shape == ref_sig.shape
        assert hyp_sig.ndim == 2

        # Resample to standard sample rate
        hyp_sig = torchaudio.functional.resample(hyp_sig, self.sample_rate, SAMPLE_RATE)
        ref_sig = torchaudio.functional.resample(ref_sig, self.sample_rate, SAMPLE_RATE)

        # Compute STFT -> magnitude -> dB
        hyp_stft = torch.stft(
            hyp_sig,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(hyp_sig.device),
            return_complex=True,
        ).abs()

        ref_stft = torch.stft(
            ref_sig,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(ref_sig.device),
            return_complex=True,
        ).abs()

        hyp_db = self.amplitude_to_db(hyp_stft)
        ref_db = self.amplitude_to_db(ref_stft)

        # Compute L2 distance between log magnitude spectrograms
        scores = (hyp_db - ref_db).norm(dim=1).mean(dim=1).cpu().tolist()

        self.ids += ids
        self.scores += scores


if __name__ == "__main__":
    sample_rate = 24000
    ids = ["A", "B"]
    hyp_sig = torch.randn(2, 2 * sample_rate)
    ref_sig = torch.randn(2, 2 * sample_rate)

    stft_dist = STFTDistance(sample_rate)
    stft_dist.append(ids, hyp_sig, ref_sig)
    print(stft_dist.summarize("average"))
