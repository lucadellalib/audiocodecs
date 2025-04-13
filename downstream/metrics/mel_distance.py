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

"""Mel distance."""

import torch
import torchaudio
from speechbrain.utils.metric_stats import MetricStats


__all__ = ["MelDistance"]


SAMPLE_RATE = 16000


class MelDistance(MetricStats):
    def __init__(self, sample_rate, n_mels=80, n_fft=1024, hop_length=320):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=1.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.clear()

    @torch.no_grad()
    def append(self, ids, hyp_sig, ref_sig, lens=None):
        assert hyp_sig.shape == ref_sig.shape
        assert hyp_sig.ndim == 2

        # Resample to standard sample rate
        hyp_sig = torchaudio.functional.resample(hyp_sig, self.sample_rate, SAMPLE_RATE)
        ref_sig = torchaudio.functional.resample(ref_sig, self.sample_rate, SAMPLE_RATE)

        self.mel_spec.to(hyp_sig.device)
        hyp_mel = self.amplitude_to_db(self.mel_spec(hyp_sig))
        ref_mel = self.amplitude_to_db(self.mel_spec(ref_sig))

        # Compute L2 distance between Mel spectrograms
        scores = (hyp_mel - ref_mel).norm(dim=1).mean(dim=1).cpu().tolist()

        self.ids += ids
        self.scores += scores


if __name__ == "__main__":
    sample_rate = 24000
    ids = ["A", "B"]
    hyp_sig = torch.randn(2, 2 * sample_rate)
    ref_sig = torch.randn(2, 2 * sample_rate)

    mel_dist = MelDistance(sample_rate)
    mel_dist.append(ids, hyp_sig, ref_sig)
    print(mel_dist.summarize("average"))
