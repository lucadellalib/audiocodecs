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

"""Speaker encoders."""

import torch
import torchaudio
from speechbrain.dataio.dataio import length_to_mask
from transformers import AutoModelForAudioXVector


__all__ = ["WavLM"]


SAMPLE_RATE = 16000


class WavLM(torch.nn.Module):
    def __init__(self, model_hub, save_path, sample_rate, pool=True):
        super().__init__()
        self.model_hub = model_hub
        self.save_path = save_path
        self.sample_rate = sample_rate
        self.pool = pool
        self.model = AutoModelForAudioXVector.from_pretrained(
            model_hub, cache_dir=save_path
        )

    @torch.no_grad()
    def forward(self, sig, lens=None):
        # Resample
        sig = torchaudio.functional.resample(sig, self.sample_rate, SAMPLE_RATE)

        self.model.to(sig.device)
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
        )

        if self.pool:
            return embs.embeddings

        return embs.hidden_states[-1]


if __name__ == "__main__":
    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

    x = torch.randn(1, 16000)
    model = WavLM(
        "microsoft/wavlm-base-sv",
        HUGGINGFACE_HUB_CACHE,
        16000,
        pool=True,
    )
    y = model(x)
    print(y.shape)
