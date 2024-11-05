# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
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
