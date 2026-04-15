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

"""Perplexity metrics."""

import torch
import torchaudio
from faster_whisper import WhisperModel
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from speechbrain.utils.metric_stats import MetricStats
from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperTokenizer


__all__ = ["ASRPerplexity"]


SAMPLE_RATE = 16000


class ASRPerplexity(MetricStats):
    def __init__(
        self,
        model_hub,
        sample_rate,
        save_path=HUGGINGFACE_HUB_CACHE,
        model=None,
        asr_model_hub="large-v3",
        asr_model=None,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.tokenizer = AutoTokenizer.from_pretrained(model_hub, cache_dir=save_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = model
        if model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_hub, cache_dir=save_path
            )

        # ASR model
        self.asr_model = asr_model
        if asr_model is None:
            self.asr_model = WhisperModel(
                asr_model_hub, download_root=save_path, **kwargs
            )
        self.asr_tokenizer = WhisperTokenizer.from_pretrained(
            f"openai/whisper-{asr_model_hub}", cache_dir=save_path
        )

        self.clear()

    @torch.no_grad()
    def append(self, ids, sig, lens=None, locales=None):
        assert sig.ndim == 2

        if locales is None:
            locales = ["en"] * len(ids)
        locales = locales

        # Move to device
        device = sig.device
        self.asr_model.device = device

        # Resample
        sig = (
            torchaudio.functional.resample(sig, self.sample_rate, SAMPLE_RATE)
            .cpu()
            .numpy()
        )

        texts = []
        for x, locale in zip(sig, locales):
            # Forward
            segs, _ = self.asr_model.transcribe(
                x,
                beam_size=1,
                language=locale,
                without_timestamps=True,  # temperature=0.0,
            )
            text = ""
            for seg in segs:
                text += seg.text
            texts.append(text)

        texts = [self.asr_tokenizer.normalize(x) for x in texts]

        self.model.to(device)
        self.model.eval()

        # Tokenize the input texts
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = tokenized["input_ids"].to(self.model.device)
        attention_mask = tokenized["attention_mask"].to(self.model.device)

        # Forward pass for log-likelihoods
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous()

        counts = shift_attention_mask.sum(dim=1)
        log_perplexities = (
            torch.nn.functional.cross_entropy(
                shift_logits.movedim(-1, -2), shift_labels, reduction="none"
            )
            * shift_attention_mask
        ).sum(dim=1) / counts

        # valid samples: at least 1 predicted token + finite loss
        valid = log_perplexities.isfinite()

        if not valid.any():
            return  # nothing to add

        valid_list = valid.cpu().tolist()
        texts = [t for t, v in zip(texts, valid_list) if v]
        ids = [i for i, v in zip(ids, valid_list) if v]
        log_perplexities = log_perplexities[valid]
        counts = counts[valid]

        self.texts += texts
        self.perplexities += log_perplexities.exp().cpu().tolist()
        self.ids += ids
        self.scores += log_perplexities.cpu().tolist()
        self.counts += counts.cpu().tolist()

    def summarize(self, field=None):
        perplexity = torch.tensor(
            sum([x * y for x, y in zip(self.scores, self.counts)]) / sum(self.counts)
        ).exp()
        self.summary = {"average": perplexity.item()}
        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def clear(self):
        super().clear()
        self.perplexities = []
        self.texts = []
        self.counts = []


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_rate = 24000
    ids = ["A", "B"]
    sig = torch.ones(2, 2 * sample_rate, device=device)

    perplexity = ASRPerplexity(
        model_hub="openai-community/gpt2-large",
        sample_rate=sample_rate,
        asr_model_hub="small",
    )
    perplexity.append(ids, sig)
    print(perplexity.summarize("average"))
