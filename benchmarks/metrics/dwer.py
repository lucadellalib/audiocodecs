# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Differential WER (dWER) (see https://arxiv.org/abs/1911.07953)."""

import torch
import torchaudio
from faster_whisper import WhisperModel
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from speechbrain.utils.metric_stats import ErrorRateStats, MetricStats
from transformers import WhisperTokenizer


__all__ = ["DWER"]


SAMPLE_RATE = 16000


class DWER(MetricStats):
    def __init__(
        self,
        model_hub,
        sample_rate,
        save_path=HUGGINGFACE_HUB_CACHE,
        model=None,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.model = model
        if model is None:
            self.model = WhisperModel(model_hub, download_root=save_path, **kwargs)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            f"openai/whisper-{model_hub}", cache_dir=save_path
        )
        self.wer_computer = ErrorRateStats()
        self.cer_computer = ErrorRateStats(split_tokens=True)

    def clear(self):
        self.wer_computer.clear()
        self.cer_computer.clear()

    @torch.no_grad()
    def append(self, ids, hyp_sig, ref_sig, lens=None):
        assert hyp_sig.shape == ref_sig.shape
        assert hyp_sig.ndim == 2

        # Concatenate
        sig = torch.cat([hyp_sig, ref_sig])

        # Move to device
        self.model.device = sig.device

        # Resample
        sig = (
            torchaudio.functional.resample(sig, self.sample_rate, SAMPLE_RATE)
            .cpu()
            .numpy()
        )

        texts = []
        for x in sig:
            segs, _ = self.model.transcribe(
                x, beam_size=1, language="en", without_timestamps=True
            )
            text = ""
            for seg in segs:
                text += seg.text
            texts.append(text)

        # Forward
        texts = [self.tokenizer.normalize(x) for x in texts]
        texts = [x.split(" ") for x in texts]
        hyp_text = texts[: hyp_sig.shape[0]]
        ref_text = texts[hyp_sig.shape[0] :]

        # Compute WER
        self.wer_computer.append(ids, hyp_text, ref_text)
        self.cer_computer.append(ids, hyp_text, ref_text)

    def summarize(self, field=None):
        wer_summary = self.wer_computer.summarize()
        cer_summary = self.cer_computer.summarize()
        wer_summary["CER"] = wer_summary["error_rate_char"] = cer_summary["error_rate"]
        if field is None:
            return wer_summary
        return wer_summary[field]

    def write_stats(self, filestream, verbose=False):
        self.wer_computer.write_stats(filestream)


if __name__ == "__main__":
    import torch

    sample_rate = 24000
    ids = ["A", "B"]
    hyp_sig = torch.randn(2, 2 * sample_rate)
    ref_sig = torch.randn(2, 2 * sample_rate)

    dwer = DWER("large-v3", sample_rate)
    dwer.append(ids, hyp_sig, ref_sig)
    print(dwer.summarize("error_rate"))
    print(dwer.summarize("WER"))
    print(dwer.summarize("error_rate_char"))
    print(dwer.summarize("CER"))
