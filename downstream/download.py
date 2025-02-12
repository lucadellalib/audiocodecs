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

"""Download weights for all models (useful if no connection is available on the cluster's compute nodes)."""

import logging

import torch
from metrics.dnsmos import DNSMOS
from metrics.dwer import DWER
from metrics.speaker_similarity import SpkSimECAPATDNN, SpkSimWavLM
from metrics.utmos import UTMOS

from audiocodecs.bigcodec import BigCodec
from audiocodecs.dac import DAC
from audiocodecs.encodec import Encodec
from audiocodecs.focalcodec import FocalCodec
from audiocodecs.mimi import Mimi
from audiocodecs.semanticodec import SemantiCodec
from audiocodecs.speechtokenizer import SpeechTokenizer
from audiocodecs.stablecodec import StableCodec
from audiocodecs.wavlm_kmeans import WavLMKmeans
from audiocodecs.wavtokenizer import WavTokenizer


@torch.no_grad()
def download_weights():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 1000
    batch_size = 2

    # BigCodec
    try:
        codec = BigCodec(sample_rate).eval()
        input = torch.zeros(batch_size, 100, device=device)
        codec(input)
    except Exception as e:
        logging.warning(e)

    # DAC
    for num_codebooks in [2, 4, 8, 16, 32]:
        try:
            codec = DAC(sample_rate, num_codebooks=num_codebooks).eval().to(device)
            input = torch.zeros(batch_size, 100, device=device)
            codec(input)
        except Exception as e:
            logging.warning(e)

    # EnCodec
    for num_codebooks in [2, 4, 8, 16, 32]:
        try:
            codec = Encodec(sample_rate, num_codebooks=num_codebooks).eval().to(device)
            input = torch.zeros(batch_size, 100, device=device)
            codec(input)
        except Exception as e:
            logging.warning(e)

    # FocalCodec
    try:
        for config in FocalCodec.CONFIGS:
            codec = FocalCodec(sample_rate, config=config).eval()
            input = torch.zeros(batch_size, 100, device=device)
            codec(input)
    except Exception as e:
        logging.warning(e)

    # Mimi
    for num_codebooks in range(1, 9):
        try:
            codec = Mimi(sample_rate, num_codebooks=num_codebooks).eval().to(device)
            input = torch.zeros(batch_size, 100, device=device)
            codec(input)
        except Exception as e:
            logging.warning(e)

    # SemantiCodec
    for token_rate, semantic_vocab_size in zip(
        SemantiCodec.TOKEN_RATES, SemantiCodec.SEMANTIC_VOCAB_SIZES
    ):
        try:
            codec = (
                SemantiCodec(
                    sample_rate,
                    token_rate=token_rate,
                    semantic_vocab_size=semantic_vocab_size,
                )
                .eval()
                .to(device)
            )
            input = torch.zeros(batch_size, 100, device=device)
            codec(input)
        except Exception as e:
            logging.warning(e)

    # SpeechTokenizer
    for num_codebooks in range(8):
        try:
            codec = (
                SpeechTokenizer(sample_rate, num_codebooks=num_codebooks)
                .eval()
                .to(device)
            )
            input = torch.zeros(batch_size, 100, device=device)
            codec(input)
        except Exception as e:
            logging.warning(e)

    # Stable Codec
    for num_codebooks, vocab_size in zip(
        StableCodec.NUM_CODEBOOKS, StableCodec.VOCAB_SIZES
    ):
        try:
            codec = (
                StableCodec(
                    sample_rate, num_codebooks=num_codebooks, vocab_size=vocab_size
                )
                .eval()
                .to(device)
            )
            input = torch.zeros(batch_size, 100, device=device)
            codec(input)
        except Exception as e:
            logging.warning(e)

    # WavLM + K-means
    for layer_ids in WavLMKmeans.LAYER_IDS:
        try:
            codec = WavLMKmeans(sample_rate, layer_ids=layer_ids).eval().to(device)
            input = torch.zeros(batch_size, 100, device=device)
            codec(input)
        except Exception as e:
            logging.warning(e)

    # WavTokenizer
    try:
        sources = WavTokenizer.SOURCES
        configs = WavTokenizer.CONFIGS
        checkpoints = WavTokenizer.CHECKPOINTS
        for source, config, checkpoint in zip(sources, configs, checkpoints):
            codec = WavTokenizer(
                sample_rate, source=source, config=config, checkpoint=checkpoint
            ).eval()
            input = torch.zeros(batch_size, 100, device=device)
            codec(input)
    except Exception as e:
        logging.warning(e)

    # Metrics
    UTMOS(sample_rate)
    DNSMOS(sample_rate)
    DWER("small", sample_rate, device="cpu")
    SpkSimWavLM("microsoft/wavlm-base-sv", sample_rate)
    SpkSimECAPATDNN("speechbrain/spkrec-ecapa-voxceleb", sample_rate)


if __name__ == "__main__":
    download_weights()
