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

"""DyCAST (see https://arxiv.org/abs/2601.23174)."""

from typing import Any, Dict, Literal

import torch
import torchaudio

from audiocodecs.codec import Codec


__all__ = ["DyCAST"]


class DyCAST(Codec):
    CONFIGS = ["lucadellalib/dycast"]

    def __init__(
        self,
        sample_rate,
        num_codebooks=32,
        vocab_size=4,
        mode="reconstruct",
        config="lucadellalib/dycast",
        # Inference mode
        boundary_source: Literal[
            "char_aligner", "boundary_decode", "boundary_sample"
        ] = "boundary_decode",
        duration_source: Literal[
            "original", "duration_decode", "duration_sample"
        ] = "duration_decode",
        budget_decode: "bool" = False,
        # Retrieval
        use_retriever: "bool" = False,
        sim_threshold: "float" = 0.97,
        blend: "float" = 1.0,
        # Component kwargs
        aligner_kwargs: "Dict[str, Any]" = None,
        boundary_predictor_kwargs: "Dict[str, Any]" = None,
        use_wavenext_checkpoint: "bool" = False,
    ):
        super().__init__(sample_rate, 16000, mode)
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

        self.boundary_source = boundary_source
        self.duration_source = duration_source
        self.budget_decode = budget_decode
        self.use_retriever = use_retriever
        self.sim_threshold = sim_threshold
        self.blend = blend
        self.aligner_kwargs = aligner_kwargs or {}
        self.boundary_predictor_kwargs = boundary_predictor_kwargs or {}
        self.use_wavenext_checkpoint = use_wavenext_checkpoint

        # Minimal rule: if we need to propagate durations, require K == 33 (32 + dur)
        if mode != "encode" and duration_source == "original" and num_codebooks != 33:
            raise ValueError(
                "When duration_source='original', set num_codebooks=33 "
                "(32 token channels + 1 duration channel)."
            )

        overrides = {}
        if boundary_source != "char_aligner":
            overrides["char_aligner_name"] = None

        if not use_retriever:
            overrides["retriever_name"] = None

        self.model = torch.hub.load(
            "lucadellalib/dycast",
            "dycast",
            config=config,
            overrides=overrides,
        )
        self.sample_rate_input = self.model.sample_rate_input
        self.sample_rate_output = self.model.sample_rate_output

        if use_wavenext_checkpoint:
            wavenext = torch.hub.load(
                "lucadellalib/focalcodec",
                "focalcodec",
                config="lucadellalib/focalcodec_50hz_2k_causal",
            ).decoder
            self.model.decoder = wavenext

        if boundary_source != "char_aligner":
            self.model.char_aligner = None

        if duration_source == "original":
            self.model.duration_predictor = None

        if mode == "encode":
            self.model.decoder = None
            self.model.retriever = None

        if not use_retriever:
            self.model.retriever = None

        if mode == "decode":
            self.model.encoder = None
            self.model.compressor = None
            self.model.char_aligner = None
            self.model.boundary_predictor = None

    def toks_to_codes(self, toks, length):
        codes = self.model.toks_to_pooled_codes(toks)
        return codes

    # override
    def embs(self):
        return self.model.codebook

    # override
    def _sig_to_toks(self, sig, length):
        feats = self.model.sig_to_feats(sig, length=length)
        T = feats.shape[1]
        if length is None:
            self._cached_num_frames = torch.full(
                (len(feats),), fill_value=T, device=device
            )
        else:
            self._cached_num_frames = (
                (length * float(T)).ceil().clamp(0, T).to(dtype=torch.long)
            )
        self._cached_sig = sig

        # durations source for encode/reconstruct
        if self.boundary_source == "char_aligner":
            durs_enc = self.model.sig_to_durs(
                sig,
                length=length,
                **self.aligner_kwargs,
            )
        else:
            durs_enc = self.model.feats_to_durs(
                feats,
                length=length,
                sample=(self.boundary_source == "boundary_sample"),
                **self.boundary_predictor_kwargs,
            )

        lats = self.model.feats_to_lats(feats)
        plats, plats_length = self.model.lats_to_plats(lats, durs_enc)
        toks = self.model.plats_to_toks(plats)

        if self.duration_source == "original":
            toks = torch.cat(
                [
                    toks.to(dtype=torch.long),
                    durs_enc.to(dtype=torch.long)[..., None],
                ],
                dim=-1,
            )

        return toks  # [B, U, K]

    # override
    def _sig_to_feats(self, sig, length):
        feats = self.model.sig_to_feats(sig, length=length)
        return feats

    # override
    def _sig_to_qfeats(self, sig, length):
        toks = self._sig_to_toks(sig, length)
        qfeats = self._toks_to_qfeats(toks, length)
        return qfeats

    # override
    def _toks_to_sig(self, toks, length):
        qfeats = self._toks_to_qfeats(toks, length)
        sig = self._feats_to_sig(qfeats, length)
        return sig

    # override
    def _toks_to_qfeats(self, toks, length):
        if self.duration_source == "original":
            toks, durs_dec = (
                toks[..., : self.num_codebooks - 1],
                toks[..., self.num_codebooks - 1],
            )

        pcodes = self.model.toks_to_pcodes(toks)
        if self.duration_source != "original":
            duration_predictor_kwargs = {}
            if self.budget_decode:
                duration_predictor_kwargs["num_frames"] = self._cached_num_frames
            durs_dec = self.model.pcodes_to_durs(
                pcodes,
                length=length,
                sample=(self.duration_source == "duration_sample"),
                **duration_predictor_kwargs,
            )
        codes, _codes_length = self.model.pcodes_to_codes(pcodes, durs_dec)
        qfeats = self.model.codes_to_qfeats(codes)
        return qfeats

    # override
    def _feats_to_sig(self, feats, length):
        feats_dec = feats
        if self.use_retriever:
            feats_dec = self.model.qfeats_to_feats(
                feats_dec,
                sim_threshold=self.sim_threshold,
                blend=self.blend,
            )
        sig = self.model.feats_to_sig(feats_dec)
        if self.use_wavenext_checkpoint:
            sig = torchaudio.functional.resample(sig, 24000, 16000)
        return sig

    # override
    def _feats_to_toks(self, feats, length):
        # durations source for encode/reconstruct
        if self.boundary_source == "char_aligner":
            durs_enc = self.model.sig_to_durs(
                self._cached_sig,
                length=length,
                **self.aligner_kwargs,
            )
        else:
            durs_enc = self.model.feats_to_durs(
                feats,
                length=length,
                sample=(self.boundary_source == "boundary_sample"),
                **self.boundary_predictor_kwargs,
            )
        lats = self.model.feats_to_lats(feats)
        plats, plats_length = self.model.lats_to_plats(lats, durs_enc)
        toks = self.model.plats_to_toks(plats)
        return toks


# Test
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 10000
    batch_size = 2
    num_codebooks = 32

    boundary_source = "char_aligner"
    duration_source = "duration_decode"
    if duration_source == "original":
        num_codebooks += 1

    for mode in ["encode", "decode", "reconstruct"]:
        codec = (
            DyCAST(
                sample_rate,
                mode=mode,
                num_codebooks=num_codebooks,
                boundary_source=boundary_source,
                duration_source=duration_source,
            )
            .eval()
            .to(device)
        )
        input = (
            torch.ones(batch_size, 10, num_codebooks).long()
            if mode == "decode"
            else torch.randn(batch_size, sample_rate)
        ).to(device)
        with torch.no_grad():
            output = codec(input)
            print(output.shape)
            if mode in ["encode", "reconstruct"]:
                output = codec.sig_to_feats(input)
                print(output.shape)
                output = codec.sig_to_qfeats(input)
                print(output.shape)

    sig, sample_rate = torchaudio.load("example.wav")
    codec = DyCAST(
        sample_rate,
        num_codebooks=num_codebooks,
        boundary_source=boundary_source,
        duration_source=duration_source,
        budget_decode=True,
        use_retriever=False,
    ).eval()
    with torch.no_grad():
        rec_sig = codec(sig)
    print(sig.shape)
    print(rec_sig.shape)
    torchaudio.save("reconstruction.wav", rec_sig, sample_rate)
