#!/usr/bin/env/python

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

"""Recipe for testing a voice conversion system based on audio tokens.

To run this recipe:
> python test_vc.py hparams/tasks/<config>.yaml hparams/codecs/<config>.yaml hparams/datasets/<config>.yaml

"""

import os
import sys
import time
import warnings

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataio import write_audio
from speechbrain.utils.distributed import if_main_process


class VoiceConversion(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward pass."""
        assert stage == sb.Stage.TEST

        batch = batch.to(self.device)
        in_sig, in_lens = batch.in_sig  # [B, T_1]
        out_sig, out_lens = batch.out_sig  # [B, T_2]

        # Extract tokens
        with torch.no_grad():
            self.hparams.codec.eval().to(self.device)
            ts = time.time()
            in_toks = self.hparams.codec.sig_to_toks(in_sig, in_lens)  # [B, N_1, K]
            torch.cuda.synchronize()
            self.process_time_encode += time.time() - ts
            out_toks = self.hparams.codec.sig_to_toks(out_sig, out_lens)  # [B, N_2, K]

        return in_toks, out_toks

    def compute_objectives(self, predictions, batch, stage):
        """Computes the objectives."""
        hyp_toks, out_toks = predictions  # [B, N_1, K], [B, N_2, K]

        IDs = batch.id
        in_sig, _ = batch.in_sig
        out_sig, out_lens = batch.out_sig
        spk_sigs = batch.spk_sigs

        # Vocode
        if self.hparams.compute_metrics or self.hparams.save_audios:
            assert out_lens.numel() == 1 and out_lens[0] == 1.0
            spk_sigs = spk_sigs[0]
            self.vocode(IDs, in_sig, out_sig, hyp_toks, out_toks, out_lens, spk_sigs)

        return torch.tensor(0.0, device=self.device)

    def vocode(self, IDs, in_sig, out_sig, hyp_toks, out_toks, lens, spk_sigs):
        # Select longest reference to have enough acoustic information
        max_length = 0
        spk_sig = None
        for x in spk_sigs:
            if len(x) <= max_length:
                continue
            max_length = len(x)
            spk_sig = x
        spk_sig = spk_sig[None].to(self.device)

        with torch.no_grad():
            self.hparams.codec.eval().to(self.device)
            ts = time.time()

            # Voice conversion
            if self.hparams.codec.__class__.__name__ == "BiCodec":
                # Stack semantic tokens from source with acoustic tokens from reference
                spk_toks = self.hparams.codec.sig_to_toks(spk_sig)  # [B, N_s, K]
                assert spk_toks.shape[1] >= 32
                acoustic_spk_toks = spk_toks[:, :32]
                semantic_toks = hyp_toks[:, 32:]
                hyp_toks = torch.cat(
                    [acoustic_spk_toks, semantic_toks], dim=-2
                )  # [B, N_1, K]
                hyp_sig = self.hparams.codec.toks_to_sig(hyp_toks, lens)  # [B, T]

            elif self.hparams.num_codebooks > 1:
                # Stack codebook 0 from source with codebooks 1:K from reference
                spk_toks = self.hparams.codec.sig_to_toks(spk_sig)  #  [B, N_s, K]
                if spk_toks.shape[1] > hyp_toks.shape[1]:
                    spk_toks = spk_toks.narrow(1, 0, hyp_toks.shape[1])
                elif spk_toks.shape[1] < hyp_toks.shape[1]:
                    delta = hyp_toks.shape[1] - spk_toks.shape[1]
                    pad = [0, 0, delta // 2, delta - delta // 2]
                    spk_toks = torch.nn.functional.pad(spk_toks, pad, mode="circular")
                hyp_toks = torch.cat(
                    [hyp_toks[..., :1], spk_toks[..., 1:]], dim=-1
                )  # [B, N_1, K]
                hyp_sig = self.hparams.codec.toks_to_sig(hyp_toks, lens)  # [B, T]

            else:
                # Single codebook, use k-NN
                spk_sigs = [spk_sig]
                matching_set = [
                    self.hparams.codec.sig_to_feats(sig).flatten(end_dim=-2)
                    for sig in spk_sigs
                ]
                matching_set = torch.cat(matching_set)
                hyp_feats = self.hparams.codec.toks_to_qfeats(hyp_toks)
                hyp_feats = knn(
                    hyp_feats, matching_set, self.hparams.topk, self.hparams.num_splits
                ).mean(dim=-2)
                hyp_sig = self.hparams.codec.feats_to_sig(hyp_feats)

            torch.cuda.synchronize()
            self.process_time_decode += time.time() - ts
            self.real_time += len(hyp_sig[0]) / 16000
            rec_sig = self.hparams.codec.toks_to_sig(out_toks, lens)  # [B, T]

        # Adjust length
        if out_sig.shape[-1] > hyp_sig.shape[-1]:
            pad = [0, out_sig.shape[-1] - hyp_sig.shape[-1]]
            hyp_sig = torch.nn.functional.pad(
                hyp_sig, pad, mode="replicate"
            )  # [B, T_out]
        elif out_sig.shape[-1] < hyp_sig.shape[-1]:
            hyp_sig = hyp_sig.narrow(-1, 0, out_sig.shape[-1])  # [B, T_out]

        if out_sig.shape[-1] > rec_sig.shape[-1]:
            pad = [0, out_sig.shape[-1] - rec_sig.shape[-1]]
            rec_sig = torch.nn.functional.pad(
                rec_sig, pad, mode="replicate"
            )  # [B, T_out]
        elif out_sig.shape[-1] < rec_sig.shape[-1]:
            rec_sig = rec_sig.narrow(-1, 0, out_sig.shape[-1])  # [B, T_out]

        if self.hparams.compute_metrics:
            self.utmos_metric.append(IDs, hyp_sig, lens)
            self.dnsmos_metric.append(IDs, hyp_sig, lens)
            self.stoi_metric.append(IDs, hyp_sig, out_sig, lens)
            self.pesq_metric.append(IDs, hyp_sig, out_sig, lens)
            self.meld_metric.append(IDs, hyp_sig, out_sig, lens)
            self.stftd_metric.append(IDs, hyp_sig, out_sig, lens)
            self.dwer_metric.append(IDs, hyp_sig, out_sig, lens)
            self.wavlm_sim_metric.append(IDs, hyp_sig, out_sig, lens)
            self.ecapatdnn_sim_metric.append(IDs, hyp_sig, out_sig, lens)
            self.codebook_util_metric.append(out_toks, lens)

            if self.hparams.compute_ref_metrics:
                self.rec_utmos_metric.append(IDs, rec_sig, lens)
                self.ref_utmos_metric.append(IDs, out_sig, lens)

                self.rec_dnsmos_metric.append(IDs, rec_sig, lens)
                self.ref_dnsmos_metric.append(IDs, out_sig, lens)

                self.rec_stoi_metric.append(IDs, rec_sig, out_sig, lens)

                self.rec_pesq_metric.append(IDs, rec_sig, out_sig, lens)

                self.rec_meld_metric.append(IDs, rec_sig, out_sig, lens)

                self.rec_stftd_metric.append(IDs, rec_sig, out_sig, lens)

                self.rec_dwer_metric.append(IDs, rec_sig, out_sig, lens)

                self.rec_wavlm_sim_metric.append(IDs, rec_sig, out_sig, lens)

                self.rec_ecapatdnn_sim_metric.append(IDs, rec_sig, out_sig, lens)

        if self.hparams.save_audios:
            save_folder = os.path.join(self.hparams.output_folder, "audios")
            os.makedirs(save_folder, exist_ok=True)
            for i in range(len(IDs)):
                write_audio(
                    os.path.join(save_folder, f"{IDs[i]}_hyp.wav"),
                    hyp_sig[i].cpu(),
                    self.hparams.sample_rate,
                )
                write_audio(
                    os.path.join(save_folder, f"{IDs[i]}_rec.wav"),
                    rec_sig[i].cpu(),
                    self.hparams.sample_rate,
                )
                write_audio(
                    os.path.join(save_folder, f"{IDs[i]}_ref.wav"),
                    out_sig[i].cpu(),
                    self.hparams.sample_rate,
                )
                write_audio(
                    os.path.join(save_folder, f"{IDs[i]}_in.wav"),
                    in_sig[i].cpu(),
                    self.hparams.sample_rate,
                )
                write_audio(
                    os.path.join(save_folder, f"{IDs[i]}_spk.wav"),
                    spk_sig[i].cpu(),
                    self.hparams.sample_rate,
                )

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch."""
        super().on_stage_start(stage, epoch)
        self.real_time = 0.0
        self.process_time_encode = 0.0
        self.process_time_decode = 0.0
        if self.hparams.compute_metrics:
            self.utmos_metric = self.hparams.utmos_computer()
            self.dnsmos_metric = self.hparams.dnsmos_computer()
            self.stoi_metric = self.hparams.stoi_computer()
            self.pesq_metric = self.hparams.pesq_computer()
            self.meld_metric = self.hparams.meld_computer()
            self.stftd_metric = self.hparams.stftd_computer()
            self.dwer_metric = self.hparams.dwer_computer()
            self.wavlm_sim_metric = self.hparams.wavlm_sim_computer()
            self.ecapatdnn_sim_metric = self.hparams.ecapatdnn_sim_computer()
            self.codebook_util_metric = self.hparams.codebook_util_computer()

            if self.hparams.compute_ref_metrics:
                self.rec_utmos_metric = self.hparams.utmos_computer(
                    model=self.utmos_metric.model
                )
                self.ref_utmos_metric = self.hparams.utmos_computer(
                    model=self.utmos_metric.model
                )

                self.rec_dnsmos_metric = self.hparams.dnsmos_computer(
                    model=self.dnsmos_metric.model
                )
                self.ref_dnsmos_metric = self.hparams.dnsmos_computer(
                    model=self.dnsmos_metric.model
                )

                self.rec_stoi_metric = self.hparams.stoi_computer()

                self.rec_pesq_metric = self.hparams.pesq_computer()

                self.rec_meld_metric = self.hparams.meld_computer()

                self.rec_stftd_metric = self.hparams.stftd_computer()

                self.rec_dwer_metric = self.hparams.dwer_computer(
                    model=self.dwer_metric.model
                )

                self.rec_wavlm_sim_metric = self.hparams.wavlm_sim_computer(
                    model=self.wavlm_sim_metric.model
                )

                self.rec_ecapatdnn_sim_metric = self.hparams.ecapatdnn_sim_computer(
                    model=self.ecapatdnn_sim_metric.model
                )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of each epoch."""
        # Compute/store important stats
        assert stage == sb.Stage.TEST
        stage_stats = {"loss": stage_loss}

        if self.hparams.compute_metrics:
            stage_stats["UTMOS"] = self.utmos_metric.summarize("average")
            stage_stats["DNSMOS"] = self.dnsmos_metric.summarize("average")
            stage_stats["STOI"] = self.stoi_metric.summarize("average")
            stage_stats["PESQ"] = self.pesq_metric.summarize("average")
            stage_stats["MelD"] = self.meld_metric.summarize("average")
            stage_stats["STFTD"] = self.stftd_metric.summarize("average")
            stage_stats["dWER"] = self.dwer_metric.summarize("error_rate")
            stage_stats["dCER"] = self.dwer_metric.summarize("error_rate_char")
            stage_stats["WavLMSim"] = self.wavlm_sim_metric.summarize("average")
            stage_stats["ECAPATDNNSim"] = self.ecapatdnn_sim_metric.summarize("average")
            stage_stats["CodebookUtil"] = self.codebook_util_metric.summarize(
                "codebook_util"
            )
            stage_stats["NormEntropy"] = self.codebook_util_metric.summarize(
                "norm_entropy"
            )
            stage_stats["RealTime"] = self.real_time
            stage_stats["ProcessTimeEncode"] = self.process_time_encode
            stage_stats["ProcessTimeDecode"] = self.process_time_decode
            stage_stats["RTF"] = (
                self.process_time_encode + self.process_time_decode
            ) / self.real_time
            stage_stats["iRTF"] = 1 / stage_stats["RTF"]

            if self.hparams.compute_ref_metrics:
                stage_stats["RecUTMOS"] = self.rec_utmos_metric.summarize("average")
                stage_stats["RefUTMOS"] = self.ref_utmos_metric.summarize("average")

                stage_stats["RecDNSMOS"] = self.rec_dnsmos_metric.summarize("average")
                stage_stats["RefDNSMOS"] = self.ref_dnsmos_metric.summarize("average")

                stage_stats["RecSTOI"] = self.rec_stoi_metric.summarize("average")

                stage_stats["RecPESQ"] = self.rec_pesq_metric.summarize("average")

                stage_stats["RecMelD"] = self.rec_meld_metric.summarize("average")

                stage_stats["RecSTFTD"] = self.rec_stftd_metric.summarize("average")

                stage_stats["RecdWER"] = self.rec_dwer_metric.summarize("error_rate")
                stage_stats["RecdCER"] = self.rec_dwer_metric.summarize(
                    "error_rate_char"
                )

                stage_stats["RecWavLMSim"] = self.rec_wavlm_sim_metric.summarize(
                    "average"
                )

                stage_stats["RecECAPATDNNSim"] = (
                    self.rec_ecapatdnn_sim_metric.summarize("average")
                )
        self.hparams.train_logger.log_stats(
            stats_meta={},
            test_stats=stage_stats,
        )
        if if_main_process():
            # Save dWER
            if self.hparams.compute_metrics:
                dwer_file = os.path.join(self.hparams.output_folder, "dwer.txt")
                with open(dwer_file, "w") as w:
                    self.dwer_metric.write_stats(w)

                if self.hparams.compute_ref_metrics:
                    dwer_file = os.path.join(self.hparams.output_folder, "rec_dwer.txt")
                    with open(dwer_file, "w") as w:
                        self.rec_dwer_metric.write_stats(w)


# Adapted from:
# https://github.com/bshall/knn-vc/blob/848302a262f7299c738af49d74209790ed442a9f/matcher.py#L21
def _cosine_distance(query, target):
    # [T, H], [M, K]
    source_norm2 = (query**2).sum(dim=-1)
    target_norm2 = (target**2).sum(dim=-1)
    dotprod = (
        source_norm2[:, None]
        + target_norm2[None]
        - torch.cdist(query[None], target[None])[0] ** 2
    )
    dotprod /= 2
    dists = 1 - dotprod * (source_norm2[:, None] * target_norm2[None]).rsqrt()
    return dists


def knn(input, matching_set, topk=4, num_splits=1):
    chunk_size = matching_set.shape[0] // num_splits
    if num_splits > 1:
        matching_subsets = matching_set.split(chunk_size)
    else:
        matching_subsets = [matching_set]
    topk_smallest_dists = []
    topk_smallest_idxes = []
    for i, matching_subset in enumerate(matching_subsets):
        dists = _cosine_distance(input.flatten(end_dim=-2), matching_subset)
        topk_smallest_dists_i, topk_smallest_idxes_i = dists.topk(
            k=min(topk, matching_subset.shape[0]), largest=False, dim=-1
        )
        topk_smallest_dists.append(topk_smallest_dists_i)
        topk_smallest_idxes.append(i * chunk_size + topk_smallest_idxes_i)
    if num_splits > 1:
        dists = torch.cat(topk_smallest_dists, dim=-1)
        idxes = torch.cat(topk_smallest_idxes, dim=-1)
        _, dist_idxes = dists.topk(k=min(topk, dists.shape[-1]), largest=False, dim=-1)
        output = matching_set[idxes.gather(1, dist_idxes)]
    else:
        output = matching_set[topk_smallest_idxes[0]]
    output = output.reshape(input.shape[:-1] + (-1, input.shape[-1]))
    return output


if __name__ == "__main__":
    # Command-line interface
    from utils import parse_arguments

    hparams_file, run_opts, overrides = parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Filter warnings
    warnings.filterwarnings("once")
    warnings.filterwarnings("ignore", module="torch")

    # If --distributed_launch then create ddp_init_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Log command and dump hyperpameters for reproducibility
    sb.core.logger.warn(f"Command: {' '.join(sys.argv)}")
    sb.core.shutil.copy(
        hparams_file, os.path.join(hparams["output_folder"], "config.yaml")
    )

    # Prepare recipe
    from utils import prepare_recipe

    hparams, _, _, test_data = prepare_recipe(hparams, run_opts)

    # Log number of parameters/buffers
    codec_params = sum([x.numel() for x in hparams["codec"].state_dict().values()])
    hparams["train_logger"].log_stats(
        stats_meta={
            f"Codec parameters/buffers (M)": f"{codec_params / 1e6:.2f}",
        },
    )

    # Trainer initialization
    brain = VoiceConversion(
        hparams=hparams,
        run_opts=run_opts,
    )

    # Test
    brain.evaluate(
        test_data,
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_kwargs"],
    )
