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

"""Recipe for training a speech enhancement system based on audio tokens.

To run this recipe:
> python train_se.py hparams/se/<dataset>/<config>.yaml

"""

import os
import sys
import warnings

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataio import write_audio
from speechbrain.utils.distributed import if_main_process


_CACHE = {"size": 0}


class SpeechEnhancement(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward pass."""
        batch = batch.to(self.device)
        in_sig, in_lens = batch.in_sig  # [B, T]
        out_sig, out_lens = batch.out_sig  # [B, T]

        # Augment if specified
        if stage == sb.Stage.TRAIN and self.hparams.augment:
            in_sig, in_lens = self.hparams.augmentation(in_sig, in_lens)

        # Extract tokens (cache them at first epoch if augmentation is disabled)
        key = tuple(sorted(batch.id))
        try:
            in_toks, out_toks = _CACHE[key]
            in_toks = in_toks.to(self.device)
            out_toks = out_toks.to(self.device)
        except KeyError:
            assert (in_lens == out_lens).all()
            sig = torch.cat([in_sig, out_sig])  # [B * 2, T]
            lens = torch.cat([in_lens, out_lens])  # [B * 2, T]
            with torch.no_grad():
                self.hparams.codec.eval().to(self.device)
                toks = self.hparams.codec.sig_to_toks(sig, lens)  # [B * 2, N, K]
            in_toks, out_toks = toks.split(
                [len(in_sig), len(out_sig)]
            )  # [B, N, K], [B, N, K]
            out_toks = (
                out_toks.clone()
            )  # Fix WavTokenizer bug ("Inference tensors cannot be saved for backward.")
            if stage != sb.Stage.TRAIN or (
                stage == sb.Stage.TRAIN and (not self.hparams.augment)
            ):
                if _CACHE["size"] < self.hparams.cache_size:
                    _CACHE[key] = in_toks.cpu(), out_toks.cpu()
                    _CACHE["size"] += in_toks.numel() + out_toks.numel()

        # Extract embeddings
        in_embs = self.modules.embedding(in_toks)  # [B, N, H, K]

        # Pooling
        in_embs = self.modules.pooling(in_embs)  # [B, N, H]

        # Forward encoder
        out_embs = self.modules.encoder.encode(in_embs, in_lens)  # [B, N, D]

        # Forward head
        log_probs = self.modules.head(out_embs).log_softmax(dim=-1)  # [B, N, K, C]

        return log_probs, out_toks

    def compute_objectives(self, predictions, batch, stage):
        """Computes the objectives."""
        log_probs, out_toks = predictions  # [B, N, K, C], [B, N, K]

        IDs = batch.id
        in_sig, _ = batch.in_sig
        out_sig, out_lens = batch.out_sig

        # Cross-entropy loss
        loss = self.hparams.ce_loss(
            log_probs.flatten(start_dim=1, end_dim=2),  # [B, NK, C]
            out_toks.flatten(start_dim=1),  # [B, NK]
            length=out_lens,
        )

        # Compute TER
        if stage != sb.Stage.TRAIN:
            self.ter_metric.append(
                IDs,
                log_probs.flatten(start_dim=1, end_dim=2),  # [B, NK, C]
                out_toks.flatten(start_dim=1),  # [B, NK]
                length=out_lens,
            )

        # Vocode
        if stage == sb.Stage.TEST and (
            self.hparams.compute_metrics or self.hparams.save_audios
        ):
            hyp_toks = log_probs.argmax(dim=-1)  # [B, N, K]
            self.vocode(IDs, in_sig, out_sig, hyp_toks, out_toks, out_lens)

        return loss

    def vocode(self, IDs, in_sig, out_sig, hyp_toks, out_toks, lens):
        with torch.no_grad():
            self.hparams.codec.eval().to(self.device)
            hyp_sig = self.hparams.codec.toks_to_sig(hyp_toks, lens)  # [B, T]
            rec_sig = self.hparams.codec.toks_to_sig(out_toks, lens)  # [B, T]

        # Adjust length
        if out_sig.shape[-1] > hyp_sig.shape[-1]:
            pad = [0, out_sig.shape[-1] - hyp_sig.shape[-1]]
            hyp_sig = torch.nn.functional.pad(
                hyp_sig, pad, mode="replicate"
            )  # [B, T_out]
            rec_sig = torch.nn.functional.pad(
                rec_sig, pad, mode="replicate"
            )  # [B, T_out]
        elif out_sig.shape[-1] < hyp_sig.shape[-1]:
            hyp_sig = hyp_sig.narrow(-1, 0, out_sig.shape[-1])  # [B, T_out]
            rec_sig = rec_sig.narrow(-1, 0, out_sig.shape[-1])  # [B, T_out]

        if self.hparams.compute_metrics:
            self.utmos_metric.append(IDs, hyp_sig, lens)
            self.rec_utmos_metric.append(IDs, rec_sig, lens)
            self.ref_utmos_metric.append(IDs, out_sig, lens)

            self.dnsmos_metric.append(IDs, hyp_sig, lens)
            self.rec_dnsmos_metric.append(IDs, rec_sig, lens)
            self.ref_dnsmos_metric.append(IDs, out_sig, lens)

            self.stoi_metric.append(IDs, hyp_sig, out_sig, lens)
            self.rec_stoi_metric.append(IDs, rec_sig, out_sig, lens)

            self.pesq_metric.append(IDs, hyp_sig, out_sig, lens)
            self.rec_pesq_metric.append(IDs, rec_sig, out_sig, lens)

            self.meld_metric.append(IDs, hyp_sig, out_sig, lens)
            self.rec_meld_metric.append(IDs, rec_sig, out_sig, lens)

            self.stftd_metric.append(IDs, hyp_sig, out_sig, lens)
            self.rec_stftd_metric.append(IDs, rec_sig, out_sig, lens)

            self.dwer_metric.append(IDs, hyp_sig, out_sig, lens)
            self.rec_dwer_metric.append(IDs, rec_sig, out_sig, lens)

            self.wavlm_sim_metric.append(IDs, hyp_sig, out_sig, lens)
            self.rec_wavlm_sim_metric.append(IDs, rec_sig, out_sig, lens)

            self.ecapatdnn_sim_metric.append(IDs, hyp_sig, out_sig, lens)
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

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch."""
        super().on_stage_start(stage, epoch)
        if stage != sb.Stage.TRAIN:
            self.ter_metric = self.hparams.ter_computer()
        if stage == sb.Stage.TEST and self.hparams.compute_metrics:
            self.utmos_metric = self.hparams.utmos_computer()
            self.rec_utmos_metric = self.hparams.utmos_computer(
                model=self.utmos_metric.model
            )
            self.ref_utmos_metric = self.hparams.utmos_computer(
                model=self.utmos_metric.model
            )

            self.dnsmos_metric = self.hparams.dnsmos_computer()
            self.rec_dnsmos_metric = self.hparams.dnsmos_computer(
                model=self.dnsmos_metric.model
            )
            self.ref_dnsmos_metric = self.hparams.dnsmos_computer(
                model=self.dnsmos_metric.model
            )

            self.stoi_metric = self.hparams.stoi_computer()
            self.rec_stoi_metric = self.hparams.stoi_computer()

            self.pesq_metric = self.hparams.pesq_computer()
            self.rec_pesq_metric = self.hparams.pesq_computer()

            self.meld_metric = self.hparams.meld_computer()
            self.rec_meld_metric = self.hparams.meld_computer()

            self.stftd_metric = self.hparams.stftd_computer()
            self.rec_stftd_metric = self.hparams.stftd_computer()

            self.dwer_metric = self.hparams.dwer_computer()
            self.rec_dwer_metric = self.hparams.dwer_computer(
                model=self.dwer_metric.model
            )

            self.wavlm_sim_metric = self.hparams.wavlm_sim_computer()
            self.rec_wavlm_sim_metric = self.hparams.wavlm_sim_computer(
                model=self.wavlm_sim_metric.model
            )

            self.ecapatdnn_sim_metric = self.hparams.ecapatdnn_sim_computer()
            self.rec_ecapatdnn_sim_metric = self.hparams.ecapatdnn_sim_computer(
                model=self.ecapatdnn_sim_metric.model
            )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of each epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}

        # Save cache
        cache_path = os.path.join(
            self.hparams.save_folder, f"cache_bs{self.hparams.train_batch_size}.pt"
        )
        torch.save(_CACHE, cache_path)

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["TER"] = self.ter_metric.summarize("average") * 100

        # Perform end-of-iteration operations, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            _, lr = self.hparams.scheduler(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, lr)
            steps = self.optimizer_step
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr, "steps": steps},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            if if_main_process():
                self.checkpointer.save_and_keep_only(
                    meta={"TER": stage_stats["TER"]},
                    min_keys=["TER"],
                    num_to_keep=self.hparams.keep_checkpoints,
                )

        elif stage == sb.Stage.TEST:
            if self.hparams.compute_metrics:
                stage_stats["UTMOS"] = self.utmos_metric.summarize("average")
                stage_stats["RecUTMOS"] = self.rec_utmos_metric.summarize("average")
                stage_stats["RefUTMOS"] = self.ref_utmos_metric.summarize("average")

                stage_stats["DNSMOS"] = self.dnsmos_metric.summarize("average")
                stage_stats["RecDNSMOS"] = self.rec_dnsmos_metric.summarize("average")
                stage_stats["RefDNSMOS"] = self.ref_dnsmos_metric.summarize("average")

                stage_stats["STOI"] = self.stoi_metric.summarize("average")
                stage_stats["RecSTOI"] = self.rec_stoi_metric.summarize("average")

                stage_stats["PESQ"] = self.pesq_metric.summarize("average")
                stage_stats["RecPESQ"] = self.rec_pesq_metric.summarize("average")

                stage_stats["MelD"] = self.meld_metric.summarize("average")
                stage_stats["RecMelD"] = self.rec_meld_metric.summarize("average")

                stage_stats["STFTD"] = self.stftd_metric.summarize("average")
                stage_stats["RecSTFTD"] = self.rec_stftd_metric.summarize("average")

                stage_stats["dWER"] = self.dwer_metric.summarize("error_rate")
                stage_stats["dCER"] = self.dwer_metric.summarize("error_rate_char")
                stage_stats["RecdWER"] = self.rec_dwer_metric.summarize("error_rate")
                stage_stats["RecdCER"] = self.rec_dwer_metric.summarize(
                    "error_rate_char"
                )

                stage_stats["WavLMSim"] = self.wavlm_sim_metric.summarize("average")
                stage_stats["RecWavLMSim"] = self.rec_wavlm_sim_metric.summarize(
                    "average"
                )

                stage_stats["ECAPATDNNSim"] = self.ecapatdnn_sim_metric.summarize(
                    "average"
                )
                stage_stats["RecECAPATDNNSim"] = (
                    self.rec_ecapatdnn_sim_metric.summarize("average")
                )
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                # Save dWER
                if self.hparams.compute_metrics:
                    dwer_file = os.path.join(self.hparams.output_folder, "dwer.txt")
                    with open(dwer_file, "w") as w:
                        self.dwer_metric.write_stats(w)

                if self.hparams.compute_metrics:
                    dwer_file = os.path.join(self.hparams.output_folder, "rec_dwer.txt")
                    with open(dwer_file, "w") as w:
                        self.rec_dwer_metric.write_stats(w)


if __name__ == "__main__":
    # Command-line interface
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
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

    # Prepare recipe
    from utils import prepare_recipe

    hparams, train_data, valid_data, test_data = prepare_recipe(hparams, run_opts)

    # Use pretrained embeddings
    if hparams["pretrain_embeddings"]:
        embs = hparams["codec"].to(run_opts.get("device", "cpu")).embs()
        embs = embs.detach().flatten(end_dim=-2)
        hparams["embedding"].weight.data.copy_(embs)

    # Log number of parameters/buffers
    codec_params = sum([x.numel() for x in hparams["codec"].state_dict().values()])
    model_params = sum(
        [
            x.numel()
            for module in hparams["modules"].values()
            for x in module.state_dict().values()
        ]
    )
    hparams["train_logger"].log_stats(
        stats_meta={
            f"Codec parameters/buffers (M)": f"{codec_params / 1e6:.2f}",
            "Model parameters/buffers (M)": f"{model_params / 1e6:.2f}",
        },
    )

    # Cache initialization
    cache_path = os.path.join(
        hparams["save_folder"], f"cache_bs{hparams['train_batch_size']}.pt"
    )
    if os.path.exists(cache_path):
        _CACHE.update(torch.load(cache_path))

    # Trainer initialization
    brain = SpeechEnhancement(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Train
    brain.fit(
        brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_kwargs"],
        valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
    )

    # Test
    brain.evaluate(
        test_data,
        min_key="TER",
        test_loader_kwargs=hparams["test_dataloader_kwargs"],
    )
