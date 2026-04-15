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

"""Recipe for training a speech language modeling system based on audio tokens.

To run this recipe:
> python train_slm.py hparams/tasks/<config>.yaml hparams/codecs/<config>.yaml hparams/datasets/<config>.yaml

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


class SpeechLanguageModeling(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward pass."""
        batch = batch.to(self.device)
        sig, lens = batch.sig  # [B, T]

        # Extract tokens (cache them at first epoch)
        key = tuple(sorted(batch.id))
        try:
            toks = _CACHE[key]
            toks = toks.to(self.device)
        except KeyError:
            with torch.no_grad():
                self.hparams.codec.eval().to(self.device)
                toks = self.hparams.codec.sig_to_toks(sig, lens)  # [B, N, K]
            if _CACHE["size"] < self.hparams.cache_size:
                _CACHE[key] = toks.cpu()
                _CACHE["size"] += toks.numel()

        # Prepare BOS tokens (flatten tokens along time dimension)
        toks_bos = torch.nn.functional.pad(
            toks.flatten(start_dim=-2),
            [1, 0],
            value=self.hparams.bos_id,
        )  # [B, 1 + NK]

        # Forward decoder
        embs_bos = self.modules.decoder.embed(toks_bos)  # [B, 1 + NK, H]
        logits, _ = self.modules.decoder(embs_bos)  # [B, NK + 1, C]
        log_probs = logits.log_softmax(dim=-1)  # [B, NK + 1, C]

        return log_probs, toks

    def compute_objectives(self, predictions, batch, stage):
        """Computes the objectives."""
        log_probs, toks = predictions  # [B, NK, C], [B, N, K], [B, L, H]

        IDs = batch.id
        sig, lens = batch.sig

        # Prepare EOS tokens (flatten tokens along time dimension)
        toks_eos = torch.nn.functional.pad(
            toks.flatten(start_dim=-2),
            [0, 1],
            value=self.hparams.eos_id,
        )  # [B, NK + 1]

        # Cross-entropy loss
        loss = self.hparams.ce_loss(log_probs, toks_eos, length=None)

        # Compute TER
        if stage != sb.Stage.TRAIN:
            self.ter_metric.append(IDs, log_probs, toks_eos)

        # Vocode
        if stage == sb.Stage.TEST and (
            self.hparams.compute_metrics or self.hparams.save_audios
        ):
            toks_bos = torch.nn.functional.pad(
                toks[:, : toks.shape[1] // 2].flatten(start_dim=-2),
                [1, 0],
                value=self.hparams.bos_id,
            )  # [B, 1 + (N // 2)K]
            hyp_toks = self.modules.decoder.generate(
                toks_bos,
                eos_id=-1,
                max_gen_toks=toks.shape[1:].numel() // 2,
                top_p=self.hparams.top_p,
                temp=self.hparams.temp,
                use_kv_cache=True,
            )  # B x [(N // 2)K]
            hyp_toks = torch.stack(hyp_toks)  # [B, (N // 2)K]
            hyp_toks = torch.cat([toks_bos[:, 1:], hyp_toks], dim=-1)  # [B, NK - J]
            if hyp_toks.shape[-1] < toks.shape[1:].numel():
                pad = [0, toks.shape[-2:].numel() - hyp_toks.shape[-1]]
                hyp_toks = torch.nn.functional.pad(
                    hyp_toks, pad, mode="replicate"
                )  # [B, NK]
            hyp_toks = hyp_toks.reshape(len(hyp_toks), -1, toks.shape[-1])  # [B, N, K]

            # Remove special tokens if any
            hyp_toks[hyp_toks >= self.hparams.vocab_size] = self.hparams.vocab_size - 1

            self.vocode(IDs, sig, hyp_toks, toks, lens)

        if stage == sb.Stage.TEST and len(toks) == 2:
            tgt_toks = toks.flatten(start_dim=-2).repeat_interleave(
                2, dim=0
            )  # A, A, B, B
            tgt_toks_1 = tgt_toks[:, : tgt_toks.shape[1] // 2]
            tgt_toks_2 = tgt_toks[:, tgt_toks.shape[1] // 2 :]
            tgt_toks_2 = tgt_toks_2.roll(1, dims=0)
            tgt_toks = torch.cat([tgt_toks_1, tgt_toks_2], dim=-1)  # AB, AA, BA, BB

            tgt_toks_bos = torch.nn.functional.pad(
                tgt_toks[:, :-1],
                [1, 0],
                value=self.hparams.bos_id,
            )  # [B, NK]

            # Forward decoder
            embs_bos = self.modules.decoder.embed(tgt_toks_bos)  # [B, 1 + NK, H]
            logits, _ = self.modules.decoder(embs_bos)  # [B, NK + 1, C]
            log_probs = logits.log_softmax(dim=-1)  # [B, NK + 1, C]
            log_probs = log_probs.gather(dim=-1, index=tgt_toks[..., None].long())[
                ..., 0
            ]
            tgt_lengths = (tgt_toks.shape[1] * lens).ceil().clamp(max=tgt_toks.shape[1])
            tgt_lengths = tgt_lengths.repeat_interleave(2, dim=0).roll(1, dims=0)
            mask = (
                torch.arange(tgt_lengths.max(), device=self.device)[None]
                < tgt_lengths[:, None]
            ).float()
            log_probs = (log_probs * mask).sum(dim=-1) / tgt_lengths
            log_probs = log_probs.reshape(2, 2)
            self.er_metric.append(
                IDs, log_probs, torch.tensor([1, 1], device=self.device)
            )

        return loss

    def vocode(self, IDs, sig, hyp_toks, toks, lens):
        with torch.no_grad():
            self.hparams.codec.eval().to(self.device)
            hyp_sig = self.hparams.codec.toks_to_sig(hyp_toks, lens)  # [B, T]
            rec_sig = self.hparams.codec.toks_to_sig(toks, lens)  # [B, T]

        # Split prompt / generated
        prompt_hyp_sig = sig[:, : -rec_sig.shape[-1] // 2]
        gen_hyp_sig = hyp_sig[:, -rec_sig.shape[-1] // 2 :]
        hyp_sig = torch.cat([prompt_hyp_sig, gen_hyp_sig], dim=-1)

        # Adjust length
        if prompt_hyp_sig.shape[-1] > gen_hyp_sig.shape[-1]:
            pad = [0, prompt_hyp_sig.shape[-1] - gen_hyp_sig.shape[-1]]
            gen_hyp_sig = torch.nn.functional.pad(
                gen_hyp_sig, pad, mode="replicate"
            )  # [B, T_out // 2]
        elif prompt_hyp_sig.shape[-1] < gen_hyp_sig.shape[-1]:
            gen_hyp_sig = gen_hyp_sig.narrow(
                -1, 0, prompt_hyp_sig.shape[-1]
            )  # [B, T_out // 2]

        if sig.shape[-1] > rec_sig.shape[-1]:
            pad = [0, sig.shape[-1] - rec_sig.shape[-1]]
            rec_sig = torch.nn.functional.pad(
                rec_sig, pad, mode="replicate"
            )  # [B, T_out]
        elif sig.shape[-1] < rec_sig.shape[-1]:
            rec_sig = rec_sig.narrow(-1, 0, sig.shape[-1])  # [B, T_out]

        if self.hparams.compute_metrics:
            self.utmos_metric.append(IDs, gen_hyp_sig, lens)
            self.dnsmos_metric.append(IDs, gen_hyp_sig, lens)
            self.perplexity_metric.append(IDs, hyp_sig, lens)
            self.wavlm_sim_metric.append(IDs, gen_hyp_sig, prompt_hyp_sig, lens)
            self.ecapatdnn_sim_metric.append(IDs, gen_hyp_sig, prompt_hyp_sig, lens)

            if self.hparams.compute_ref_metrics:
                self.rec_utmos_metric.append(IDs, rec_sig, lens)
                self.ref_utmos_metric.append(IDs, sig, lens)

                self.rec_dnsmos_metric.append(IDs, rec_sig, lens)
                self.ref_dnsmos_metric.append(IDs, sig, lens)

                self.rec_stoi_metric.append(IDs, rec_sig, sig, lens)

                self.rec_pesq_metric.append(IDs, rec_sig, sig, lens)

                self.rec_meld_metric.append(IDs, rec_sig, sig, lens)

                self.rec_stftd_metric.append(IDs, rec_sig, sig, lens)

                self.ref_perplexity_metric.append(IDs, sig, lens)

                self.rec_wavlm_sim_metric.append(IDs, rec_sig, sig, lens)

                self.rec_ecapatdnn_sim_metric.append(IDs, rec_sig, sig, lens)

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
                    os.path.join(save_folder, f"{IDs[i]}_prompt.wav"),
                    prompt_hyp_sig[i].cpu(),
                    self.hparams.sample_rate,
                )
                write_audio(
                    os.path.join(save_folder, f"{IDs[i]}_gen.wav"),
                    gen_hyp_sig[i].cpu(),
                    self.hparams.sample_rate,
                )
                write_audio(
                    os.path.join(save_folder, f"{IDs[i]}_rec.wav"),
                    rec_sig[i].cpu(),
                    self.hparams.sample_rate,
                )
                write_audio(
                    os.path.join(save_folder, f"{IDs[i]}_ref.wav"),
                    sig[i].cpu(),
                    self.hparams.sample_rate,
                )
                if self.hparams.compute_metrics:
                    with open(
                        os.path.join(save_folder, f"{IDs[i]}_hyp.txt"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(f"{self.perplexity_metric.texts[i]}\n")
                        f.write(str(self.perplexity_metric.perplexities[i]))

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch."""
        super().on_stage_start(stage, epoch)
        if stage != sb.Stage.TRAIN:
            self.ter_metric = self.hparams.ter_computer()
        if stage == sb.Stage.TEST and self.hparams.test_batch_size == 2:
            self.er_metric = self.hparams.er_computer()
        if stage == sb.Stage.TEST and self.hparams.compute_metrics:
            self.utmos_metric = self.hparams.utmos_computer()
            self.dnsmos_metric = self.hparams.dnsmos_computer()
            self.perplexity_metric = self.hparams.perplexity_computer()
            self.wavlm_sim_metric = self.hparams.wavlm_sim_computer()
            self.ecapatdnn_sim_metric = self.hparams.ecapatdnn_sim_computer()

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

                self.ref_perplexity_metric = self.hparams.perplexity_computer(
                    model=self.perplexity_metric.model,
                    asr_model=self.perplexity_metric.asr_model,
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
        stage_stats = {"loss": stage_loss}

        # Save cache
        cache_path = os.path.join(
            hparams["save_folder"], f"cache_bs{self.hparams.train_batch_size}.pt"
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
            if self.hparams.test_batch_size == 2:
                stage_stats["ER"] = self.er_metric.summarize("average") * 100

            if self.hparams.compute_metrics:
                stage_stats["UTMOS"] = self.utmos_metric.summarize("average")
                stage_stats["DNSMOS"] = self.dnsmos_metric.summarize("average")
                stage_stats["Perplexity"] = self.perplexity_metric.summarize("average")
                stage_stats["WavLMSim"] = self.wavlm_sim_metric.summarize("average")
                stage_stats["ECAPATDNNSim"] = self.ecapatdnn_sim_metric.summarize(
                    "average"
                )

                if self.hparams.compute_ref_metrics:
                    stage_stats["RecUTMOS"] = self.rec_utmos_metric.summarize("average")
                    stage_stats["RefUTMOS"] = self.ref_utmos_metric.summarize("average")

                    stage_stats["RecDNSMOS"] = self.rec_dnsmos_metric.summarize(
                        "average"
                    )
                    stage_stats["RefDNSMOS"] = self.ref_dnsmos_metric.summarize(
                        "average"
                    )

                    stage_stats["RecSTOI"] = self.rec_stoi_metric.summarize("average")

                    stage_stats["RecPESQ"] = self.rec_pesq_metric.summarize("average")

                    stage_stats["RecMelD"] = self.rec_meld_metric.summarize("average")

                    stage_stats["RecSTFTD"] = self.rec_stftd_metric.summarize("average")

                    stage_stats["RefPerplexity"] = self.ref_perplexity_metric.summarize(
                        "average"
                    )

                    stage_stats["RecWavLMSim"] = self.rec_wavlm_sim_metric.summarize(
                        "average"
                    )

                    stage_stats["RecECAPATDNNSim"] = (
                        self.rec_ecapatdnn_sim_metric.summarize("average")
                    )

            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


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

    hparams, train_data, valid_data, test_data = prepare_recipe(hparams, run_opts)

    # Use pretrained embeddings
    if hparams["pretrain_embeddings"]:
        embs = hparams["codec"].to(run_opts.get("device", "cpu")).embs()
        embs = torch.nn.functional.pad(embs.detach(), [0, 0, 0, 2]).flatten(end_dim=-2)
        hparams["decoder"].tok_embeddings.weight.data.copy_(embs)

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
            "Codec parameters/buffers (M)": f"{codec_params / 1e6:.2f}",
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
    brain = SpeechLanguageModeling(
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
