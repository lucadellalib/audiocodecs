#!/usr/bin/env/python

# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Recipe for training a text-to-speech system using discrete audio representations.

To run this recipe:
> python train_tts.py hparams/tts/<dataset>/<config>.yaml

"""

import os
import sys
import warnings

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataio import length_to_mask, write_audio
from speechbrain.utils.distributed import if_main_process


_TOK_CACHE = {"size": 0}

_SPK_CACHE = {"size": 0}


class TextToSpeech(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward pass."""
        batch = batch.to(self.device)
        in_toks, in_lens = batch.toks  # [B, L]
        out_sig, out_lens = batch.sig  # [B, T]

        # Extract tokens (cache them at first epoch)
        key = tuple(sorted(batch.id))
        try:
            out_toks = _TOK_CACHE[key]
            out_toks = out_toks.to(self.device)
        except KeyError:
            with torch.no_grad():
                self.hparams.codec.eval().to(self.device)
                out_toks = self.hparams.codec.sig_to_toks(
                    out_sig, out_lens
                )  # [B, N, K]
            if _TOK_CACHE["size"] < self.hparams.tok_cache_size:
                _TOK_CACHE[key] = out_toks.cpu()
                _TOK_CACHE["size"] += out_toks.numel()

        # Extract speaker embeddings (cache them at first epoch)
        try:
            spk_embs = _SPK_CACHE[key]
            spk_embs = spk_embs.to(self.device)
        except KeyError:
            with torch.no_grad():
                self.hparams.spk_encoder.eval().to(self.device)
                spk_embs = self.hparams.spk_encoder(out_sig, out_lens)  # [B, H_spk]
            if _SPK_CACHE["size"] < self.hparams.spk_cache_size:
                _SPK_CACHE[key] = spk_embs.cpu()
                _SPK_CACHE["size"] += spk_embs.numel()

        # Extract embeddings
        in_embs = self.modules.embedding(in_toks)  # [B, L, H]

        # Project speaker embeddings
        spk_embs = self.modules.spk_proj(spk_embs)  # [B, H]

        # Forward encoder-decoder (flatten tokens along time dimension)
        out_toks_bos = torch.nn.functional.pad(
            out_toks.flatten(start_dim=-2),
            [1, 0],
            value=self.hparams.bos_id,
        )  # [B, NK]
        enc_embs, dec_embs = self.modules.encoder_decoder(
            in_embs,
            out_toks_bos,
            in_lens,
            pad_idx=-1,
            extra_embs=spk_embs,
        )  # [B, NK, D]

        # Forward head
        log_probs = self.modules.head(dec_embs).log_softmax(dim=-1)  # [B, NK, C]

        return log_probs, out_toks, enc_embs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the objectives."""
        log_probs, out_toks, enc_embs = predictions  # [B, NK, C], [B, N, K], [B, L, H]

        IDs = batch.id
        wrd = batch.wrd
        _, in_lens = batch.toks
        out_sig, out_lens = batch.sig

        # Cross-entropy loss
        out_toks_eos = torch.nn.functional.pad(
            out_toks.flatten(start_dim=-2),
            [0, 1],
            value=self.hparams.eos_id,
        )
        out_lens = (out_lens * out_toks.shape[1:].numel() + 1) / (
            out_toks.shape[1:].numel() + 1
        )
        mask = length_to_mask(
            out_toks_eos.shape[1] * out_lens, max_len=out_toks_eos.shape[1]
        ).bool()
        out_toks_eos[~mask] = self.hparams.eos_id
        loss = self.hparams.ce_loss(log_probs, out_toks_eos, length=out_lens)

        # Vocode
        if stage == sb.Stage.TEST and self.hparams.compute_metrics:
            self.hparams.searcher.max_decode_ratio = (
                log_probs.shape[1] / enc_embs.shape[1]
            )
            hyp_toks, _, _, _ = self.hparams.searcher(enc_embs, in_lens)
            max_len = max([len(x) for x in hyp_toks])
            hyp_toks = [
                torch.nn.functional.pad(
                    torch.as_tensor(x, device=self.device), [0, max_len - len(x)]
                )
                for x in hyp_toks
            ]
            hyp_toks = torch.stack(hyp_toks)
            hyp_toks = hyp_toks.narrow(
                -1, 0, (hyp_toks.shape[1] // out_toks.shape[-1]) * out_toks.shape[-1]
            )  # [B, MK]
            hyp_toks = hyp_toks.reshape(
                len(hyp_toks), -1, out_toks.shape[-1]
            )  # [B, M, K]
            self.vocode(IDs, wrd, out_sig, hyp_toks, out_toks, out_lens)

        return loss

    def vocode(self, IDs, wrd, out_sig, hyp_toks, out_toks, lens):
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
        elif out_sig.shape[-1] < hyp_sig.shape[-1]:
            hyp_sig = hyp_sig.narrow(-1, 0, out_sig.shape[-1])  # [B, T_out]

        if out_sig.shape[-1] > rec_sig.shape[-1]:
            pad = [0, out_sig.shape[-1] - rec_sig.shape[-1]]
            rec_sig = torch.nn.functional.pad(
                rec_sig, pad, mode="replicate"
            )  # [B, T_out]
        elif out_sig.shape[-1] < rec_sig.shape[-1]:
            rec_sig = rec_sig.narrow(-1, 0, out_sig.shape[-1])  # [B, T_out]

        self.utmos_metric.append(IDs, hyp_sig, lens)
        self.rec_utmos_metric.append(IDs, rec_sig, lens)
        self.ref_utmos_metric.append(IDs, out_sig, lens)

        self.dnsmos_metric.append(IDs, hyp_sig, lens)
        self.rec_dnsmos_metric.append(IDs, rec_sig, lens)
        self.ref_dnsmos_metric.append(IDs, out_sig, lens)

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
                with open(
                    os.path.join(save_folder, f"{IDs[i]}_in.txt"), "w", encoding="utf-8"
                ) as f:
                    f.write(wrd[i])

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch."""
        super().on_stage_start(stage, epoch)
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

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

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
                    meta={"loss": stage_stats["loss"]},
                    min_keys=["loss"],
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
        embs = hparams["codec"].embs().flatten(end_dim=-2)
        hparams["embedding"].weight.data.copy_(embs.detach())

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

    # Trainer initialization
    brain = TextToSpeech(
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
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_kwargs"],
    )
