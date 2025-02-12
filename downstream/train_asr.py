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

"""Recipe for training a speech recognition system based on audio tokens.

To run this recipe:
> python train_asr.py hparams/asr/<dataset>/<config>.yaml

"""

import os
import sys
import warnings

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import if_main_process


_CACHE = {"size": 0}


class AutomaticSpeechRecognition(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward pass."""
        batch = batch.to(self.device)
        in_sig, in_lens = batch.sig  # [B, T]
        out_toks, out_lens = batch.toks  # [B, L]

        # Augment if specified
        if stage == sb.Stage.TRAIN and self.hparams.augment:
            in_sig, in_lens = self.hparams.augmentation(in_sig, in_lens)

        # Extract tokens (cache them at first epoch if augmentation is disabled)
        key = tuple(sorted(batch.id))
        try:
            in_toks = _CACHE[key]
            in_toks = in_toks.to(self.device)
        except KeyError:
            with torch.no_grad():
                self.hparams.codec.eval().to(self.device)
                in_toks = self.hparams.codec.sig_to_toks(in_sig, in_lens)  # [B, N, K]
            if stage != sb.Stage.TRAIN or (
                stage == sb.Stage.TRAIN and (not self.hparams.augment)
            ):
                if _CACHE["size"] < self.hparams.cache_size:
                    _CACHE[key] = in_toks.cpu()
                    _CACHE["size"] += in_toks.numel()

        # Extract embeddings
        in_embs = self.modules.embedding(in_toks)  # [B, N, H, K]

        # Pooling
        in_embs = self.modules.pooling(in_embs)  # [B, N, H]

        # Forward encoder
        out_embs, _ = self.modules.encoder(in_embs, lengths=in_lens)  # [B, N, D]

        # Forward head
        log_probs = self.modules.head(out_embs).log_softmax(dim=-1)  # [B, N, C]

        return log_probs, out_toks, in_toks

    def compute_objectives(self, predictions, batch, stage):
        """Computes the objectives."""
        log_probs, out_toks, in_toks = predictions  # [B, N, C], [B, L], [B, N, K]

        IDs = batch.id
        _, in_lens = batch.sig
        _, out_lens = batch.toks

        # CTC loss
        assert log_probs.shape[1] >= out_toks.shape[1], (
            log_probs.shape[1],
            out_toks.shape[1],
        )
        loss = self.hparams.ctc_loss(log_probs, out_toks, in_lens, out_lens)

        # Decode
        if stage != sb.Stage.TRAIN:
            hyp_toks = sb.decoders.ctc_greedy_decode(
                log_probs, in_lens, blank_id=self.hparams.blank_id
            )
            hyp_wrd = [
                self.hparams.tokenizer.sp.decode_ids(x).split(" ") for x in hyp_toks
            ]
            out_wrd = [x.split(" ") for x in batch.wrd]
            self.wer_metric.append(IDs, hyp_wrd, out_wrd)
            self.cer_metric.append(IDs, hyp_wrd, out_wrd)
            for x, y in zip(in_toks.flatten(), log_probs.argmax(dim=-1).flatten()):
                self.freq_matrix[x, y] += 1

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch."""
        super().on_stage_start(stage, epoch)
        if stage != sb.Stage.TRAIN:
            self.wer_metric = self.hparams.wer_computer()
            self.cer_metric = self.hparams.cer_computer()
            self.freq_matrix = torch.zeros(
                (self.hparams.vocab_size, self.hparams.num_tokens + 1),
                device=self.device,
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
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")

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
                    meta={"WER": stage_stats["WER"]},
                    min_keys=["WER"],
                    num_to_keep=self.hparams.keep_checkpoints,
                )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                # Save WER
                wer_file = os.path.join(self.hparams.output_folder, "wer.txt")
                with open(wer_file, "w") as w:
                    self.wer_metric.write_stats(w)
                # Save frequency matrix
                freq_matrix_file = os.path.join(
                    self.hparams.output_folder, "freq_matrix.txt"
                )
                torch.save(
                    self.freq_matrix.cpu(), freq_matrix_file.replace(".txt", ".pt")
                )
                tok2freq, tok2char = self.freq_matrix.sort(
                    dim=-1, descending=True, stable=True
                )
                tok2char[tok2freq == 0] = -1
                tok2char = tok2char.cpu().tolist()
                with open(freq_matrix_file, "w") as w:
                    for row in tok2char:
                        line = [
                            (
                                self.hparams.tokenizer.sp.decode(x)
                                if x != self.hparams.blank_id
                                else "<B>"
                            )
                            for x in row
                            if x != -1
                        ]
                        w.write(f"{line}\n")


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
    brain = AutomaticSpeechRecognition(
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
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_kwargs"],
    )
