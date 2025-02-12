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

"""SemantiCodec (see https://arxiv.org/abs/2405.00233)."""

import os
import sys

import torch

from audiocodecs.codec import Codec


__all__ = ["SemantiCodec"]


_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


class SemantiCodec(Codec):
    TOKEN_RATES = [25, 50, 100]
    SEMANTIC_VOCAB_SIZES = [4096, 8192, 16384, 32768]

    def __init__(
        self,
        sample_rate,
        mode="reconstruct",
        token_rate=100,
        semantic_vocab_size=8192,
        ddim_sample_step=50,
        cfg_scale=2.0,
    ):
        try:
            # Workaround to avoid name collisions with installed modules
            root_dir = os.path.dirname(os.path.realpath(__file__))
            sys_path = [x for x in sys.path]
            sys.path = [x for x in sys.path if root_dir not in x]
            import semanticodec

            global semanticodec

            sys.path = sys_path
        except ImportError:
            raise ImportError(
                "`pip install git+https://github.com/haoheliu/SemantiCodec-inference.git` to use this module"
            )

        super().__init__(sample_rate, 16000, mode)
        self.token_rate = token_rate
        self.semantic_vocab_size = semantic_vocab_size
        self.cfg_scale = cfg_scale
        self.num_codebooks = 2
        self.acoustic_vocab_size = 8192

        self.model = semanticodec.SemantiCodec(
            token_rate=token_rate,
            semantic_vocab_size=semantic_vocab_size,
            ddim_sample_step=ddim_sample_step,
            cfg_scale=cfg_scale,
            cache_path=_CACHE_DIR,
        ).to("cpu")

        if mode == "encode":
            self.model.decoder = None

    # override
    @torch.no_grad()
    def to(self, *args, **kwargs):
        self.model.encoder.centroid_npy = self.model.encoder.centroid_npy.to(
            *args, **kwargs
        )
        return super().to(*args, **kwargs)

    # override
    @torch.no_grad()
    def embs(self):
        if self.semantic_vocab_size != 8192:
            raise NotImplementedError("The size of acoustic codebook is fixed to 8192")
        device = next(iter(self.model.state_dict().values())).device
        toks = torch.arange(self.semantic_vocab_size, device=device)
        toks = (
            toks[:, None, None].expand(-1, -1, self.num_codebooks).clone()
        )  # [C, 1, K]
        embs = self._token_to_quantized_feature(toks)
        embs = torch.cat(
            embs.split(embs.shape[-1] // self.num_codebooks, dim=-1), dim=-2
        )  # [C, K, H]
        embs = embs.movedim(0, 1)  # [K, C, H]
        return embs

    # override
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        toks = self._encode(sig)  # [B, N, K]
        return toks

    # override
    def _sig_to_feats(self, sig, length):
        # sig: [B, T]
        feats = self._encode_unquantized(sig)  # [B, N, K]
        return feats

    # override
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K]
        sig = self._decode(toks)[:, 0]  # [B, T]
        return sig

    # See https://github.com/haoheliu/SemantiCodec-inference/blob/8dc464c3385d2389a695ed3f718f4a0caf3ed33f/semanticodec/main.py
    def _token_to_quantized_feature(self, tokens):
        semantic_tokens, acoustic_tokens = tokens[..., 0], tokens[..., 1]
        semantic_feature = self.model.encoder.unquant(semantic_tokens)
        token_num, feature_dim = semantic_feature.shape[-2], semantic_feature.shape[-1]
        acoustic_feature = self.model.encoder.quantizer.get_output_from_indices(
            acoustic_tokens
        ).reshape(-1, token_num, feature_dim)
        return torch.cat([acoustic_feature, semantic_feature], dim=-1)

    def _encode(self, waveform):
        # Calculate the original duration
        original_duration = waveform.shape[1] / semanticodec.main.SAMPLE_RATE
        # This is to pad the audio to the multiplication of 0.16 seconds so that the original audio can be reconstructed
        original_duration = original_duration + (
            semanticodec.main.AUDIOMAE_PATCH_DURATION
            - original_duration % semanticodec.main.AUDIOMAE_PATCH_DURATION
        )
        # Calculate the token length in theory
        target_token_len = (
            8
            * original_duration
            / semanticodec.main.AUDIOMAE_PATCH_DURATION
            / self.model.stack_factor_K
        )
        segment_sample_length = int(
            semanticodec.main.SAMPLE_RATE * semanticodec.main.SEGMENT_DURATION
        )
        # Pad audio to the multiplication of 10.24 seconds for easier segmentations

        if waveform.shape[1] % segment_sample_length < segment_sample_length:
            diff = int(
                segment_sample_length - waveform.shape[1] % segment_sample_length
            )
            waveform = torch.nn.functional.pad(waveform, [0, diff])

        mel_target_length = semanticodec.main.MEL_TARGET_LENGTH * int(
            waveform.shape[1] / segment_sample_length
        )
        # Calculate the mel spectrogram
        mels = [
            semanticodec.main.extract_kaldi_fbank_feature(
                x[None], semanticodec.main.SAMPLE_RATE, target_length=mel_target_length
            )["ta_kaldi_fbank"]
            for x in waveform
        ]
        mel = torch.stack(mels)
        assert mel.shape[-1] == 128 and mel.shape[-2] % 1024 == 0
        tokens = self.model.encoder(mel.to(waveform.device))
        tokens = tokens[:, : semanticodec.main.math.ceil(target_token_len), :]
        return tokens

    def _encode_unquantized(self, waveform):
        # Calculate the original duration
        original_duration = waveform.shape[1] / semanticodec.main.SAMPLE_RATE
        # This is to pad the audio to the multiplication of 0.16 seconds so that the original audio can be reconstructed
        original_duration = original_duration + (
            semanticodec.main.AUDIOMAE_PATCH_DURATION
            - original_duration % semanticodec.main.AUDIOMAE_PATCH_DURATION
        )
        # Calculate the token length in theory
        target_token_len = (
            8
            * original_duration
            / semanticodec.main.AUDIOMAE_PATCH_DURATION
            / self.model.stack_factor_K
        )
        segment_sample_length = int(
            semanticodec.main.SAMPLE_RATE * semanticodec.main.SEGMENT_DURATION
        )
        # Pad audio to the multiplication of 10.24 seconds for easier segmentations

        if waveform.shape[1] % segment_sample_length < segment_sample_length:
            diff = int(
                segment_sample_length - waveform.shape[1] % segment_sample_length
            )
            waveform = torch.nn.functional.pad(waveform, [0, diff])

        mel_target_length = semanticodec.main.MEL_TARGET_LENGTH * int(
            waveform.shape[1] / segment_sample_length
        )
        # Calculate the mel spectrogram
        mels = [
            semanticodec.main.extract_kaldi_fbank_feature(
                x[None], semanticodec.main.SAMPLE_RATE, target_length=mel_target_length
            )["ta_kaldi_fbank"]
            for x in waveform
        ]
        mel = torch.stack(mels)
        assert mel.shape[-1] == 128 and mel.shape[-2] % 1024 == 0
        feats = self._encoder_forward(mel.to(waveform.device))
        feats = feats[:, : semanticodec.main.math.ceil(target_token_len), :]
        return feats

    def _decode(self, tokens):
        windowed_token_list = self.model.encoder.long_token_split_window(
            tokens,
            window_length=int(512 / self.model.stack_factor_K),
            overlap=semanticodec.main.SEGMENT_OVERLAP_RATIO,
        )
        windowed_waveform = []
        for _, windowed_token in enumerate(windowed_token_list):
            latent = self._token_to_quantized_feature(windowed_token)
            latent = torch.cat(
                [
                    latent,
                    torch.ones(
                        latent.shape[0],
                        int(512 / self.model.stack_factor_K) - latent.shape[1],
                        latent.shape[2],
                    ).to(latent.device)
                    * -1,
                ],
                dim=1,
            )
            waveform = self.model.decoder.generate_sample(
                latent,
                ddim_steps=self.model.ddim_sample_step,
                unconditional_guidance_scale=self.model.cfg_scale,
            )
            windowed_waveform.append(waveform)
        output = semanticodec.main.overlap_add_waveform(
            windowed_waveform,
            overlap_duration=semanticodec.main.SEGMENT_DURATION
            * semanticodec.main.SEGMENT_OVERLAP_RATIO,
        )
        # Each patch step equal 16 mel time frames, which have 0.01 second
        trim_duration = (tokens.shape[1] / 8) * 16 * 0.01 * self.model.stack_factor_K
        return torch.as_tensor(
            output[..., : int(trim_duration * semanticodec.main.SAMPLE_RATE)],
            device=tokens.device,
        )

    def _encoder_forward(self, batch):
        # Perform padding before this function
        # Trim the audio token after this function
        assert batch.size(-1) == 128 and batch.size(-2) % 1024 == 0
        if self.model.encoder.device is None:
            self.model.encoder.device = batch.device
            self.model.encoder.centroid_npy = self.model.encoder.centroid_npy.to(
                self.model.encoder.device
            )

        window_length = 1024
        current_start = 0
        total_length_batch = batch.size(-2)

        feats_list = []
        while current_start + window_length <= total_length_batch:
            current_batch = batch[:, current_start : current_start + window_length, :]
            with torch.no_grad():
                # [bs, 513, 768]
                output = self._encoder_forward_inner(current_batch)
                feats_list.append(output)
            current_start += window_length
        return torch.cat(feats_list, dim=1)

    def _encoder_forward_inner(self, batch):
        assert batch.size(-2) == 1024 and batch.size(-1) == 128

        if self.model.encoder.device is None:
            self.model.encoder.device = batch.device
            self.model.encoder.centroid_npy = self.model.encoder.centroid_npy.to(
                self.model.encoder.device
            )

        batch = batch.unsqueeze(1)

        padding_cutoff_index = []
        temporal_dim = batch.shape[-2]
        for i in range(batch.shape[0]):
            active_index = (
                torch.std(batch[i, 0], dim=-1) <= 1e-7
            )  # F F T T F F T T T T T
            # If there are empty segment in the audio or there are padding in the audio
            try:
                if active_index.any():
                    # Convert boolean tensor to integer tensor where False becomes 0
                    int_tensor = active_index == False
                    # Find indices where the tensor is False
                    false_indices = torch.nonzero(int_tensor, as_tuple=False).squeeze()
                    # Get the last index of False
                    # last_false_index = false_indices[-1].item() if false_indices.numel() > 0 else -1
                    if false_indices.numel() > 0:
                        last_false_index = false_indices[-1].item()
                    else:
                        last_false_index = -1
                    column_max = last_false_index + 1
                # If there are no any empty segment in the audio
                else:
                    column_max = temporal_dim
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(false_indices)
                print(false_indices.numel())
                column_max = 0

            padding_cutoff_index.append(column_max / temporal_dim)

        with torch.no_grad():
            # [bs, 513, 768]
            representation = self.model.encoder.audiomae(
                batch,
                no_mask=self.model.encoder.no_audiomae_mask,
                no_average=self.model.encoder.no_audiomae_average,
            )

            if self.model.encoder.downsampling_rate != 1:
                representation = self.model.encoder.concate(representation)
                representation = (
                    representation.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
                )
            else:
                representation = representation[:, 1:, :]

        return representation


# Test
if __name__ == "__main__":
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 10000
    batch_size = 2

    for mode in ["encode", "decode", "reconstruct"]:
        codec = SemantiCodec(sample_rate, mode=mode).eval().to(device)
        input = (
            torch.zeros(batch_size, 10, 2).long()
            if mode == "decode"
            else torch.randn(batch_size, sample_rate)
        ).to(device)
        with torch.no_grad():
            output = codec(input)
            print(output.shape)
            embs = codec.embs()
            print(embs.shape)
            if mode in ["encode", "reconstruct"]:
                output = codec.sig_to_feats(input)
                print(output.shape)

    sig, sample_rate = torchaudio.load("example.wav")
    codec = SemantiCodec(sample_rate).eval()
    with torch.no_grad():
        rec_sig = codec(sig)
    torchaudio.save("reconstruction.wav", rec_sig, sample_rate)
