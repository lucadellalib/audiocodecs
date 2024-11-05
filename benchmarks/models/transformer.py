# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Transformer models."""

import torch
from speechbrain.lobes.models.transformer.Conformer import ConformerEncoder
from speechbrain.lobes.models.transformer.TransformerASR import (
    TransformerASR,
    make_transformer_src_tgt_masks,
)
from speechbrain.nnet.activations import Swish


__all__ = ["EncoderDecoderWithExtraEmbeddings"]


class EncoderDecoderWithExtraEmbeddings(TransformerASR):
    def __init__(self, **kwargs):
        injection_mode = kwargs.pop("injection_mode", "cat")
        super().__init__(**kwargs)
        if kwargs.get("encoder_module", "transformer") != "conformer":
            raise NotImplementedError
        self.encoder = ConformerEncoderWithExtraEmbeddings(
            nhead=kwargs.get("nhead", 8),
            num_layers=kwargs.get("num_encoder_layers", 6),
            d_ffn=kwargs.get("d_ffn", 2048),
            d_model=kwargs.get("d_model", 512),
            dropout=kwargs.get("dropout", 0.0),
            activation=kwargs.get("conformer_activation", Swish),
            kernel_size=kwargs.get("kernel_size", 31),
            bias=kwargs.get("bias", True),
            causal=self.causal,
            attention_type=self.attention_type,
            injection_mode=injection_mode,
        )

    def forward(self, src, tgt, wav_len=None, pad_idx=0, extra_embs=None):
        # reshape the src vector to [Batch, Time, Fea] is a 4d vector is given
        if src.ndim == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask,
        ) = make_transformer_src_tgt_masks(
            src, tgt, wav_len, causal=self.causal, pad_idx=pad_idx
        )

        src = self.custom_src_module(src)
        # add pos encoding to queries if are sinusoidal ones else
        if self.attention_type == "hypermixing":
            pos_embs_encoder = None
        elif self.attention_type == "RelPosMHAXL":
            pos_embs_encoder = self.positional_encoding(src)
        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(src)  # add the encodings here
            pos_embs_encoder = None

        encoder_out, _ = self.encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
            extra_embs=extra_embs,
        )

        tgt = self.custom_tgt_module(tgt)

        if self.attention_type == "RelPosMHAXL":
            tgt = tgt + self.positional_encoding_decoder(tgt)
            pos_embs_encoder = None  # self.positional_encoding(src)
            pos_embs_target = None
        elif (
            self.positional_encoding_type == "fixed_abs_sine"
            or self.attention_type == "hypermixing"
        ):
            tgt = tgt + self.positional_encoding(tgt)
            pos_embs_target = None
            pos_embs_encoder = None

        decoder_out, _, _ = self.decoder(
            tgt=tgt,
            memory=encoder_out,
            memory_mask=None,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_target,
            pos_embs_src=pos_embs_encoder,
        )

        return encoder_out, decoder_out


class ConformerEncoderWithExtraEmbeddings(ConformerEncoder):
    def __init__(self, injection_mode="cat", **kwargs):
        super().__init__(**kwargs)
        self.injection_mode = injection_mode
        if injection_mode == "cat":
            self.cat_proj = torch.nn.Linear(2 * kwargs["d_model"], kwargs["d_model"])

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        pos_embs=None,
        dynchunktrain_config=None,
        extra_embs=None,
    ):
        if self.attention_type == "RelPosMHAXL":
            if pos_embs is None:
                raise ValueError(
                    "The chosen attention type for the Conformer is RelPosMHAXL. "
                    "For this attention type, the positional embeddings are mandatory"
                )

        output = src
        output = self._inject_extra_embs(output, extra_embs)
        attention_lst = []
        for enc_layer in self.layers:
            output, attention = enc_layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos_embs=pos_embs,
                dynchunktrain_config=dynchunktrain_config,
            )
            output = self._inject_extra_embs(output, extra_embs)
            attention_lst.append(attention)
        output = self.norm(output)

        return output, attention_lst

    def _inject_extra_embs(self, src, extra_embs):
        if self.injection_mode == "prod":
            return src * extra_embs[:, None]
        elif self.injection_mode == "sum":
            return src + extra_embs[:, None]
        elif self.injection_mode == "cat":
            src = torch.cat(
                [src, extra_embs[:, None].expand(-1, src.shape[-2], -1)], dim=-1
            )
            return self.cat_proj(src)
        elif self.injection_mode is None:
            return src
        else:
            raise NotImplementedError


if __name__ == "__main__":
    B = 2
    T = 10
    L = 20
    H = 64
    C = 10

    model = EncoderDecoderWithExtraEmbeddings(
        input_size=H,
        tgt_vocab=C,
        d_model=256,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ffn=256,
        dropout=0.1,
        activation=torch.nn.GELU,
        max_length=2000,
        encoder_module="conformer",
        normalize_before=True,
        causal=False,
        injection_mode="cat",
    )
    src = torch.randn(B, T, H)
    tgt = torch.randint(0, C, size=(B, L))
    extra_embs = torch.randn(B, 256)
    enc_out, dec_out = model(src, tgt, extra_embs=extra_embs)
    print(enc_out.shape)
    print(dec_out.shape)
