# ==================================================================================================================
# Copyright 2025 Luca Della Libera.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
# ==================================================================================================================

"""Llama 3 implementation (see https://arxiv.org/abs/2407.21783).

Supports jitting, KV caching and prompting.

"""

# Adapted from:
# https://github.com/meta-llama/llama3/blob/11817d47e1ba7a4959b025eb1ca308572e0e3963/llama/model.py

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


try:
    from multihead import MultiHeadEmbedding
except ImportError:
    from .multihead import MultiHeadEmbedding


__all__ = ["LlamaDecoder", "LlamaEncoder", "LlamaLayer"]


_LOGGER = logging.getLogger(__file__)


class RMSNorm(nn.Module):
    """Root-mean-square normalization layer.

    This layer normalizes the input tensor along its feature dimension by its
    root-mean-square and scales the result by a learned weight parameter.

    Parameters
    ----------
    dim:
        The feature dimension.
    norm_eps:
        A small constant added to the denominator for numerical stability.

    """

    def __init__(self, dim: "int" = 512, norm_eps: "float" = 1e-6) -> "None":
        super().__init__()
        self.dim = dim
        self.norm_eps = norm_eps

        # Parameters
        self.weight = nn.Parameter(torch.empty(dim))
        self.reset_parameters()

    def reset_parameters(self) -> "None":
        nn.init.ones_(self.weight)

    def forward(self, input: "Tensor") -> "Tensor":
        # input: [B, T, H]
        input_type = input.type()
        output = input.float()
        output = output * ((output**2).mean(-1, keepdim=True) + self.norm_eps).rsqrt()
        output = output.type(input_type)
        return self.weight * output

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__}(dim={self.dim}, norm_eps={self.norm_eps})"


class FeedForward(nn.Module):
    """Feed-forward layer.

    This layer consists of a two-layer fully connected network with a specified
    hidden dimension and dropout for regularization.

    Parameters
    ----------
    dim:
        The input and output dimension of the layer.
    ffn_dim:
        The hidden layer dimension in the feed-forward network.
    dropout:
        Dropout probability applied after the last hidden layer for regularization.

    """

    def __init__(
        self, dim: "int" = 512, ffn_dim: "int" = 2048, dropout: "float" = 0.0
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.dropout_ = dropout

        # Modules
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: "Tensor"):
        # input: [B, T, H]
        gate = self.act(self.w1(input))
        return self.dropout(self.w2(gate * self.w3(input)))


class GroupedQueryAttention(nn.Module):
    """Grouped-query attention layer.

    This layer performs a grouped multi-head attention mechanism, where the number
    of heads for queries can be different from the number of heads for keys and values.
    This approach reduces memory and computational requirements by sharing keys and
    values across multiple query heads.

    Parameters
    ----------
    dim:
        The feature dimension.
    n_heads:
        The number of attention heads for the queries.
    n_kv_heads:
        The number of attention heads for the keys and values, which
        is typically smaller than `n_heads` to save on computation.
    dropout:
        The dropout probability applied to the attention output for
        regularization.

    """

    def __init__(
        self,
        dim: "int" = 512,
        n_heads: "int" = 4,
        n_kv_heads: "int" = 1,
        dropout: "float" = 0.0,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.dropout = dropout
        self.head_dim = dim // n_heads
        self.n_kv_head_reps = n_heads // n_kv_heads

        # Modules
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        input: "Tensor",
        freqs_cis: "Tensor",
        mask: "Optional[Tensor]" = None,
        curr_pos: "int" = 0,
        kv_cache: "Optional[Tensor]" = None,
    ) -> "Tuple[Tensor, int, Tensor]":
        """Forward pass.

        This method applies rotary positional embeddings based on the provided `freqs_cis`,
        applies the grouped multi-head attention mechanism and handles key-value caching
        for efficient autoregressive decoding.

        Parameters
        ----------
        input:
            The input tensor of shape (batch_size, seq_length, dim), typically token embeddings.
        freqs_cis:
            Precomputed rotary positional embeddings for the current input sequence,
            corresponding to positions from `curr_pos` to `curr_pos + seq_length`.
        mask:
            The attention mask, shape (batch_size, ..., tgt_seq_length, src_seq_length). Two types of masks are supported:
            - a boolean mask where a value of True indicates that the element should take part in attention;
            - a float mask of the same type as query, key, value that is added to the attention score.
        curr_pos:
            The starting position of the current input sequence within the overall
            positional embedding space (default is 0).
        kv_cache:
            A tensor to cache key-value pairs for efficient autoregressive decoding.
            If provided, it should be of shape (batch_size, seq_length, n_kv_heads, head_dim, 2).

        Returns
        -------
            - The output tensor of shape (batch_size, seq_length, dim);
            - the updated position `curr_pos + seq_length` after processing the input;
            - the updated `kv_cache` with stored key-value pairs.

        """
        B = input.shape[0]
        T = input.shape[1]

        if kv_cache is None:
            assert curr_pos == 0
            kv_cache = torch.zeros(
                B,
                2 * T,  # Double the size
                self.n_kv_heads,
                self.head_dim,
                2,
                device=input.device,
                dtype=input.dtype,
            )
        elif curr_pos + T > kv_cache.shape[1]:
            # Expand along time dimension
            new_size = 2 * (curr_pos + T)  # Double the size
            kv_cache = F.pad(
                kv_cache, [0, 0, 0, 0, 0, 0, 0, new_size - kv_cache.shape[1]]
            )

        qs = self.wq(input).view(B, T, self.n_heads, -1)
        ks = self.wk(input).view(B, T, self.n_kv_heads, -1)
        vs = self.wv(input).view(B, T, self.n_kv_heads, -1)

        qs, ks = self._apply_rotary_emb(qs, ks, freqs_cis)

        kv_cache[:, curr_pos : curr_pos + T, :, :, 0] = ks
        kv_cache[:, curr_pos : curr_pos + T, :, :, 1] = vs

        ks = kv_cache[:, : curr_pos + T, :, :, 0]
        vs = kv_cache[:, : curr_pos + T, :, :, 1]

        ks = torch.repeat_interleave(
            ks, dim=-2, repeats=self.n_kv_head_reps
        )  # [B, curr_pos + T, n_heads, head_dim]
        vs = torch.repeat_interleave(
            vs, dim=-2, repeats=self.n_kv_head_reps
        )  # [B, curr_pos + T, n_heads, head_dim]

        # Reshape for scaled_dot_product_attention
        qs = qs.transpose(-3, -2)  # [B, n_heads, T, head_dim]
        ks = ks.transpose(-3, -2)  # [B, n_heads, curr_pos + T, head_dim]
        vs = vs.transpose(-3, -2)  # [B, n_heads, curr_pos + T, head_dim]

        output = F.scaled_dot_product_attention(
            qs,
            ks,
            vs,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # [B, n_heads, curr_pos + T, head_dim]

        output = (
            output.transpose(1, 2).contiguous().view(B, T, -1)
        )  # [B, curr_pos + T, n_heads * head_dim]
        output = self.wo(output)  # [B, curr_pos + T, dim]

        next_pos = curr_pos + T

        return output, next_pos, kv_cache

    @torch.jit.export
    def _apply_rotary_emb(
        self, xq: "Tensor", xk: "Tensor", freqs_cis: "Tensor"
    ) -> "Tuple[Tensor, Tensor]":
        xq_ = torch.view_as_complex(xq.float().reshape(xq.shape[:-1] + (-1, 2)))
        xk_ = torch.view_as_complex(xk.float().reshape(xk.shape[:-1] + (-1, 2)))

        # Reshape for broadcast
        assert xq_.ndim > 1
        assert freqs_cis.shape == (xq_.shape[1], xq_.shape[-1])
        shape = [1] * len(xq_.shape)
        shape[1] = xq_.shape[1]
        shape[-1] = xq_.shape[-1]
        freqs_cis = freqs_cis.view(shape)

        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(start_dim=3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(start_dim=3)
        return xq_out.type_as(xq), xk_out.type_as(xk)


class LlamaLayer(nn.Module):
    """Llama layer.

    This layer combines a grouped-query attention mechanism with a
    feed-forward network, both of which are normalized by RMSNorm layers.

    Parameters
    ----------
    dim:
        The feature dimension.
    ffn_dim:
        The hidden layer dimension in the feed-forward network.
    n_heads:
        The number of attention heads for the queries in the
        grouped-query attention mechanism.
    n_kv_heads:
        The number of heads for the keys and values in the grouped-query
        attention mechanism, typically fewer than `n_heads`.
    dropout:
        Dropout probability applied in both the attention and
        feed-forward networks.
    norm_eps:
        A small constant added to the RMS normalization denominator
        for numerical stability.

    """

    def __init__(
        self,
        dim: "int" = 512,
        ffn_dim: "int" = 2048,
        n_heads: "int" = 4,
        n_kv_heads: "int" = 1,
        dropout: "float" = 0.0,
        norm_eps: "float" = 1e-6,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.dropout = dropout
        self.norm_eps = norm_eps

        # Modules
        self.attention = GroupedQueryAttention(dim, n_heads, n_kv_heads, dropout)
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.feed_forward = FeedForward(dim, ffn_dim, dropout)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def forward(
        self,
        input: "Tensor",
        freqs_cis: "Tensor",
        mask: "Optional[Tensor]" = None,
        curr_pos: "int" = 0,
        kv_cache: "Optional[Tensor]" = None,
    ) -> "Tuple[Tensor, int, Tensor]":
        """See documentation of `GroupedQueryAttention.forward`."""
        hidden, curr_pos, kv_cache = self.attention(
            self.attention_norm(input),
            freqs_cis,
            mask,
            curr_pos,
            kv_cache=kv_cache,
        )
        hidden += input
        output = self.feed_forward(self.ffn_norm(hidden))
        output += hidden
        return output, curr_pos, kv_cache


class LlamaEncoder(nn.Module):
    """Llama encoder.

    This class implements a multi-layer encoder stack with grouped-query attention,
    feed-forward layers, and RMS normalization. It incorporates rotary positional
    embeddings and optionally includes an embedding/projection layer for the input,
    a projection layer for the output, and a projection layer for the prompt.

    Parameters
    ----------
    vocab_size:
        The size of the vocabulary, used for token embedding (optional).
    input_dim:
        The dimension of the input projection (optional).
    output_dim:
        The dimension of the output projection (optional).
    n_layers:
        The number of layers in the stack.
    dim:
        The dimension of the input features and hidden states.
    ffn_dim:
        The hidden layer dimension in the feed-forward networks.
        If None, defaults to 4 * `dim`.
    n_heads:
        The number of attention heads for the queries in each layer.
    n_kv_heads:
        The number of heads for the keys and values in the
        grouped-query attention mechanism.
    dropout:
        Dropout probability applied in each layer’s attention
        and feed-forward networks.
    norm_eps:
        A small constant added to the RMS normalization denominator
        for numerical stability.
    rope_theta:
        The scaling factor for the rotary positional embeddings,
        controlling the frequency range of positional encoding.
    max_seq_len:
        The maximum sequence length supported by the positional embeddings.
    prompt_dim:
        The dimension of the prompt embeddings (optional).
    num_codebooks:
        The number of codebooks (optional).
    embedding_kwargs:
        Additional keyword arguments for the token embedding layer.

    """

    def __init__(
        self,
        vocab_size: "Optional[int]" = None,
        input_dim: "Optional[int]" = None,
        output_dim: "Optional[int]" = None,
        n_layers: "int" = 8,
        dim: "int" = 512,
        ffn_dim: "Optional[int]" = None,
        n_heads: "int" = 4,
        n_kv_heads: "int" = 1,
        dropout: "float" = 0.0,
        norm_eps: "float" = 1e-6,
        rope_theta: "float" = 10000.0,
        max_seq_len: "int" = 1024,
        prompt_dim: "Optional[int]" = None,
        num_codebooks: "int" = 1,
        embedding_kwargs: "Optional[Dict[str, Any]]" = None,
        **kwargs: "Any",
    ) -> "None":
        super().__init__()
        if kwargs:
            _LOGGER.warning(f"Unused initialization arguments: {kwargs}")
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dim = dim
        self.ffn_dim = ffn_dim = 4 * dim if ffn_dim is None else ffn_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.dropout = dropout
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.prompt_dim = prompt_dim
        self.num_codebooks = num_codebooks
        self.embedding_kwargs = embedding_kwargs or {}

        # Modules
        if vocab_size is not None:
            self.tok_embeddings = MultiHeadEmbedding(
                vocab_size,
                dim if input_dim is None else input_dim,
                num_codebooks,
                **self.embedding_kwargs,
            )
        if input_dim is not None:
            self.input = nn.Linear(input_dim, dim, bias=False)
        self.layers = nn.ModuleList(
            LlamaLayer(
                dim=dim,
                ffn_dim=ffn_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                dropout=dropout,
                norm_eps=norm_eps,
            )
            for _ in range(self.n_layers)
        )
        self.norm = RMSNorm(dim, norm_eps)
        if output_dim is not None:
            if num_codebooks > 1:
                self.output = nn.ModuleList(
                    nn.Linear(dim, output_dim, bias=False) for _ in range(num_codebooks)
                )
            else:
                self.output = nn.Linear(dim, output_dim, bias=False)
        if prompt_dim is not None:
            self.prompt = nn.Linear(prompt_dim, dim, bias=False)

        # Non-persistent buffers
        self.register_buffer(
            "freqs_cis",
            self._precompute_freqs_cis(
                dim // n_heads,
                rope_theta,
                max_seq_len * 2,
            ),
            persistent=False,
        )

    @torch.jit.export
    def embed(
        self,
        toks: "Tensor" = None,
        prompt_embs: "Optional[Tensor]" = None,
        curr_pos: "int" = 0,
    ) -> "Tensor":
        """Embed input tokens and optional prompt embeddings.

        This method generates the initial embeddings for the input sequence by embedding the
        input tokens, optionally prepending the provided prompt embeddings. The resulting tensor
        is ready to be passed through the model's layers.

        Parameters
        ----------
        toks:
            A tensor containing the input tokens. The tensor should have shape (batch_size, seq_length).
        prompt_embs:
            A tensor containing the prompt embeddings to prepend to the input tokens. The tensor should
            have shape (batch_size, prompt_length, prompt_dim), where `prompt_length` is the length of the prompt
            and `dim` is the embedding dimension. If `None`, no prompt embeddings are prepended.
        curr_pos:
            The starting position of the current input sequence within the overall
            positional embedding space (default is 0).

        Returns
        -------
            A tensor of shape (batch_size, prompt_length + seq_length, dim) containing the
            optional prompt embeddings concatenated with the input token embeddings.

        """
        if self.vocab_size is None:
            raise NotImplementedError
        if self.num_codebooks > 1:
            shift = curr_pos % self.num_codebooks
            if shift > 0:
                toks = nn.functional.pad(toks, (shift, 0))
            orig_length = toks.shape[-1]
            rem = orig_length % self.num_codebooks
            if rem != 0:
                toks = nn.functional.pad(toks, (0, self.num_codebooks - rem))
            toks = toks.reshape(toks.shape[0], -1, self.num_codebooks)  # [B, T', K]
            embs = self.tok_embeddings(toks)  # [B, T', K, E]
            embs = embs.flatten(start_dim=1, end_dim=2)  # [B, T'K, E]
            embs = embs[:, :orig_length]
            if shift > 0:
                embs = embs[:, shift:]
        else:
            embs = self.tok_embeddings(toks[..., None])[..., 0, :]
        if self.input_dim is not None:
            embs = self.input(embs)
        if prompt_embs is not None:
            assert embs.shape[0] == prompt_embs.shape[0]
            if self.prompt_dim is not None:
                if prompt_embs.shape[-1] == self.prompt_dim:
                    prompt_embs = self.prompt(prompt_embs)
            # [B, M, H] + [B, T, H] = [B, M + T, H]
            embs = torch.cat([prompt_embs, embs], dim=-2)
        return embs

    def forward(
        self,
        input: "Tensor",
        mask: "Optional[Tensor]" = None,
        state: "Optional[Tuple[int, List[Tensor]]]" = None,
    ) -> "Tuple[Tensor, Tuple[int, List[Tensor]]]":
        """Forward pass.

        Parameters
        ----------
        input:
            The input tensor of shape (batch_size, seq_length, dim), representing the input embeddings.
        mask:
            The attention mask, shape (batch_size, ..., tgt_seq_length, src_seq_length). Two types of masks are supported:
            - a boolean mask where a value of True indicates that the element should take part in attention;
            - a float mask of the same type as query, key, value that is added to the attention score.
        state:
            The state containing the current position (`curr_pos`) and a list of key-value
            caches for each layer. The `state` is used for autoregressive decoding where
            the cached key-value pairs are updated and used to avoid recomputing them
            during each forward pass.

        Returns
        -------
            - The output tensor of shape (batch_size, seq_length, dim);
            - the updated state, including:
              - `next_pos`: the updated position after processing the input sequence;
              - `kv_caches`: the updated key-value cache for each layer.

        """
        T = input.shape[1]
        device = input.device
        next_pos = -1

        curr_pos = 0
        kv_caches = [torch.empty(0)] * self.n_layers  # JIT compilable
        if state is not None:
            curr_pos, kv_caches = state

        output = input
        self.freqs_cis = self.freqs_cis.to(device)
        freqs_cis = self.freqs_cis[curr_pos : curr_pos + T]
        for i, layer in enumerate(self.layers):
            output, next_pos, kv_caches[i] = layer(
                output,
                freqs_cis,
                mask,
                curr_pos,
                None if state is None else kv_caches[i],  # JIT compilable
            )
        output = self.norm(output)
        if self.output_dim is not None:
            if self.num_codebooks > 1:
                shift = curr_pos % self.num_codebooks
                if shift > 0:
                    output = nn.functional.pad(output, (0, 0, shift, 0))
                orig_length = output.shape[-2]
                rem = orig_length % self.num_codebooks
                if rem != 0:
                    output = nn.functional.pad(
                        output, (0, 0, 0, self.num_codebooks - rem)
                    )
                output = output.reshape(
                    output.shape[0], -1, self.num_codebooks, output.shape[-1]
                )  # [B, T', K, H]
                outputs = []
                assert isinstance(self.output, nn.ModuleList)
                for k, output_k in enumerate(self.output):
                    outputs.append(output_k(output[:, :, k])[:, :, None])
                output = torch.cat(outputs, dim=2)  # [B, T', K, C]
                output = output.flatten(start_dim=1, end_dim=2)  # [B, T'K, C]
                output = output[:, :orig_length]  # [B, T, C]
                if shift > 0:
                    output = output[:, shift:]
            else:
                output = self.output(output)
        state = (next_pos, kv_caches)
        return output, state

    @torch.jit.export
    def _precompute_freqs_cis(
        self,
        dim: "int" = 128,
        rope_theta: "float" = 10000.0,
        max_seq_len: "int" = 1024,
        device: "torch.device" = "cpu",
    ) -> "Tensor":
        freqs = 1.0 / (
            rope_theta
            ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
        )
        t = torch.arange(max_seq_len, device=freqs.device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis


class LlamaDecoder(LlamaEncoder):
    """Llama decoder.

    This class implements a multi-layer decoder stack with grouped-query attention,
    feed-forward layers, and RMS normalization. It incorporates rotary positional
    embeddings and optionally includes an embedding/projection layer for the input
    and a projection layer for the output.

    Parameters
    ----------
    vocab_size:
        The size of the vocabulary, used for token embedding (optional).
    n_layers:
        The number of layers in the stack.
    dim:
        The dimension of the input features and hidden states.
    ffn_dim:
        The hidden layer dimension in the feed-forward networks.
        If None, defaults to 4 * `dim`.
    n_heads:
        The number of attention heads for the queries in each layer.
    n_kv_heads:
        The number of heads for the keys and values in the
        grouped-query attention mechanism.
    dropout:
        Dropout probability applied in each layer’s attention
        and feed-forward networks.
    norm_eps:
        A small constant added to the RMS normalization denominator
        for numerical stability.
    rope_theta:
        The scaling factor for the rotary positional embeddings,
        controlling the frequency range of positional encoding.
    max_seq_len:
        The maximum sequence length supported by the positional embeddings.
    prompt_dim:
        The dimension of the prompt embeddings (optional).
    num_codebooks:
        The number of codebooks (optional).
    embedding_kwargs:
        Additional keyword arguments for the token embedding layer.

    """

    def __init__(
        self,
        vocab_size: "int",
        n_layers: "int" = 8,
        dim: "int" = 512,
        ffn_dim: "Optional[int]" = None,
        n_heads: "int" = 4,
        n_kv_heads: "int" = 1,
        dropout: "float" = 0.0,
        norm_eps: "float" = 1e-6,
        rope_theta: "float" = 10000.0,
        max_seq_len: "int" = 1024,
        prompt_dim: "Optional[int]" = None,
        num_codebooks: "int" = 1,
        embedding_kwargs: "Optional[Dict[str, Any]]" = None,
        **kwargs: "Any",
    ) -> "None":
        super().__init__(
            vocab_size=vocab_size,
            output_dim=vocab_size,
            n_layers=n_layers,
            dim=dim,
            ffn_dim=ffn_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            norm_eps=norm_eps,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            prompt_dim=prompt_dim,
            num_codebooks=num_codebooks,
            embedding_kwargs=embedding_kwargs,
            **kwargs,
        )

    def forward(
        self,
        input: "Tensor",
        mask: "Optional[Union[str, Tensor]]" = "causal",
        state: "Optional[Tuple[int, List[Tensor]]]" = None,
    ) -> "Tuple[Tensor, Tuple[int, List[Tensor]]]":
        """Forward pass.

        Parameters
        ----------
        input:
            The input tensor of shape (batch_size, seq_length, dim), representing the input embeddings.
        mask:
            The attention mask, shape (batch_size, ..., tgt_seq_length, src_seq_length). Two types of masks are supported:
            - a boolean mask where a value of True indicates that the element should take part in attention;
            - a float mask of the same type as query, key, value that is added to the attention score.
            If "causal", applies a causal mask to prevent attending to future tokens.
        state:
            The state containing the current position (`curr_pos`) and a list of key-value
            caches for each layer. The `state` is used for autoregressive decoding where
            the cached key-value pairs are updated and used to avoid recomputing them
            during each forward pass.

        Returns
        -------
            - The output tensor of shape (batch_size, seq_length, dim);
            - the updated state, including:
              - `next_pos`: the updated position after processing the input sequence;
              - `kv_caches`: the updated key-value cache for each layer.

        """
        # NOTE: here we cannot call super().forward because JIT does not support inheritance
        T = input.shape[1]
        device = input.device
        next_pos = -1

        curr_pos = 0
        kv_caches = [torch.empty(0)] * self.n_layers  # JIT compilable
        if state is not None:
            curr_pos, kv_caches = state

        if isinstance(mask, str):
            if mask != "causal":
                raise NotImplementedError
            if T > 1:
                # Instead of overwriting `mask`, define a new variable `mask_` to make it JIT compilable
                mask_ = torch.full((T, T), float("-inf"), device=device)
                mask_ = torch.triu(mask_, diagonal=1)

                # When performing key-value caching, we compute the attention scores
                # only for the new sequence. Thus, the matrix of scores is of size
                # (T, curr_pos + T), and the only masked entries are (i, j) for
                # j > curr_pos + i, since row i corresponds to token curr_pos + i
                if curr_pos > 0:
                    mask_ = torch.hstack(
                        [torch.zeros((T, curr_pos), device=device), mask_]
                    ).type_as(input)
            else:
                mask_ = None
        else:
            mask_ = mask

        output = input
        self.freqs_cis = self.freqs_cis.to(device)
        freqs_cis = self.freqs_cis[curr_pos : curr_pos + T]
        for i, layer in enumerate(self.layers):
            output, next_pos, kv_caches[i] = layer(
                output,
                freqs_cis,
                mask_,
                curr_pos,
                None if state is None else kv_caches[i],  # JIT compilable
            )
        output = self.norm(output)
        if self.output_dim is not None:
            if self.num_codebooks > 1:
                shift = curr_pos % self.num_codebooks
                if shift > 0:
                    output = nn.functional.pad(output, (0, 0, shift, 0))
                orig_length = output.shape[-2]
                rem = orig_length % self.num_codebooks
                if rem != 0:
                    output = nn.functional.pad(
                        output, (0, 0, 0, self.num_codebooks - rem)
                    )
                output = output.reshape(
                    output.shape[0], -1, self.num_codebooks, output.shape[-1]
                )  # [B, T', K, H]
                outputs = []
                assert isinstance(self.output, nn.ModuleList)
                for k, output_k in enumerate(self.output):
                    outputs.append(output_k(output[:, :, k])[:, :, None])
                output = torch.cat(outputs, dim=2)  # [B, T', K, C]
                output = output.flatten(start_dim=1, end_dim=2)  # [B, T'K, C]
                output = output[:, :orig_length]  # [B, T, C]
                if shift > 0:
                    output = output[:, shift:]
            else:
                output = self.output(output)
        state = (next_pos, kv_caches)
        return output, state

    @torch.jit.ignore
    def generate(
        self,
        bos_toks: "Tensor",
        eos_id: "int",
        prompt_embs: "Optional[Tensor]" = None,
        max_gen_toks: "int" = 100,
        eos_threshold: "float" = float("inf"),
        top_p: "float" = 0.9,
        temp: "float" = 1.0,
        use_kv_cache: "bool" = True,
    ) -> "List[Tensor]":
        """Autoregressively generate a sequence of tokens starting from the given prefix tokens.

        This method generates tokens from the given prefix tokens and continues until an EOS token
        is generated or the maximum number of tokens (max_gen_toks) is reached. The generation
        can use different sampling strategies, including greedy search (for top_p=0.0), and
        top-p sampling with temperature scaling for more diverse outputs.

        Parameters
        ----------
        bos_toks:
            A tensor containing the prefix tokens (the first token should be BOS). The tensor should
            have shape (batch_size, seq_length).
        eos_id:
            The token ID representing the EOS token. The generation process will stop when this token
            is generated.
        prompt_embs:
            A tensor of shape (batch_size, prompt_length, prompt_dim) containing optional prompt embeddings
            to prepend to the BOS tokens. If provided, these embeddings will be used as context to
            guide the generation. If None, no prompt embeddings are included.
        max_gen_toks:
            The maximum number of tokens to generate. Generation stops either when an EOS token
            is produced or when this number of tokens is generated.
        eos_threshold:
            A threshold that limits the probability of generating an EOS token at each step. When
            `eos_threshold` is finite, generation will avoid producing EOS tokens until their
            probability reaches the specified threshold relative to the maximum log probability.
        top_p:
            The cumulative probability threshold for nucleus sampling (top-p sampling). This parameter
            controls the diversity of the generated text by limiting the token selection to a subset
            that covers the top-p probability mass. This is useful for balancing diversity and coherence.
            A top-p value of 0.0 results in greedy search (selecting the most probable token at each step).
        temp:
            The temperature parameter used to control randomness in the top-p sampling process.
            Values higher than 1.0 increase the randomness by flattening the probability distribution,
            values lower than 1.0 decrease the randomness by sharpening the probability distribution.
        use_kv_cache:
            True to speed up generation via KV caching, False otherwise.

        Returns
        -------
            A list of tensors, where each tensor is of shape (batch_size, seq_length) containing
            the generated sequence of tokens. The generation process will stop once the EOS
            token is generated or the maximum number of tokens (max_gen_toks) is reached.

        """
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()

        bos_toks = bos_toks.to(device)
        if prompt_embs is not None:
            if self.num_codebooks > 1:
                assert prompt_embs.shape[1] % self.num_codebooks == 0
            prompt_embs = prompt_embs.to(device)

        with torch.no_grad():
            hyp_toks = self._greedy_search(
                bos_toks,
                eos_id,
                prompt_embs,
                max_gen_toks,
                eos_threshold,
                top_p,
                temp,
                use_kv_cache,
            )

        if was_training:
            self.train()
        else:
            self.eval()

        return hyp_toks

    @torch.jit.ignore
    def _greedy_search(
        self,
        bos_toks: "Tensor",
        eos_id: "int",
        prompt_embs: "Optional[Tensor]" = None,
        max_gen_toks: "int" = 100,
        eos_threshold: "float" = float("inf"),
        top_p: "float" = 0.9,
        temp: "float" = 1.0,
        use_kv_cache: "bool" = True,
    ) -> "List[Tensor]":
        batch_size = bos_toks.shape[0]
        device = bos_toks.device

        embs = self.embed(bos_toks, prompt_embs)
        hyp_toks = torch.full(
            (batch_size, max_gen_toks),
            eos_id,
            device=device,
        )
        lens = torch.zeros(batch_size, device=device)
        alive_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Autoregressive loop
        state: "Optional[Tuple[int, List[Tensor]]]" = None
        num_gen_toks = 0
        while num_gen_toks < max_gen_toks:
            if not use_kv_cache:
                state = None
            logits, state = self.forward(embs, mask="causal", state=state)
            logits = logits[:, -1]
            log_probs = (logits / temp).log_softmax(dim=-1)

            if eos_threshold < float("inf"):
                max_log_probs, _ = log_probs.max(dim=-1)
                eos_log_probs = log_probs[:, eos_id]
                eos_mask = eos_log_probs <= (eos_threshold * max_log_probs)
                log_probs[:, eos_id][eos_mask] = -1e20

            if top_p != 0.0:
                probs = log_probs.exp()
                next_tok = self._sample_top_p(probs, top_p)
            else:
                next_tok = log_probs.argmax(dim=-1)

            hyp_toks[:, num_gen_toks] = next_tok
            eos_mask = next_tok == eos_id
            alive_mask = alive_mask & (~eos_mask)
            lens[alive_mask] += 1
            num_gen_toks += 1
            if not alive_mask.any():
                break
            embs = self.embed(
                next_tok[:, None],
                prompt_embs=None if use_kv_cache else embs,
                curr_pos=num_gen_toks,
            )

        num_gen_toks = max(num_gen_toks, lens.max().item())
        hyp_toks = hyp_toks[:, :num_gen_toks]
        hyp_toks = [hyp_toks[i, : lens[i].long()] for i in range(batch_size)]

        return hyp_toks

    @torch.jit.export
    def _sample_top_p(self, probs: "Tensor", p: "float") -> "Tensor":
        # [B, C]
        probs, idx = probs.sort(dim=-1, descending=True)
        probs_sum = probs.cumsum(dim=-1)
        mask = probs_sum - probs > p
        probs[mask] = 0.0
        probs = probs / probs.sum(dim=-1, keepdim=True)
        next_tok = torch.multinomial(probs, num_samples=1)
        next_tok = idx.gather(-1, next_tok)
        # [B]
        return next_tok[:, 0]


def test_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, H, K, I, O = 3, 512, 30, 128, 12
    model = LlamaEncoder(vocab_size=K, input_dim=I, output_dim=O).to(device)
    print(model)

    # Process 50 timesteps
    input = torch.randint(0, K, size=(B, 50), device=device)
    output, state = model(model.embed(input))
    model = torch.jit.script(model)
    output_jit, state_jit = model(model.embed(input))
    assert torch.allclose(output, output_jit, atol=1e-6), (
        output.sum(),
        output_jit.sum(),
    )
    assert state[0] == state_jit[0]
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(state[1], state_jit[1]))
    print(output.shape)
    print(state[0])
    output.sum().backward()
    for k, v in model.named_parameters():
        if "embedding" not in k:
            assert v.grad is not None

    # Without embedding layer
    model = LlamaEncoder(input_dim=I, output_dim=O).to(device)
    input = torch.randn(B, 50, H, device=device)
    output, _ = model(input)
    assert output.shape[-1] == O

    # Without input/output projection
    input = torch.randint(0, K, size=(B, 50), device=device)
    model = LlamaEncoder(vocab_size=K).to(device)
    output, _ = model(model.embed(input))
    assert output.shape[-1] == H


def test_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, H, K = 3, 512, 30
    model = LlamaDecoder(K).to(device)
    print(model)

    # Process 50 timesteps
    input = torch.randn(B, 50, H, device=device)
    output, state = model(input)
    output_jit, state_jit = torch.jit.script(model)(input)
    assert torch.allclose(output, output_jit, atol=1e-6), (
        output.sum(),
        output_jit.sum(),
    )
    assert state[0] == state_jit[0]
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(state[1], state_jit[1]))
    print(output.shape)
    print(state[0])
    output.sum().backward()
    for k, v in model.named_parameters():
        if "embedding" not in k:
            assert v.grad is not None

    # Process 2 additional timesteps
    input = torch.randn(B, 2, H, device=device)
    output, state_ = model(input, state=state)
    output_jit, state_jit = torch.jit.script(model)(input, state=state)
    state = state_
    assert torch.allclose(output, output_jit, atol=1e-6), (
        output.sum(),
        output_jit.sum(),
    )
    assert state[0] == state_jit[0]
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(state[1], state_jit[1]))
    print(output.shape)
    print(state[0])

    # Reset and process 2 timesteps
    input = torch.randn(B, 2, H, device=device)
    output, state = model(input)
    output_jit, state_jit = torch.jit.script(model)(input)
    assert torch.allclose(output, output_jit, atol=1e-6), (
        output.sum(),
        output_jit.sum(),
    )
    assert state[0] == state_jit[0]
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(state[1], state_jit[1]))
    print(output.shape)
    print(state[0])

    # Non-causal mask
    input = torch.randn(B, 2, H, device=device)
    output, state = model(input, mask=None)
    output_jit, state_jit = torch.jit.script(model)(input, mask=None)
    assert torch.allclose(output, output_jit, atol=1e-6), (
        output.sum(),
        output_jit.sum(),
    )
    assert state[0] == state_jit[0]
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(state[1], state_jit[1]))
    print(output.shape)
    print(state[0])


def test_generation():
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, H, K, P = 3, 512, 30, 80
    model = LlamaDecoder(K, prompt_dim=P).to(device)
    print(model)

    bos_id = 0
    eos_id = 1

    bos_toks = torch.full((B, 1), bos_id, device=device)
    prompt_embs = torch.ones(B, 10, P, device=device)

    print("-----------")
    print("KV caching")
    output = model.generate(
        bos_toks,
        eos_id,
        prompt_embs=prompt_embs,
        top_p=0.0,
        use_kv_cache=False,
    )
    output_kv_cache = model.generate(
        bos_toks,
        eos_id,
        prompt_embs=prompt_embs,
        top_p=0.0,
        use_kv_cache=True,
    )
    assert all(
        torch.allclose(x, y, atol=1e-6) for x, y in zip(output, output_kv_cache)
    ), (output[0], output_kv_cache[0])
    print("Test passed")
    print("-----------")

    print("Jitting")
    output = model.generate(
        bos_toks,
        eos_id,
        prompt_embs=prompt_embs,
        top_p=0.0,
        use_kv_cache=False,
    )
    output_jit = torch.jit.script(model).generate(
        bos_toks,
        eos_id,
        prompt_embs=prompt_embs,
        top_p=0.0,
        use_kv_cache=False,
    )
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(output, output_jit)), (
        output[0],
        output_jit[0],
    )
    output = model.generate(
        bos_toks,
        eos_id,
        prompt_embs=prompt_embs,
        top_p=0.0,
        use_kv_cache=True,
    )
    output_jit = torch.jit.script(model).generate(
        bos_toks,
        eos_id,
        prompt_embs=prompt_embs,
        top_p=0.0,
        use_kv_cache=True,
    )
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(output, output_jit)), (
        output[0],
        output_jit[0],
    )
    model.generate(bos_toks, eos_id, prompt_embs=prompt_embs)
    print("Test passed")
    print("-----------")

    print("Benchmarking")
    print("-----------")
    print("Without JIT")
    for i in range(4):
        print(f"Iteration {i}")
        ts = time.time()
        for i in range(2):
            # Generation
            model.generate(
                bos_toks,
                eos_id,
                prompt_embs=prompt_embs,
                top_p=0.0,
                use_kv_cache=False,
            )
            model.generate(
                bos_toks,
                eos_id,
                prompt_embs=prompt_embs,
                top_p=0.0,
                use_kv_cache=True,
            )
            model.generate(bos_toks, eos_id, prompt_embs=prompt_embs)
        torch.cuda.synchronize()
        print(time.time() - ts)
    print("-----------")
    print("With JIT")
    model_jit = torch.jit.script(model)
    for i in range(4):
        print(f"Iteration {i}")
        ts = time.time()
        for i in range(2):
            # Generation
            model_jit.generate(
                bos_toks,
                eos_id,
                prompt_embs=prompt_embs,
                top_p=0.0,
                use_kv_cache=False,
            )
            model_jit.generate(
                bos_toks,
                eos_id,
                prompt_embs=prompt_embs,
                top_p=0.0,
                use_kv_cache=True,
            )
            model_jit.generate(bos_toks, eos_id, prompt_embs=prompt_embs)
        torch.cuda.synchronize()
        print(time.time() - ts)
    print("-----------")


if __name__ == "__main__":
    test_encoder()
    test_decoder()
    test_generation()
