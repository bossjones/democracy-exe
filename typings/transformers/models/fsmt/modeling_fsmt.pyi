"""
This type stub file was generated by pyright.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from ...modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from .configuration_fsmt import FSMTConfig

"""PyTorch Fairseq model, ported from https://github.com/pytorch/fairseq/tree/master/examples/wmt19"""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
FSMT_START_DOCSTRING = ...
FSMT_GENERATION_EXAMPLE = ...
FSMT_INPUTS_DOCSTRING = ...
def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    ...

def triu_onnx(x, diagonal=...):
    ...

class PretrainedFSMTModel(PreTrainedModel):
    config_class = FSMTConfig
    base_model_prefix = ...
    @property
    def dummy_inputs(self): # -> dict[str, Tensor]:
        ...



def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    ...

def make_padding_mask(input_ids, padding_idx=...): # -> None:
    """True for pad tokens"""
    ...

class EncoderLayer(nn.Module):
    def __init__(self, config: FSMTConfig) -> None:
        ...

    def forward(self, x, encoder_padding_mask, layer_head_mask, output_attentions=...): # -> tuple[Any, Any]:
        """
        Args:
            x (`torch.Tensor`): input to the layer of shape *(seq_len, batch, embed_dim)*
            encoder_padding_mask (`torch.ByteTensor`): binary ByteTensor of shape
                *(batch, src_len)* where padding elements are indicated by `1`.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                *(config.encoder_attention_heads,)*.

        Returns:
            encoded output of shape *(seq_len, batch, embed_dim)*
        """
        ...



class FSMTEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a [`EncoderLayer`].

    Args:
        config: FSMTConfig
    """
    def __init__(self, config: FSMTConfig, embed_tokens) -> None:
        ...

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., inputs_embeds: torch.Tensor = ..., head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, ...] | BaseModelOutput:
        """
        Args:
            input_ids (`torch.LongTensor`): tokens in the source language of shape
                *(batch, src_len)*
            attention_mask (`torch.LongTensor`): indicating which indices are padding tokens
            inputs_embeds (`torch.FloatTensor`):
                embedding vectors of shape *(batch, src_len, embed_dim)*
            head_mask (`torch.Tensor` of shape `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

        Returns:
            BaseModelOutput or Tuple comprised of:

                - **x** (`torch.Tensor`): the last encoder layer's output of shape *(src_len, batch, embed_dim)*
                - **encoder_states** (`Tuple(torch.FloatTensor`)): all intermediate hidden states of shape *(src_len,
                  batch, embed_dim)*. Only populated if *output_hidden_states:* is True.
                - **all_attentions** (`Tuple(torch.FloatTensor`)): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        ...



class DecoderLayer(nn.Module):
    def __init__(self, config: FSMTConfig) -> None:
        ...

    def forward(self, x, encoder_hidden_states, encoder_attn_mask=..., layer_state=..., causal_mask=..., layer_head_mask=..., cross_attn_layer_head_mask=..., decoder_padding_mask=..., output_attentions=...): # -> tuple[Any, Any, Any | dict[Any, Any], Any]:
        ...



class FSMTDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DecoderLayer`]

    Args:
        config: FSMTConfig
        embed_tokens (nn.Embedding): output embedding
    """
    def __init__(self, config: FSMTConfig, embed_tokens: nn.Embedding) -> None:
        ...

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor, encoder_padding_mask: torch.Tensor, decoder_padding_mask: torch.Tensor, decoder_causal_mask: torch.Tensor, head_mask: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., cross_attn_head_mask: Optional[torch.Tensor] = ..., past_key_values: Optional[List[torch.FloatTensor]] = ..., use_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, ...] | BaseModelOutputWithPastAndCrossAttentions:
        """
        Includes several features from "Jointly Learning to Align and Translate with Transformer Models" (Garg et al.,
        EMNLP 2019).

        Args:
            input_ids (`torch.LongTensor` of shape `(batch, tgt_len)`):
                previous decoder outputs for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            past_key_values (dict or None): dictionary used for storing state during generation
            head_mask (`torch.Tensor` of shape `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

        Returns:
            BaseModelOutputWithPast or tuple:

                - the decoder's features of shape *(batch, tgt_len, embed_dim)*
                - the cache
                - hidden states
                - attentions
        """
        ...



class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, embed_dim, num_heads, dropout=..., bias=..., encoder_decoder_attention=...) -> None:
        ...

    def forward(self, query, key: Optional[Tensor], key_padding_mask: Optional[Tensor] = ..., layer_state: Optional[Dict[str, Optional[Tensor]]] = ..., attn_mask: Optional[Tensor] = ..., layer_head_mask: Optional[Tensor] = ..., output_attentions=...) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        ...



def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    ...

@add_start_docstrings("The bare FSMT Model outputting raw hidden-states without any specific head on top.", FSMT_START_DOCSTRING)
class FSMTModel(PretrainedFSMTModel):
    _tied_weights_keys = ...
    def __init__(self, config: FSMTConfig) -> None:
        ...

    def get_encoder(self): # -> FSMTEncoder:
        ...

    def get_decoder(self): # -> FSMTDecoder:
        ...

    @add_start_docstrings_to_model_forward(FSMT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = ..., decoder_input_ids: Optional[torch.LongTensor] = ..., decoder_attention_mask: Optional[torch.BoolTensor] = ..., head_mask: Optional[torch.Tensor] = ..., decoder_head_mask: Optional[torch.Tensor] = ..., cross_attn_head_mask: Optional[torch.Tensor] = ..., encoder_outputs: Optional[Tuple[torch.FloatTensor]] = ..., past_key_values: Optional[Tuple[torch.FloatTensor]] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., decoder_inputs_embeds: Optional[torch.FloatTensor] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        ...

    def get_input_embeddings(self):
        ...

    def set_input_embeddings(self, value): # -> None:
        ...

    def get_output_embeddings(self): # -> Embedding:
        ...

    def set_output_embeddings(self, value): # -> None:
        ...



@add_start_docstrings("The FSMT Model with a language modeling head. Can be used for summarization.", FSMT_START_DOCSTRING)
class FSMTForConditionalGeneration(PretrainedFSMTModel):
    base_model_prefix = ...
    _tied_weights_keys = ...
    def __init__(self, config: FSMTConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(FSMT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(FSMT_GENERATION_EXAMPLE)
    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = ..., decoder_input_ids: Optional[torch.LongTensor] = ..., decoder_attention_mask: Optional[torch.BoolTensor] = ..., head_mask: Optional[torch.Tensor] = ..., decoder_head_mask: Optional[torch.Tensor] = ..., cross_attn_head_mask: Optional[torch.Tensor] = ..., encoder_outputs: Optional[Tuple[torch.FloatTensor]] = ..., past_key_values: Optional[Tuple[torch.FloatTensor]] = ..., inputs_embeds: Optional[torch.Tensor] = ..., decoder_inputs_embeds: Optional[torch.Tensor] = ..., labels: Optional[torch.LongTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        ...

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=..., attention_mask=..., head_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., use_cache=..., encoder_outputs=..., **kwargs): # -> dict[str, Any]:
        ...

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor): # -> Tensor:
        ...

    def get_encoder(self): # -> FSMTEncoder:
        ...

    def get_decoder(self): # -> FSMTDecoder:
        ...

    def get_output_embeddings(self): # -> Embedding:
        ...

    def set_output_embeddings(self, value): # -> None:
        ...



class SinusoidalPositionalEmbedding(nn.Embedding):
    """
    This module produces sinusoidal positional embeddings of any length.

    We don't want to save the weight of this embedding since it's not trained (deterministic) and it can be huge.

    Padding symbols are ignored.

    These embeddings get automatically extended in forward if more positions is needed.
    """
    def __init__(self, num_positions, embedding_dim, padding_idx) -> None:
        ...

    def make_weight(self, num_positions, embedding_dim, padding_idx): # -> None:
        ...

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx): # -> Tensor:
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        ...

    @staticmethod
    def make_positions(tensor, padding_idx: int):
        """
        Replace non-padding symbols with their position numbers.

        Position numbers begin at padding_idx+1. Padding symbols are ignored.
        """
        ...

    def forward(self, input, incremental_state: Optional[Any] = ..., timestep: Optional[Tensor] = ...): # -> Tensor:
        """Input is expected to be of size [bsz x seqlen]."""
        ...
