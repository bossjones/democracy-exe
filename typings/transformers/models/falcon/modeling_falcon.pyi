"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
from torch import nn

from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
)
from .configuration_falcon import FalconConfig

"""PyTorch Falcon model."""
if TYPE_CHECKING:
    ...
if is_flash_attn_2_available():
    ...
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
class FalconLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ...



def rotate_half(x): # -> Tensor:
    """Rotates half the hidden dims of the input."""
    ...

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=...): # -> tuple[Any, Any]:
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    ...

class FalconRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=..., base=..., device=...) -> None:
        ...

    def forward(self, x, seq_len=...): # -> tuple[Any, Any]:
        ...



class FalconLinearScalingRotaryEmbedding(FalconRotaryEmbedding):
    """FalconRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""
    def __init__(self, dim, max_position_embeddings=..., base=..., device=..., scaling_factor=...) -> None:
        ...



class FalconDynamicNTKScalingRotaryEmbedding(FalconRotaryEmbedding):
    """FalconRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""
    def __init__(self, dim, max_position_embeddings=..., base=..., device=..., scaling_factor=...) -> None:
        ...



def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    ...

def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *required*):
            residual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    ...

class FalconAttention(nn.Module):
    def __init__(self, config: FalconConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, alibi: Optional[torch.Tensor], attention_mask: torch.Tensor, position_ids: Optional[torch.LongTensor] = ..., layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = ..., head_mask: Optional[torch.Tensor] = ..., use_cache: bool = ..., output_attentions: bool = ..., **kwargs): # -> tuple[Any, tuple[Tensor | Any, Tensor] | None, Tensor | None] | tuple[Any, tuple[Tensor | Any, Tensor] | None] | tuple[Any, tuple[Tensor | Any, Tensor] | None, Any]:
        ...



class FalconFlashAttention2(FalconAttention):
    """
    Falcon flash attention module. This module inherits from `FalconAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """
    def __init__(self, *args, **kwargs) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, alibi: Optional[torch.Tensor], attention_mask: torch.Tensor, position_ids: Optional[torch.LongTensor] = ..., layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = ..., head_mask: Optional[torch.Tensor] = ..., use_cache: bool = ..., output_attentions: bool = ..., **kwargs): # -> tuple[Any, tuple[Tensor | Any, Tensor] | None, Any | None]:
        ...



class FalconMLP(nn.Module):
    def __init__(self, config: FalconConfig) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...



FALCON_ATTENTION_CLASSES = ...
class FalconDecoderLayer(nn.Module):
    def __init__(self, config: FalconConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, alibi: Optional[torch.Tensor], attention_mask: torch.Tensor, position_ids: Optional[torch.LongTensor] = ..., layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = ..., head_mask: Optional[torch.Tensor] = ..., use_cache: bool = ..., output_attentions: bool = ..., **kwargs):
        ...



FALCON_START_DOCSTRING = ...
FALCON_INPUTS_DOCSTRING = ...
class FalconPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = FalconConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn_2 = ...
    _supports_sdpa = ...
    def __init__(self, *inputs, **kwargs) -> None:
        ...



@add_start_docstrings("The bare Falcon Model transformer outputting raw hidden-states without any specific head on top.", FALCON_START_DOCSTRING)
class FalconModel(FalconPreTrainedModel):
    def __init__(self, config: FalconConfig) -> None:
        ...

    def get_input_embeddings(self): # -> Embedding | Tensor:
        ...

    def set_input_embeddings(self, new_embeddings: torch.Tensor): # -> None:
        ...

    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPastAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.LongTensor] = ..., inputs_embeds: Optional[torch.LongTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        ...



@add_start_docstrings("The Falcon Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).", FALCON_START_DOCSTRING)
class FalconForCausalLM(FalconPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: FalconConfig) -> None:
        ...

    def get_output_embeddings(self): # -> Linear | Tensor:
        ...

    def set_output_embeddings(self, new_embeddings: torch.Tensor): # -> None:
        ...

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, past_key_values: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., **kwargs) -> dict:
        ...

    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., labels: Optional[torch.Tensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        ...



@add_start_docstrings("""
    The Falcon Model transformer with a sequence classification head on top (linear layer).

    [`FalconForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """, FALCON_START_DOCSTRING)
class FalconForSequenceClassification(FalconPreTrainedModel):
    def __init__(self, config: FalconConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = ..., attention_mask: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., labels: Optional[torch.Tensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...



@add_start_docstrings("""
    Falcon Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, FALCON_START_DOCSTRING)
class FalconForTokenClassification(FalconPreTrainedModel):
    def __init__(self, config: FalconConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = ..., attention_mask: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., labels: Optional[torch.Tensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...



@add_start_docstrings("""
    The Falcon Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, FALCON_START_DOCSTRING)
class FalconForQuestionAnswering(FalconPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., start_positions: Optional[torch.LongTensor] = ..., end_positions: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        ...
