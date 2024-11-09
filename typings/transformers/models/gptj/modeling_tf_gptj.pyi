"""
This type stub file was generated by pyright.
"""

from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPast,
    TFCausalLMOutputWithPast,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutputWithPast,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from .configuration_gptj import GPTJConfig

""" TF 2.0 GPT-J model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
def create_sinusoidal_positions(num_pos: int, dim: int) -> tf.Tensor:
    ...

def rotate_every_two(x: tf.Tensor) -> tf.Tensor:
    ...

def apply_rotary_pos_emb(tensor: tf.Tensor, sincos: tf.Tensor) -> tf.Tensor:
    ...

class TFGPTJAttention(keras.layers.Layer):
    def __init__(self, config: GPTJConfig, **kwargs) -> None:
        ...

    def get_causal_mask(self, key_length, query_length) -> tf.Tensor:
        ...

    @staticmethod
    def get_masked_bias(dtype: tf.DType) -> tf.Tensor:
        ...

    def call(self, hidden_states: tf.Tensor, layer_past: Optional[Tuple[tf.Tensor, tf.Tensor]] = ..., attention_mask: tf.Tensor | None = ..., position_ids: tf.Tensor | None = ..., head_mask: tf.Tensor | None = ..., use_cache: bool = ..., output_attentions: bool = ...): # -> tuple[Any, tuple[Any, Any] | None, Any] | tuple[Any, tuple[Any, Any] | None]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFGPTJMLP(keras.layers.Layer):
    def __init__(self, intermediate_size: int, config: GPTJConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFGPTJBlock(keras.layers.Layer):
    def __init__(self, config: GPTJConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor, layer_past: tf.Tensor | None = ..., attention_mask: tf.Tensor | None = ..., position_ids: tf.Tensor | None = ..., head_mask: tf.Tensor | None = ..., use_cache: bool = ..., output_attentions: bool = ...):
        ...

    def build(self, input_shape=...): # -> None:
        ...



@keras_serializable
class TFGPTJMainLayer(keras.layers.Layer):
    config_class = GPTJConfig
    def __init__(self, config: GPTJConfig, *inputs, **kwargs) -> None:
        ...

    def get_input_embeddings(self): # -> TFSharedEmbeddings:
        ...

    def set_input_embeddings(self, value: tf.Tensor): # -> None:
        ...

    @unpack_inputs
    def call(self, input_ids=..., past_key_values=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=...) -> Union[TFBaseModelOutputWithPast, Tuple[tf.Tensor]]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFGPTJPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPTJConfig
    base_model_prefix = ...
    _keys_to_ignore_on_load_unexpected = ...


GPTJ_START_DOCSTRING = ...
GPTJ_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare GPT-J Model transformer outputting raw hidden-states without any specific head on top.", GPTJ_START_DOCSTRING)
class TFGPTJModel(TFGPTJPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[TFBaseModelOutputWithPast, Tuple[tf.Tensor]]:
        r"""
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past`). Set to `False` during training, `True` during generation
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    The GPT-J Model transformer with a language modeling head on top.
    """, GPTJ_START_DOCSTRING)
class TFGPTJForCausalLM(TFGPTJPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    def get_output_embeddings(self):
        ...

    def set_output_embeddings(self, new_embeddings): # -> None:
        ...

    def prepare_inputs_for_generation(self, inputs, past_key_values=..., use_cache=..., **kwargs): # -> dict[str, Any]:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., labels: np.ndarray | tf.Tensor | None = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[TFCausalLMOutputWithPast, Tuple[tf.Tensor]]:
        r"""
        labels (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    The GPT-J Model transformer with a sequence classification head on top (linear layer).

    [`GPTJForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT, GPT-2, GPT-Neo) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """, GPTJ_START_DOCSTRING)
class TFGPTJForSequenceClassification(TFGPTJPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., labels: np.ndarray | tf.Tensor | None = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[TFSequenceClassifierOutputWithPast, Tuple[tf.Tensor]]:
        r"""
        labels (`np.ndarray` or `tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    The GPT-J Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, GPTJ_START_DOCSTRING)
class TFGPTJForQuestionAnswering(TFGPTJPreTrainedModel, TFQuestionAnsweringLoss):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., start_positions: np.ndarray | tf.Tensor | None = ..., end_positions: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        start_positions (`np.ndarray` or `tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`np.ndarray` or `tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...
