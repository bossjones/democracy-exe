"""
This type stub file was generated by pyright.
"""

from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...modeling_tf_outputs import TFBaseModelOutputWithPast, TFCausalLMOutputWithPast, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_ctrl import CTRLConfig

""" TF 2.0 CTRL model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
def angle_defn(pos, i, d_model_size):
    ...

def positional_encoding(position, d_model_size):
    ...

def scaled_dot_product_attention(q, k, v, mask, attention_mask=..., head_mask=...): # -> tuple[Any, Any]:
    ...

class TFMultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model_size, num_heads, output_attentions=..., **kwargs) -> None:
        ...

    def split_into_heads(self, x, batch_size):
        ...

    def call(self, v, k, q, mask, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=...): # -> tuple[Any, Any | tuple[None], Any] | tuple[Any, Any | tuple[None]]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFPointWiseFeedForwardLayer(keras.layers.Layer):
    def __init__(self, d_model_size, dff, **kwargs) -> None:
        ...

    def call(self, inputs, trainable=...):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFEncoderLayer(keras.layers.Layer):
    def __init__(self, d_model_size, num_heads, dff, rate=..., layer_norm_epsilon=..., output_attentions=..., **kwargs) -> None:
        ...

    def call(self, x, mask, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=...):
        ...

    def build(self, input_shape=...): # -> None:
        ...



@keras_serializable
class TFCTRLMainLayer(keras.layers.Layer):
    config_class = CTRLConfig
    def __init__(self, config, **kwargs) -> None:
        ...

    def get_input_embeddings(self):
        ...

    def set_input_embeddings(self, new_embeddings): # -> None:
        ...

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None = ..., past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[Tuple, TFBaseModelOutputWithPast]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFCTRLPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = CTRLConfig
    base_model_prefix = ...


CTRL_START_DOCSTRING = ...
CTRL_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare CTRL Model transformer outputting raw hidden-states without any specific head on top.", CTRL_START_DOCSTRING)
class TFCTRLModel(TFCTRLPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[Tuple, TFBaseModelOutputWithPast]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFCTRLBiasLayer(keras.layers.Layer):
    """
    Bias as a layer. It is used for serialization purposes: `keras.Model.save_weights` stores on a per-layer basis,
    so all weights have to be registered in a layer.
    """
    def __init__(self, shape, initializer, trainable, name, **kwargs) -> None:
        ...

    def build(self, input_shape): # -> None:
        ...

    def call(self, x):
        ...



@add_start_docstrings("""
    The CTRL Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """, CTRL_START_DOCSTRING)
class TFCTRLLMHeadModel(TFCTRLPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    def get_output_embeddings(self):
        ...

    def set_output_embeddings(self, value): # -> None:
        ...

    def get_bias(self): # -> dict[str, Any]:
        ...

    def set_bias(self, value): # -> None:
        ...

    def prepare_inputs_for_generation(self, inputs, past_key_values=..., use_cache=..., **kwargs): # -> dict[str, Any]:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[Tuple, TFCausalLMOutputWithPast]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    The CTRL Model transformer with a sequence classification head on top (linear layer).

    [`TFCTRLForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1, GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """, CTRL_START_DOCSTRING)
class TFCTRLForSequenceClassification(TFCTRLPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    def get_output_embeddings(self):
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[Tuple, TFSequenceClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...
