"""
This type stub file was generated by pyright.
"""

import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
from ...modeling_tf_outputs import TFBaseModelOutput, TFMultipleChoiceModelOutput, TFQuestionAnsweringModelOutput, TFSequenceClassifierOutput, TFTokenClassifierOutput
from ...modeling_tf_utils import TFModelInputType, TFMultipleChoiceLoss, TFPreTrainedModel, TFQuestionAnsweringLoss, TFSequenceClassificationLoss, TFTokenClassificationLoss, keras, keras_serializable, unpack_inputs
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_flaubert import FlaubertConfig

"""
TF 2.0 Flaubert model.
"""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
FLAUBERT_START_DOCSTRING = ...
FLAUBERT_INPUTS_DOCSTRING = ...
def get_masks(slen, lengths, causal, padding_mask=...): # -> tuple[Any, Any]:
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    ...

class TFFlaubertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = FlaubertConfig
    base_model_prefix = ...
    @property
    def dummy_inputs(self): # -> dict[str, Any]:
        ...
    


@add_start_docstrings("The bare Flaubert Model transformer outputting raw hidden-states without any specific head on top.", FLAUBERT_START_DOCSTRING)
class TFFlaubertModel(TFFlaubertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @unpack_inputs
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: np.ndarray | tf.Tensor | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., langs: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., lengths: np.ndarray | tf.Tensor | None = ..., cache: Optional[Dict[str, tf.Tensor]] = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[Tuple, TFBaseModelOutput]:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


class TFFlaubertMultiHeadAttention(keras.layers.Layer):
    NEW_ID = ...
    def __init__(self, n_heads, dim, config, **kwargs) -> None:
        ...
    
    def prune_heads(self, heads):
        ...
    
    def call(self, input, mask, kv, cache, head_mask, output_attentions, training=...): # -> tuple[Any, Any] | tuple[Any]:
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


class TFFlaubertTransformerFFN(keras.layers.Layer):
    def __init__(self, in_dim, dim_hidden, out_dim, config, **kwargs) -> None:
        ...
    
    def call(self, input, training=...):
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


@keras_serializable
class TFFlaubertMainLayer(keras.layers.Layer):
    config_class = FlaubertConfig
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    
    def get_input_embeddings(self): # -> TFSharedEmbeddings:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    @unpack_inputs
    def call(self, input_ids: np.ndarray | tf.Tensor | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., langs: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., lengths: np.ndarray | tf.Tensor | None = ..., cache: Optional[Dict[str, tf.Tensor]] = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[Tuple, TFBaseModelOutput]:
        ...
    


class TFFlaubertPredLayer(keras.layers.Layer):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """
    def __init__(self, config, input_embeddings, **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def get_output_embeddings(self): # -> Any:
        ...
    
    def set_output_embeddings(self, value): # -> None:
        ...
    
    def get_bias(self): # -> dict[str, Any]:
        ...
    
    def set_bias(self, value): # -> None:
        ...
    
    def call(self, hidden_states):
        ...
    


@dataclass
class TFFlaubertWithLMHeadModelOutput(ModelOutput):
    """
    Base class for [`TFFlaubertWithLMHeadModel`] outputs.

    Args:
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    logits: tf.Tensor = ...
    hidden_states: Tuple[tf.Tensor] | None = ...
    attentions: Tuple[tf.Tensor] | None = ...


@add_start_docstrings("""
    The Flaubert Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """, FLAUBERT_START_DOCSTRING)
class TFFlaubertWithLMHeadModel(TFFlaubertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_lm_head(self): # -> TFFlaubertPredLayer:
        ...
    
    def get_prefix_bias_name(self):
        ...
    
    def prepare_inputs_for_generation(self, inputs, **kwargs): # -> dict[str, Any]:
        ...
    
    @unpack_inputs
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFFlaubertWithLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: np.ndarray | tf.Tensor | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., langs: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., lengths: np.ndarray | tf.Tensor | None = ..., cache: Optional[Dict[str, tf.Tensor]] = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[Tuple, TFFlaubertWithLMHeadModelOutput]:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


@add_start_docstrings("""
    Flaubert Model with a sequence classification/regression head on top (a linear layer on top of the pooled output)
    e.g. for GLUE tasks.
    """, FLAUBERT_START_DOCSTRING)
class TFFlaubertForSequenceClassification(TFFlaubertPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @unpack_inputs
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., langs: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., lengths: np.ndarray | tf.Tensor | None = ..., cache: Optional[Dict[str, tf.Tensor]] = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: bool = ...) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


@add_start_docstrings("""
    Flaubert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, FLAUBERT_START_DOCSTRING)
class TFFlaubertForQuestionAnsweringSimple(TFFlaubertPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @unpack_inputs
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., langs: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., lengths: np.ndarray | tf.Tensor | None = ..., cache: Optional[Dict[str, tf.Tensor]] = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., start_positions: np.ndarray | tf.Tensor | None = ..., end_positions: np.ndarray | tf.Tensor | None = ..., training: bool = ...) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


@add_start_docstrings("""
    Flaubert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, FLAUBERT_START_DOCSTRING)
class TFFlaubertForTokenClassification(TFFlaubertPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @unpack_inputs
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., langs: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., lengths: np.ndarray | tf.Tensor | None = ..., cache: Optional[Dict[str, tf.Tensor]] = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: bool = ...) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


@add_start_docstrings("""
    Flaubert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, FLAUBERT_START_DOCSTRING)
class TFFlaubertForMultipleChoice(TFFlaubertPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @property
    def dummy_inputs(self): # -> dict[str, Any]:
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        ...
    
    @unpack_inputs
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., langs: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., lengths: np.ndarray | tf.Tensor | None = ..., cache: Optional[Dict[str, tf.Tensor]] = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: bool = ...) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


