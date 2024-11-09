"""
This type stub file was generated by pyright.
"""

import enum
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPooling,
    TFMaskedLMOutput,
    TFSequenceClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_tensorflow_probability_available,
    replace_return_docstrings,
)
from .configuration_tapas import TapasConfig

"""TF 2.0 TAPAS model."""
logger = ...
if is_tensorflow_probability_available():
    n = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
EPSILON_ZERO_DIVISION = ...
CLOSE_ENOUGH_TO_LOG_ZERO = ...
@dataclass
class TFTableQuestionAnsweringOutput(ModelOutput):
    """
    Output type of [`TFTapasForQuestionAnswering`].

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` (and possibly `answer`, `aggregation_labels`, `numeric_values` and `numeric_values_scale` are provided)):
            Total loss as the sum of the hierarchical cell selection log-likelihood loss and (optionally) the
            semi-supervised regression loss and (optionally) supervised loss for aggregations.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Prediction scores of the cell selection head, for every token.
        logits_aggregation (`tf.Tensor`, *optional*, of shape `(batch_size, num_aggregation_labels)`):
            Prediction scores of the aggregation head, for every aggregation operator.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """
    loss: tf.Tensor | None = ...
    logits: tf.Tensor = ...
    logits_aggregation: tf.Tensor | None = ...
    hidden_states: Tuple[tf.Tensor] | None = ...
    attentions: Tuple[tf.Tensor] | None = ...


class TFTapasEmbeddings(keras.layers.Layer):
    """
    Construct the embeddings from word, position and token_type embeddings. Same as BertEmbeddings but with a number of
    additional token type embeddings to encode tabular structure.
    """
    def __init__(self, config: TapasConfig, **kwargs) -> None:
        ...

    def build(self, input_shape=...): # -> None:
        ...

    def call(self, input_ids: tf.Tensor = ..., position_ids: tf.Tensor = ..., token_type_ids: tf.Tensor = ..., inputs_embeds: tf.Tensor = ..., training: bool = ...) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        ...



class TFTapasSelfAttention(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None:
        ...

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        ...

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, encoder_hidden_states: tf.Tensor, encoder_attention_mask: tf.Tensor, past_key_value: Tuple[tf.Tensor], output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFTapasSelfOutput(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFTapasAttention(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None:
        ...

    def prune_heads(self, heads):
        ...

    def call(self, input_tensor: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, encoder_hidden_states: tf.Tensor, encoder_attention_mask: tf.Tensor, past_key_value: Tuple[tf.Tensor], output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFTapasIntermediate(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFTapasOutput(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFTapasLayer(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, encoder_hidden_states: tf.Tensor | None, encoder_attention_mask: tf.Tensor | None, past_key_value: Tuple[tf.Tensor] | None, output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFTapasEncoder(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, encoder_hidden_states: tf.Tensor | None, encoder_attention_mask: tf.Tensor | None, past_key_values: Tuple[Tuple[tf.Tensor]] | None, use_cache: Optional[bool], output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool = ...) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFTapasPooler(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFTapasPredictionHeadTransform(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFTapasLMPredictionHead(keras.layers.Layer):
    def __init__(self, config: TapasConfig, input_embeddings: keras.layers.Layer, **kwargs) -> None:
        ...

    def build(self, input_shape=...): # -> None:
        ...

    def get_output_embeddings(self) -> keras.layers.Layer:
        ...

    def set_output_embeddings(self, value: tf.Variable): # -> None:
        ...

    def get_bias(self) -> Dict[str, tf.Variable]:
        ...

    def set_bias(self, value: tf.Variable): # -> None:
        ...

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...



class TFTapasMLMHead(keras.layers.Layer):
    def __init__(self, config: TapasConfig, input_embeddings: keras.layers.Layer, **kwargs) -> None:
        ...

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



@keras_serializable
class TFTapasMainLayer(keras.layers.Layer):
    config_class = TapasConfig
    def __init__(self, config: TapasConfig, add_pooling_layer: bool = ..., **kwargs) -> None:
        ...

    def get_input_embeddings(self) -> keras.layers.Layer:
        ...

    def set_input_embeddings(self, value: tf.Variable): # -> None:
        ...

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: bool = ...) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFTapasPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = TapasConfig
    base_model_prefix = ...
    @property
    def input_signature(self): # -> dict[str, Any]:
        ...



TAPAS_START_DOCSTRING = ...
TAPAS_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Tapas Model transformer outputting raw hidden-states without any specific head on top.", TAPAS_START_DOCSTRING)
class TFTapasModel(TFTapasPreTrainedModel):
    def __init__(self, config: TapasConfig, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TapasModel
        >>> import pandas as pd

        >>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base")
        >>> model = TapasModel.from_pretrained("google/tapas-base")

        >>> data = {
        ...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        ...     "Age": ["56", "45", "59"],
        ...     "Number of movies": ["87", "53", "69"],
        ... }
        >>> table = pd.DataFrame.from_dict(data)
        >>> queries = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]

        >>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="tf")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""Tapas Model with a `language modeling` head on top.""", TAPAS_START_DOCSTRING)
class TFTapasForMaskedLM(TFTapasPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config: TapasConfig, *inputs, **kwargs) -> None:
        ...

    def get_lm_head(self) -> keras.layers.Layer:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TapasForMaskedLM
        >>> import pandas as pd

        >>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base")
        >>> model = TapasForMaskedLM.from_pretrained("google/tapas-base")

        >>> data = {
        ...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        ...     "Age": ["56", "45", "59"],
        ...     "Number of movies": ["87", "53", "69"],
        ... }
        >>> table = pd.DataFrame.from_dict(data)

        >>> inputs = tokenizer(
        ...     table=table, queries="How many [MASK] has George [MASK] played in?", return_tensors="tf"
        ... )
        >>> labels = tokenizer(
        ...     table=table, queries="How many movies has George Clooney played in?", return_tensors="tf"
        ... )["input_ids"]

        >>> outputs = model(**inputs, labels=labels)
        >>> logits = outputs.logits
        ```"""
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFTapasComputeTokenLogits(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None:
        ...

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        """
        Computes logits per token

        Args:
            sequence_output (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Also known as last_hidden_state. Sequence of hidden-states at the output of the last layer of the
                model.

        Returns:
            logits (`tf.Tensor` of shape `(batch_size, sequence_length)`): Logits per token.
        """
        ...



class TFTapasComputeColumnLogits(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None:
        ...

    def call(self, sequence_output, cell_index, cell_mask, allow_empty_column_selection) -> tf.Tensor:
        """
        Computes the column logits.

        Args:
            sequence_output (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Also known as last_hidden_state. Sequence of hidden-states at the output of the last layer of the
                model.
            cell_index (`ProductIndexMap`):
                Index that groups tokens into cells.
            cell_mask (`tf.Tensor` of shape `(batch_size, max_num_rows * max_num_cols)`):
                Mask for cells that exist in the table (i.e. that are not padding).
            allow_empty_column_selection (`bool`):
                Whether to allow not to select any column

        Returns:
            column_logits (`tf.Tensor`of shape `(batch_size, max_num_cols)`): Tensor containing the column logits for
            every example in the batch.
        """
        ...



@add_start_docstrings("""
    Tapas Model with a cell selection head and optional aggregation head on top for question-answering tasks on tables
    (linear layers on top of the hidden-states output to compute `logits` and optional `logits_aggregation`), e.g. for
    SQA, WTQ or WikiSQL-supervised tasks.
    """, TAPAS_START_DOCSTRING)
class TFTapasForQuestionAnswering(TFTapasPreTrainedModel):
    def __init__(self, config: TapasConfig, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFTableQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., table_mask: np.ndarray | tf.Tensor | None = ..., aggregation_labels: np.ndarray | tf.Tensor | None = ..., float_answer: np.ndarray | tf.Tensor | None = ..., numeric_values: np.ndarray | tf.Tensor | None = ..., numeric_values_scale: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFTableQuestionAnsweringOutput, Tuple[tf.Tensor]]:
        r"""
        table_mask (`tf.Tensor` of shape `(batch_size, seq_length)`, *optional*):
            Mask for the table. Indicates which tokens belong to the table (1). Question tokens, table headers and
            padding are 0.
        labels (`tf.Tensor` of shape `(batch_size, seq_length)`, *optional*):
            Labels per token for computing the hierarchical cell selection loss. This encodes the positions of the
            answer appearing in the table. Can be obtained using [`AutoTokenizer`].

            - 1 for tokens that are **part of the answer**,
            - 0 for tokens that are **not part of the answer**.

        aggregation_labels (`tf.Tensor` of shape `(batch_size, )`, *optional*):
            Aggregation function index for every example in the batch for computing the aggregation loss. Indices
            should be in `[0, ..., config.num_aggregation_labels - 1]`. Only required in case of strong supervision for
            aggregation (WikiSQL-supervised).
        float_answer (`tf.Tensor` of shape `(batch_size, )`, *optional*):
            Float answer for every example in the batch. Set to *float('nan')* for cell selection questions. Only
            required in case of weak supervision (WTQ) to calculate the aggregate mask and regression loss.
        numeric_values (`tf.Tensor` of shape `(batch_size, seq_length)`, *optional*):
            Numeric values of every token, NaN for tokens which are not numeric values. Can be obtained using
            [`AutoTokenizer`]. Only required in case of weak supervision for aggregation (WTQ) to calculate the
            regression loss.
        numeric_values_scale (`tf.Tensor` of shape `(batch_size, seq_length)`, *optional*):
            Scale of the numeric values of every token. Can be obtained using [`AutoTokenizer`]. Only required in case
            of weak supervision for aggregation (WTQ) to calculate the regression loss.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TapasForQuestionAnswering
        >>> import pandas as pd

        >>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
        >>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

        >>> data = {
        ...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        ...     "Age": ["56", "45", "59"],
        ...     "Number of movies": ["87", "53", "69"],
        ... }
        >>> table = pd.DataFrame.from_dict(data)
        >>> queries = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]

        >>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="tf")
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> logits_aggregation = outputs.logits_aggregation
        ```"""
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    Tapas Model with a sequence classification head on top (a linear layer on top of the pooled output), e.g. for table
    entailment tasks, such as TabFact (Chen et al., 2020).
    """, TAPAS_START_DOCSTRING)
class TFTapasForSequenceClassification(TFTapasPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: TapasConfig, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @replace_return_docstrings(output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy). Note: this is called
            "classification_class_index" in the original implementation.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TapasForSequenceClassification
        >>> import tensorflow as tf
        >>> import pandas as pd

        >>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-tabfact")
        >>> model = TapasForSequenceClassification.from_pretrained("google/tapas-base-finetuned-tabfact")

        >>> data = {
        ...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        ...     "Age": ["56", "45", "59"],
        ...     "Number of movies": ["87", "53", "69"],
        ... }
        >>> table = pd.DataFrame.from_dict(data)
        >>> queries = [
        ...     "There is only one actor who is 45 years old",
        ...     "There are 3 actors which played in more than 60 movies",
        ... ]

        >>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="tf")
        >>> labels = tf.convert_to_tensor([1, 0])  # 1 means entailed, 0 means refuted

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        ...

    def build(self, input_shape=...): # -> None:
        ...



class AverageApproximationFunction(str, enum.Enum):
    RATIO = ...
    FIRST_ORDER = ...
    SECOND_ORDER = ...


class IndexMap:
    """Index grouping entries within a tensor."""
    def __init__(self, indices, num_segments, batch_dims=...) -> None:
        """
        Creates an index.

        Args:
          indices: <int32> Tensor of indices, same shape as `values`.
          num_segments: <int32> Scalar tensor, the number of segments. All elements
            in a batched segmented tensor must have the same number of segments (although many segments can be empty).
          batch_dims: Python integer, the number of batch dimensions. The first
            `batch_dims` dimensions of a SegmentedTensor are treated as batch dimensions. Segments in different batch
            elements are always distinct even if they have the same index.
        """
        ...

    def batch_shape(self):
        ...



class ProductIndexMap(IndexMap):
    """The product of two indices."""
    def __init__(self, outer_index, inner_index) -> None:
        """
        Combines indices i and j into pairs (i, j). The result is an index where each segment (i, j) is the
        intersection of segments i and j. For example if the inputs represent table cells indexed by respectively rows
        and columns the output will be a table indexed by (row, column) pairs, i.e. by cell. The implementation
        combines indices {0, .., n - 1} and {0, .., m - 1} into {0, .., nm - 1}. The output has `num_segments` equal to
        `outer_index.num_segements` * `inner_index.num_segments`.

        Args:
          outer_index: IndexMap.
          inner_index: IndexMap, must have the same shape as `outer_index`.
        """
        ...

    def project_outer(self, index): # -> IndexMap:
        """Projects an index with the same index set onto the outer components."""
        ...

    def project_inner(self, index): # -> IndexMap:
        """Projects an index with the same index set onto the inner components."""
        ...



def gather(values, index, name=...):
    """
    Gathers from `values` using the index map. For each element in the domain of the index map this operation looks up
    a value for that index in `values`. Two elements from the same segment always get assigned the same value.

    Args:
      values: [B1, ..., Bn, num_segments, V1, ...] Tensor with segment values.
      index: [B1, ..., Bn, I1, ..., Ik] IndexMap.
      name: Name for the TensorFlow operation.

    Returns:
      [B1, ..., Bn, I1, ..., Ik, V1, ...] Tensor with the gathered values.
    """
    ...

def flatten(index, name=...): # -> IndexMap:
    """
    Flattens a batched index map to a 1d index map. This operation relabels the segments to keep batch elements
    distinct. The k-th batch element will have indices shifted by `num_segments` * (k - 1). The result is a tensor with
    `num_segments` multiplied by the number of elements in the batch.

    Args:
      index: IndexMap to flatten.
      name: Name for the TensorFlow operation.

    Returns:
      The flattened IndexMap.
    """
    ...

def range_index_map(batch_shape, num_segments, name=...): # -> IndexMap:
    """
    Constructs an index map equal to range(num_segments).

    Args:
        batch_shape (`tf.Tensor`):
            Batch shape
        num_segments (`int`):
            Number of segments
        name (`str`, *optional*, defaults to 'range_index_map'):
            Name for the operation. Currently not used

    Returns:
        (`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    ...

def reduce_mean(values, index, name=...): # -> tuple[Any, IndexMap]:
    """
    Averages a tensor over its segments. Outputs 0 for empty segments. This operations computes the mean over segments,
    with support for:

      - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
      - Vectorization using the last dimension [V1, V2, ...]. If they are present the output will be a mean of vectors
        rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
      values: [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..] tensor of values to be
        averaged.
      index: IndexMap [B1, B2, ..., Bn, I1, .., Ik] index defining the segments.
      name: Name for the TensorFlow ops.

    Returns:
      A pair (output_values, output_index) where `output_values` is a tensor of shape [B1, B2, ..., Bn, num_segments,
      V1, V2, ..] and `index` is an IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    ...

def reduce_sum(values, index, name=...): # -> tuple[Any, IndexMap]:
    """
    Sums a tensor over its segments. Outputs 0 for empty segments. This operations computes the sum over segments, with
    support for:

      - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
      - Vectorization using the last dimension [V1, V2, ...]. If they are present the output will be a sum of vectors
        rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
      values: [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..] tensor of values to be
        averaged.
      index: IndexMap [B1, B2, ..., Bn, I1, .., Ik] index defining the segments.
      name: Name for the TensorFlow ops.

    Returns:
      A pair (output_values, output_index) where `output_values` is a tensor of shape [B1, B2, ..., Bn, num_segments,
      V1, V2, ..] and `index` is an IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    ...

def reduce_max(values, index, name=...): # -> tuple[Any, IndexMap]:
    """
    Computes the maximum over segments. This operations computes the maximum over segments, with support for:

      - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
      - Vectorization using the last dimension [V1, V2, ...]. If they are present the output will be an element-wise
        maximum of vectors rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
      values: [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..] tensor of values to be
        averaged.
      index: IndexMap [B1, B2, ..., Bn, I1, .., Ik] index defining the segments.
      name: Name for the TensorFlow ops.

    Returns:
      A pair (output_values, output_index) where `output_values` is a tensor of shape [B1, B2, ..., Bn, num_segments,
      V1, V2, ..] and `index` is an IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    ...

def reduce_min(values, index, name=...): # -> tuple[Any, IndexMap]:
    """Computes the minimum over segments."""
    ...
