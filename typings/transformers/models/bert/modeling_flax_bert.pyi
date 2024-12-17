"""
This type stub file was generated by pyright.
"""

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional, Tuple
from flax.core.frozen_dict import FrozenDict
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_bert import BertConfig

logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
remat = ...
@flax.struct.dataclass
class FlaxBertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertForPreTraining`].

    Args:
        prediction_logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`jnp.ndarray` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    prediction_logits: jnp.ndarray = ...
    seq_relationship_logits: jnp.ndarray = ...
    hidden_states: Optional[Tuple[jnp.ndarray]] = ...
    attentions: Optional[Tuple[jnp.ndarray]] = ...


BERT_START_DOCSTRING = ...
BERT_INPUTS_DOCSTRING = ...
class FlaxBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = ...):
        ...
    


class FlaxBertSelfAttention(nn.Module):
    config: BertConfig
    causal: bool = ...
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, layer_head_mask, key_value_states: Optional[jnp.ndarray] = ..., init_cache: bool = ..., deterministic=..., output_attentions: bool = ...): # -> tuple[Any, Any] | tuple[Any]:
        ...
    


class FlaxBertSelfOutput(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, input_tensor, deterministic: bool = ...):
        ...
    


class FlaxBertAttention(nn.Module):
    config: BertConfig
    causal: bool = ...
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, layer_head_mask, key_value_states=..., init_cache=..., deterministic=..., output_attentions: bool = ...): # -> tuple[Any, Any] | tuple[Any]:
        ...
    


class FlaxBertIntermediate(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


class FlaxBertOutput(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_output, deterministic: bool = ...):
        ...
    


class FlaxBertLayer(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, layer_head_mask, encoder_hidden_states: Optional[jnp.ndarray] = ..., encoder_attention_mask: Optional[jnp.ndarray] = ..., init_cache: bool = ..., deterministic: bool = ..., output_attentions: bool = ...): # -> tuple[Any, Any, Any] | tuple[Any, Any] | tuple[Any]:
        ...
    


class FlaxBertLayerCollection(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, head_mask, encoder_hidden_states: Optional[jnp.ndarray] = ..., encoder_attention_mask: Optional[jnp.ndarray] = ..., init_cache: bool = ..., deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...
    


class FlaxBertEncoder(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, head_mask, encoder_hidden_states: Optional[jnp.ndarray] = ..., encoder_attention_mask: Optional[jnp.ndarray] = ..., init_cache: bool = ..., deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...
    


class FlaxBertPooler(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


class FlaxBertPredictionHeadTransform(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


class FlaxBertLMPredictionHead(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    bias_init: Callable[..., np.ndarray] = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, shared_embedding=...):
        ...
    


class FlaxBertOnlyMLMHead(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, shared_embedding=...):
        ...
    


class FlaxBertOnlyNSPHead(nn.Module):
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, pooled_output):
        ...
    


class FlaxBertPreTrainingHeads(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, pooled_output, shared_embedding=...): # -> tuple[Any, Any]:
        ...
    


class FlaxBertPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BertConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(self, config: BertConfig, input_shape: Tuple = ..., seed: int = ..., dtype: jnp.dtype = ..., _do_init: bool = ..., gradient_checkpointing: bool = ..., **kwargs) -> None:
        ...
    
    def enable_gradient_checkpointing(self): # -> None:
        ...
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = ...) -> FrozenDict:
        ...
    
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        ...
    
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(self, input_ids, attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., params: dict = ..., dropout_rng: jax.random.PRNGKey = ..., train: bool = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., past_key_values: dict = ...):
        ...
    


class FlaxBertModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    add_pooling_layer: bool = ...
    gradient_checkpointing: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids: Optional[jnp.ndarray] = ..., position_ids: Optional[jnp.ndarray] = ..., head_mask: Optional[jnp.ndarray] = ..., encoder_hidden_states: Optional[jnp.ndarray] = ..., encoder_attention_mask: Optional[jnp.ndarray] = ..., init_cache: bool = ..., deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any | tuple[Any, ...] | tuple[()], *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | Any | tuple[Any | tuple[Any, ...] | tuple[()], Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | FlaxBaseModelOutputWithPoolingAndCrossAttentions:
        ...
    


@add_start_docstrings("The bare Bert Model transformer outputting raw hidden-states without any specific head on top.", BERT_START_DOCSTRING)
class FlaxBertModel(FlaxBertPreTrainedModel):
    module_class = ...


class FlaxBertForPreTrainingModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | Any | FlaxBertForPreTrainingOutput:
        ...
    


@add_start_docstrings("""
    Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """, BERT_START_DOCSTRING)
class FlaxBertForPreTraining(FlaxBertPreTrainedModel):
    module_class = ...


FLAX_BERT_FOR_PRETRAINING_DOCSTRING = ...
class FlaxBertForMaskedLMModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | Any | tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | FlaxMaskedLMOutput:
        ...
    


@add_start_docstrings("""Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING)
class FlaxBertForMaskedLM(FlaxBertPreTrainedModel):
    module_class = ...


class FlaxBertForNextSentencePredictionModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | Any | FlaxNextSentencePredictorOutput:
        ...
    


@add_start_docstrings("""Bert Model with a `next sentence prediction (classification)` head on top.""", BERT_START_DOCSTRING)
class FlaxBertForNextSentencePrediction(FlaxBertPreTrainedModel):
    module_class = ...


FLAX_BERT_FOR_NEXT_SENT_PRED_DOCSTRING = ...
class FlaxBertForSequenceClassificationModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | Any | FlaxSequenceClassifierOutput:
        ...
    


@add_start_docstrings("""
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """, BERT_START_DOCSTRING)
class FlaxBertForSequenceClassification(FlaxBertPreTrainedModel):
    module_class = ...


class FlaxBertForMultipleChoiceModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | Any | FlaxMultipleChoiceModelOutput:
        ...
    


@add_start_docstrings("""
    Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, BERT_START_DOCSTRING)
class FlaxBertForMultipleChoice(FlaxBertPreTrainedModel):
    module_class = ...


class FlaxBertForTokenClassificationModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | Any | tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | FlaxTokenClassifierOutput:
        ...
    


@add_start_docstrings("""
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, BERT_START_DOCSTRING)
class FlaxBertForTokenClassification(FlaxBertPreTrainedModel):
    module_class = ...


class FlaxBertForQuestionAnsweringModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | Any | tuple[Any, Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | FlaxQuestionAnsweringModelOutput:
        ...
    


@add_start_docstrings("""
    Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, BERT_START_DOCSTRING)
class FlaxBertForQuestionAnswering(FlaxBertPreTrainedModel):
    module_class = ...


class FlaxBertForCausalLMModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, position_ids, token_type_ids: Optional[jnp.ndarray] = ..., head_mask: Optional[jnp.ndarray] = ..., encoder_hidden_states: Optional[jnp.ndarray] = ..., encoder_attention_mask: Optional[jnp.ndarray] = ..., init_cache: bool = ..., deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | Any | tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | FlaxCausalLMOutputWithCrossAttentions:
        ...
    


@add_start_docstrings("""
    Bert Model with a language modeling head on top (a linear layer on top of the hidden-states output) e.g for
    autoregressive tasks.
    """, BERT_START_DOCSTRING)
class FlaxBertForCausalLM(FlaxBertPreTrainedModel):
    module_class = ...
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = ...): # -> dict[str, Any]:
        ...
    
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        ...
    


