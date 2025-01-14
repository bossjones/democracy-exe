"""
This type stub file was generated by pyright.
"""

import random
import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Optional, Tuple
from flax.core.frozen_dict import FrozenDict
from jax.random import PRNGKey
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPastAndCrossAttentions, FlaxCausalLMOutputWithCrossAttentions
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_whisper import WhisperConfig

"""Flax whisper model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
remat = ...
def sinusoidal_embedding_init(key, shape, dtype=...) -> jax.Array:
    """Returns sinusoids for positional embedding"""
    ...

WHISPER_START_DOCSTRING = ...
WHISPER_INPUTS_DOCSTRING = ...
WHISPER_ENCODE_INPUTS_DOCSTRING = ...
WHISPER_DECODE_INPUTS_DOCSTRING = ...
class FlaxWhisperAttention(nn.Module):
    config: WhisperConfig
    embed_dim: int
    num_heads: int
    dropout: float = ...
    causal: bool = ...
    bias: bool = ...
    dtype: jnp.dtype = ...
    def setup(self) -> None:
        ...
    
    def __call__(self, hidden_states: jnp.ndarray, key_value_states: Optional[jnp.ndarray] = ..., attention_mask: Optional[jnp.ndarray] = ..., init_cache: bool = ..., deterministic: bool = ...) -> Tuple[jnp.ndarray]:
        ...
    


class FlaxWhisperEncoderLayer(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = ...
    def setup(self) -> None:
        ...
    
    def __call__(self, hidden_states: jnp.ndarray, attention_mask: jnp.ndarray, output_attentions: bool = ..., deterministic: bool = ...) -> Tuple[jnp.ndarray]:
        ...
    


class FlaxWhisperEncoderLayerCollection(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any | tuple[Any | None, ...] | tuple[()], ...] | FlaxBaseModelOutput:
        ...
    


class FlaxWhisperDecoderLayer(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = ...
    def setup(self) -> None:
        ...
    
    def __call__(self, hidden_states: jnp.ndarray, attention_mask: jnp.ndarray, encoder_hidden_states: Optional[jnp.ndarray] = ..., encoder_attention_mask: Optional[jnp.ndarray] = ..., init_cache: bool = ..., output_attentions: bool = ..., deterministic: bool = ...) -> Tuple[jnp.ndarray]:
        ...
    


class FlaxWhisperDecoderLayerCollection(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, encoder_hidden_states: Optional[jnp.ndarray] = ..., encoder_attention_mask: Optional[jnp.ndarray] = ..., deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...
    


class FlaxWhisperEncoder(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self) -> None:
        ...
    
    def __call__(self, input_features: jnp.ndarray, output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...) -> Tuple[jnp.ndarray]:
        ...
    


class FlaxWhisperDecoder(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self) -> None:
        ...
    
    def __call__(self, input_ids: jnp.ndarray, attention_mask: jnp.ndarray, position_ids: jnp.ndarray, encoder_hidden_states: Optional[jnp.ndarray] = ..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...) -> Tuple[jnp.ndarray]:
        ...
    


class FlaxWhisperModule(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self) -> None:
        ...
    
    def __call__(self, input_features: jnp.ndarray, decoder_input_ids: jnp.ndarray, decoder_attention_mask: jnp.ndarray, decoder_position_ids: jnp.ndarray, output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> tuple[Any, Any] | FlaxSeq2SeqModelOutput:
        ...
    


class FlaxWhisperPreTrainedModel(FlaxPreTrainedModel):
    config_class = WhisperConfig
    base_model_prefix: str = ...
    main_input_name = ...
    module_class: nn.Module = ...
    def __init__(self, config: WhisperConfig, input_shape: Tuple[int] = ..., seed: int = ..., dtype: jnp.dtype = ..., _do_init: bool = ..., gradient_checkpointing: bool = ..., **kwargs) -> None:
        ...
    
    def enable_gradient_checkpointing(self): # -> None:
        ...
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = ...) -> FrozenDict:
        ...
    
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*:
                `attentions`). `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*)
                is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
                cross-attention of the decoder.
        """
        ...
    
    @add_start_docstrings(WHISPER_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=WhisperConfig)
    def encode(self, input_features: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ..., **kwargs):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
        >>> input_features = inputs.input_features
        >>> encoder_outputs = model.encode(input_features=input_features)
        ```"""
        ...
    
    @add_start_docstrings(WHISPER_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=WhisperConfig)
    def decode(self, decoder_input_ids, encoder_outputs, encoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_position_ids: Optional[jnp.ndarray] = ..., past_key_values: dict = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
        >>> from datasets import load_dataset
        >>> import jax.numpy as jnp

        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> input_features = processor(ds[0]["audio"]["array"], return_tensors="np").input_features

        >>> encoder_outputs = model.encode(input_features=input_features)
        >>> decoder_start_token_id = model.config.decoder_start_token_id

        >>> decoder_input_ids = jnp.ones((input_features.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> last_decoder_hidden_states = outputs.last_hidden_state
        ```"""
        ...
    
    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    def __call__(self, input_features: jnp.ndarray, decoder_input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = ..., decoder_attention_mask: Optional[jnp.ndarray] = ..., position_ids: Optional[jnp.ndarray] = ..., decoder_position_ids: Optional[jnp.ndarray] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...):
        ...
    


@add_start_docstrings("The bare Whisper Model transformer outputting raw hidden-states without any specific head on top.", WHISPER_START_DOCSTRING)
class FlaxWhisperModel(FlaxWhisperPreTrainedModel):
    config: WhisperConfig
    dtype: jnp.dtype = ...
    module_class = ...


class FlaxWhisperForConditionalGenerationModule(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self) -> None:
        ...
    
    def __call__(self, input_features, decoder_input_ids, decoder_attention_mask: jnp.ndarray = ..., decoder_position_ids: jnp.ndarray = ..., position_ids: jnp.ndarray = ..., attention_mask: jnp.ndarray = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> tuple[Any, Any] | Any | FlaxSeq2SeqLMOutput:
        ...
    


@add_start_docstrings("The Whisper Model with a language modeling head.", WHISPER_START_DOCSTRING)
class FlaxWhisperForConditionalGeneration(FlaxWhisperPreTrainedModel):
    module_class = ...
    dtype: jnp.dtype = ...
    @add_start_docstrings(WHISPER_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=WhisperConfig)
    def decode(self, decoder_input_ids, encoder_outputs, encoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_position_ids: Optional[jnp.ndarray] = ..., past_key_values: dict = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...): # -> FlaxCausalLMOutputWithCrossAttentions | Any:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
        >>> input_features = inputs.input_features
        >>> encoder_outputs = model.encode(input_features=input_features)
        >>> decoder_start_token_id = model.config.decoder_start_token_id

        >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> last_decoder_hidden_states = outputs.last_hidden_state
        ```"""
        ...
    
    def generate(self, input_features, generation_config=..., logits_processor=..., return_timestamps=..., task=..., language=..., is_multilingual=..., **kwargs): # -> FlaxGreedySearchOutput | FlaxSampleOutput | FlaxBeamSearchOutput:
        ...
    
    def prepare_inputs_for_generation(self, decoder_input_ids, max_length, attention_mask: Optional[jax.Array] = ..., decoder_attention_mask: Optional[jax.Array] = ..., encoder_outputs=..., **kwargs): # -> dict[str, Any]:
        ...
    
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        ...
    


FLAX_WHISPER_CONDITIONAL_GENERATION_DOCSTRING = ...
class FlaxWhisperForAudioClassificationModule(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self) -> None:
        ...
    
    def __call__(self, input_features, encoder_outputs=..., output_attentions=..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any] | FlaxSequenceClassifierOutput:
        ...
    


@add_start_docstrings("The Whisper Model with an audio classification head on top.", WHISPER_START_DOCSTRING)
class FlaxWhisperForAudioClassification(FlaxWhisperPreTrainedModel):
    module_class = ...
    dtype: jnp.dtype = ...
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = ...) -> FrozenDict:
        ...
    
    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    def __call__(self, input_features: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ..., **kwargs):
        ...
    


FLAX_WHISPER_AUDIO_CLASSIFICATION_DOCSTRING = ...
