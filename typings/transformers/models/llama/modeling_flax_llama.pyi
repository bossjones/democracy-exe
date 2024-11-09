"""
This type stub file was generated by pyright.
"""

from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_llama import LlamaConfig

"""Flax LLaMA model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_REAL_CHECKPOINT_FOR_DOC = ...
LLAMA_START_DOCSTRING = ...
LLAMA_INPUTS_DOCSTRING = ...
def create_sinusoidal_positions(num_pos, dim):
    ...

def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    ...

def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    ...

class FlaxLlamaRMSNorm(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...

    def __call__(self, hidden_states):
        ...



class FlaxLlamaRotaryEmbedding(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...

    def __call__(self, key, query, position_ids): # -> tuple[Any, Any]:
        ...



class FlaxLlamaAttention(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = ...
    causal: bool = ...
    is_cross_attention: bool = ...
    def setup(self): # -> None:
        ...

    def __call__(self, hidden_states, attention_mask, position_ids, deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ...): # -> tuple[Any, Any] | tuple[Any]:
        ...



class FlaxLlamaMLP(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...

    def __call__(self, hidden_states):
        ...



class FlaxLlamaDecoderLayer(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...

    def __call__(self, hidden_states, attention_mask=..., position_ids=..., deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ...): # -> tuple[Any, Any] | tuple[Any]:
        ...



class FlaxLlamaPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LlamaConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(self, config: LlamaConfig, input_shape: Tuple = ..., seed: int = ..., dtype: jnp.dtype = ..., _do_init: bool = ..., **kwargs) -> None:
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

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def __call__(self, input_ids, attention_mask=..., position_ids=..., params: dict = ..., past_key_values: dict = ..., dropout_rng: jax.random.PRNGKey = ..., train: bool = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...):
        ...



class FlaxLlamaLayerCollection(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...

    def __call__(self, hidden_states, attention_mask=..., position_ids=..., deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, tuple[()] | Any | None, tuple[()] | Any | None]:
        ...



class FlaxLlamaModule(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...

    def __call__(self, input_ids, attention_mask=..., position_ids=..., deterministic=..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any | tuple[()], ...] | FlaxBaseModelOutput:
        ...



@add_start_docstrings("The bare Llama Model transformer outputting raw hidden-states without any specific head on top.", LLAMA_START_DOCSTRING)
class FlaxLlamaModel(FlaxLlamaPreTrainedModel):
    module_class = ...


class FlaxLlamaForCausalLMModule(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...

    def __call__(self, input_ids, attention_mask=..., position_ids=..., deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, *tuple[Any | tuple[()], ...]] | Any | FlaxCausalLMOutput:
        ...



@add_start_docstrings("""
    The Llama Model transformer with a language modeling head (linear layer) on top.
    """, LLAMA_START_DOCSTRING)
class FlaxLlamaForCausalLM(FlaxLlamaPreTrainedModel):
    module_class = ...
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = ...): # -> dict[str, Any]:
        ...

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        ...
