"""
This type stub file was generated by pyright.
"""

import torch
from typing import Optional, Tuple, Union
from torch import nn
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_codegen import CodeGenConfig

"""PyTorch CodeGen model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    ...

def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    ...

def apply_rotary_pos_emb(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    ...

class CodeGenAttention(nn.Module):
    def __init__(self, config, layer_idx=...) -> None:
        ...
    
    def forward(self, hidden_states: Optional[torch.FloatTensor], layer_past: Optional[Cache] = ..., attention_mask: Optional[torch.FloatTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., cache_position: Optional[torch.LongTensor] = ...) -> Union[Tuple[torch.Tensor, Tuple[torch.Tensor]], Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],]:
        ...
    


class CodeGenMLP(nn.Module):
    def __init__(self, intermediate_size, config) -> None:
        ...
    
    def forward(self, hidden_states: Optional[torch.FloatTensor]) -> torch.FloatTensor:
        ...
    


class CodeGenBlock(nn.Module):
    def __init__(self, config, layer_idx=...) -> None:
        ...
    
    def forward(self, hidden_states: Optional[torch.FloatTensor], layer_past: Optional[Cache] = ..., attention_mask: Optional[torch.FloatTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., cache_position: Optional[torch.LongTensor] = ...) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        ...
    


class CodeGenPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = CodeGenConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_cache_class = ...
    _supports_quantized_cache = ...
    _supports_static_cache = ...
    def __init__(self, *inputs, **kwargs) -> None:
        ...
    


CODEGEN_START_DOCSTRING = ...
CODEGEN_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare CodeGen Model transformer outputting raw hidden-states without any specific head on top.", CODEGEN_START_DOCSTRING)
class CodeGenModel(CodeGenPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CODEGEN_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor]]]] = ..., attention_mask: Optional[torch.FloatTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., cache_position: Optional[torch.LongTensor] = ...) -> Union[Tuple, BaseModelOutputWithPast]:
        ...
    


@add_start_docstrings("""
    The CodeGen Model transformer with a language modeling head on top.
    """, CODEGEN_START_DOCSTRING)
class CodeGenForCausalLM(CodeGenPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CODEGEN_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor]]]] = ..., attention_mask: Optional[torch.FloatTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., cache_position: Optional[torch.LongTensor] = ...) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        ...
    


