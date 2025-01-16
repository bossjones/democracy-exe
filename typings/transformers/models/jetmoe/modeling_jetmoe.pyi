"""
This type stub file was generated by pyright.
"""

import torch
from typing import List, Optional, Tuple, Union
from torch import nn
from ...cache_utils import Cache
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, is_flash_attn_2_available, replace_return_docstrings
from .configuration_jetmoe import JetMoeConfig

"""PyTorch JetMoe model."""
if is_flash_attn_2_available():
    ...
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
def load_balancing_loss_func(gate_logits: torch.Tensor, num_experts: torch.Tensor = ..., top_k=..., attention_mask: Optional[torch.Tensor] = ...) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    ...

class JetMoeParallelExperts(nn.Module):
    def __init__(self, num_experts: int, input_size: int, output_size: int) -> None:
        """
        Initialize the JetMoeParallelExperts module.
        The experts weights are stored in [num_experts, output_size, input_size] format. Such that it's comptible with
        many MoE libraries, such as [Megablock](https://github.com/databricks/megablocks) and
        [ScatterMoE](https://github.com/shawntan/scattermoe), as well as the
        [MoE kernel](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py)
        used in vllm.

        Args:
            num_experts (int):
                Number of experts.
            input_size (int):
                Size of the input.
            output_size (int):
                Size of the output.
        """
        ...
    
    def forward(self, inputs, expert_size): # -> Tensor:
        """
        Forward pass of the JetMoeParallelExperts module.

        Args:
            inputs (Tensor):
                Input tensor.
            expert_size:
                Expert size information.

        Returns:
            Tensor: Output tensor.
        """
        ...
    


class JetMoeTopKGating(nn.Module):
    def __init__(self, input_size: int, num_experts: int, top_k: int) -> None:
        """
        Initialize the top-k gating mechanism.

        Args:
            input_size (`int`):
                Size of the input.
            num_experts (`int`):
                Number of experts.
            top_k (`int`):
                Number of top experts to select.
        """
        ...
    
    def forward(self, hidden_states): # -> tuple[Any, Any, Tensor, List[Any], Any]:
        ...
    


class JetMoeMoE(nn.Module):
    """
    A Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.

    Args:
        config:
            Configuration object with model hyperparameters.
    """
    def __init__(self, config: JetMoeConfig) -> None:
        ...
    
    def forward(self, layer_input): # -> tuple[Tensor, Any]:
        """
        Forward pass of the mixture of experts layer.

        Args:
            layer_input (Tensor):
                Input tensor.

        Returns:
            Tensor:
                Output tensor.
            Tensor:
                Router logits.
        """
        ...
    


class JetMoeMoA(nn.Module):
    """
    A Sparsely gated mixture of attention layer with pairs of query- and output-projections as experts.

    Args:
        config:
            Configuration object with model hyperparameters.
    """
    def __init__(self, config: JetMoeConfig) -> None:
        ...
    
    def map(self, layer_input): # -> tuple[Tensor, Any, tuple[Any, Any, Any, Any]]:
        """
        Map inputs to attention experts according to routing decision and compute query projection inside each experts.
        """
        ...
    
    def reduce(self, layer_input, topo_info): # -> Tensor:
        """
        Compute output projection inside each attention experts and merge the outputs of different experts.
        """
        ...
    
    def forward(self, layer_input):
        ...
    


class JetMoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None:
        """
        JetMoeRMSNorm is equivalent to T5LayerNorm
        """
        ...
    
    def forward(self, hidden_states):
        ...
    
    def extra_repr(self): # -> str:
        ...
    


class JetMoeRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=..., base=..., device=...) -> None:
        ...
    
    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=...): # -> tuple[Tensor, Tensor]:
        ...
    


def rotate_half(x): # -> Tensor:
    """Rotates half the hidden dims of the input."""
    ...

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=..., unsqueeze_dim=...): # -> tuple[Any, Any]:
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
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

class JetMoeAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    """
    def __init__(self, config: JetMoeConfig, layer_idx: Optional[int] = ...) -> None:
        """
        Initialize the JetMoeAttention module.

        Args:
            config:
                Configuration object with model hyperparameters.
            layer_idx:
                Index of the layer in the model.
        """
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., past_key_value: Optional[Cache] = ..., output_attentions: bool = ..., use_cache: bool = ..., cache_position: Optional[torch.LongTensor] = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        ...
    


class JetMoeSdpaAttention(JetMoeAttention):
    """
    JetMoe attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `JetMoeAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., past_key_value: Optional[Cache] = ..., output_attentions: bool = ..., use_cache: bool = ..., cache_position: Optional[torch.LongTensor] = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]:
        ...
    


class JetMoeFlashAttention2(JetMoeAttention):
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def forward(self, hidden_states: Optional[torch.FloatTensor], attention_mask: Optional[torch.FloatTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., past_key_value: Optional[Cache] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., cache_position: Optional[torch.LongTensor] = ...) -> Union[Tuple[torch.Tensor, Tuple[torch.Tensor]], Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],]:
        """
        Forward pass of the JetMoeAttention module.

        Args:
            hidden_states (Optional[torch.FloatTensor]): Input hidden states.
            attention_mask (Optional[torch.FloatTensor]): Attention mask.
            layer_past (Optional[Tuple[torch.Tensor]]): Past layer state.
            use_cache (Optional[bool]): Whether to use cached states.
            output_attentions (Optional[bool]): Whether to output attention weights.
            cache_position (Optional[torch.LongTensor]): Position of the cache.

        Returns:
            Union[Tuple[torch.Tensor, Tuple[torch.Tensor]], Optional[Tuple[...]]]: Tuple containing outputs.
        """
        ...
    


JETMOE_ATTENTION_CLASSES = ...
class JetMoeBlock(nn.Module):
    def __init__(self, config: JetMoeConfig, layer_idx: Optional[int] = ...) -> None:
        """
        Initialize the JetMoeBlock module.

        Args:
            config:
                Configuration object with model hyperparameters.
        """
        ...
    
    def forward(self, hidden_states: Optional[torch.FloatTensor], position_ids: Optional[torch.LongTensor] = ..., past_key_value: Optional[Tuple[torch.Tensor]] = ..., attention_mask: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_router_logits: Optional[bool] = ..., use_cache: Optional[bool] = ..., cache_position: Optional[torch.LongTensor] = ...) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        ...
    


class JetMoePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = JetMoeConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn_2 = ...
    _supports_sdpa = ...
    _supports_cache_class = ...


JETMOE_START_DOCSTRING = ...
JETMOE_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare JetMoe Model outputting raw hidden-states without any specific head on top.", JETMOE_START_DOCSTRING)
class JetMoeModel(JetMoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`JetMoeBlock`]

    Args:
        config:
            JetMoeConfig
    """
    def __init__(self, config: JetMoeConfig) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding | Module:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(JETMOE_INPUTS_DOCSTRING)
    def forward(self, input_ids: torch.LongTensor = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., output_router_logits: Optional[bool] = ..., return_dict: Optional[bool] = ..., cache_position: Optional[torch.LongTensor] = ...) -> Union[Tuple, MoeModelOutputWithPast]:
        ...
    


class JetMoeForCausalLM(JetMoePreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding | Module:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    def set_decoder(self, decoder): # -> None:
        ...
    
    def get_decoder(self): # -> JetMoeModel:
        ...
    
    @add_start_docstrings_to_model_forward(JETMOE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., past_key_values: Optional[List[torch.FloatTensor]] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., output_router_logits: Optional[bool] = ..., return_dict: Optional[bool] = ..., cache_position: Optional[torch.LongTensor] = ...) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        ...
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=..., attention_mask=..., inputs_embeds=..., cache_position=..., output_router_logits=..., position_ids=..., use_cache=..., **kwargs): # -> dict[str, Any]:
        ...
    


@add_start_docstrings("""
    The JetMoe Model transformer with a sequence classification head on top (linear layer).

    [`JetMoeForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """, JETMOE_START_DOCSTRING)
class JetMoeForSequenceClassification(JetMoePreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding | Module:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(JETMOE_INPUTS_DOCSTRING)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


