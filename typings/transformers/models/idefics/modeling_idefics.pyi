"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from ... import PreTrainedModel
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PretrainedConfig
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_idefics import IdeficsConfig

""" PyTorch Idefics model."""
logger = ...
_CONFIG_FOR_DOC = ...
@dataclass
class IdeficsBaseModelOutputWithPast(ModelOutput):
    """
    Base class for Idefics model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """
    last_hidden_state: torch.FloatTensor = ...
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class IdeficsCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Idefics causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """
    loss: Optional[torch.FloatTensor] = ...
    logits: torch.FloatTensor = ...
    past_key_values: Optional[List[torch.FloatTensor]] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...


def expand_inputs_for_generation(input_ids, expand_size=..., is_encoder_decoder=..., attention_mask=..., encoder_outputs=..., **model_kwargs): # -> tuple[Any, dict[str, Any]]:
    ...

def prepare_inputs_for_generation(input_ids, past_key_values=..., **kwargs): # -> dict[str, Any]:
    ...

def freeze_model(model, module_exceptions=...):
    ...

class IdeficsDecoupledEmbedding(nn.Embedding):
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the embeddings. In practise, the
    regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `num_additional_embeddings` > 0,
    then it will create `num_additional_embeddings` additional parameters that are always trained. If
    `num_additional_embeddings=0`, then the module defaults back to the regular behavior of `nn.Embedding`.
    """
    def __init__(self, num_embeddings, num_additional_embeddings, embedding_dim, partially_freeze: Optional[bool] = ..., device=..., dtype=..., padding_idx=..., **kwargs) -> None:
        """
        Args:
            num_embeddings (`int`):
                Size of the dictionary of embeddings
            num_additional_embeddings (`int`):
                Number of additional embeddings. Only useful when you `partially_freeze=True`.
            embedding_dim (`int`):
                The size of each embedding vector
            partially_freeze: (`bool`, *optional*, defaults to `False`):
                If `True`, the regular `weight` will be frozen. `additional_weight` is never frozen.
            padding_idx (`int`, *optional*):
                The padding index (needs to be less than num_embeddings)

        Note: there are a lot of other parameters to initialize a standard `nn.Embedding` such as `padding_idx`,
        `max_norm` or `norm_type`. We are not supporting these.
        """
        ...

    def forward(self, input_ids): # -> Tensor:
        """
        we have 2 embeddings, with different indices - one pretrained self.weight and another
        self.additional_embedding.weight that is being trained.

        in order to make a lookup of the input ids, we:
        1. find out the indices of the entries belonging to the 2nd embedding
        2. extract those values while subtracting the size of the first embedding (num_embeddings), since the 2nd
           embedding starts from 0 and not num_embeddings
        3. perform the 2nd embedding lookup
        4. now we handle the 1st embedding, we overwrite indices belonging to the 2nd embedding with a padding index
        5. perform the 1st embedding lookup
        6. now we overwrite the values in the 1st embedding lookup with the values of the 2nd embedding lookup

        note: for the 1st embedding lookup we could have looked up only the low indices and not do the padding, but
        then we have to create a new tensor and populate it with 2 tensors that are spread out across various indices -
        i.e. not a simple concat - I haven't benchmarked the complex case if it's any faster, given that seqlens are
        usually relatively short it's probably not faster or if faster not by much - but might be a good idea to
        measure.

        """
        ...

    def extra_repr(self) -> str:
        ...



class IdeficsDecoupledLinear(nn.Linear):
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the parameters. In practise, the
    regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `out_additional_features` > 0,
    then it will create `out_additional_features * in_features` additional parameters that are always trained. If
    `out_additional_features=0`, then the module defaults back to the regular behavior of `nn.Linear`.
    """
    def __init__(self, in_features: int, out_features: int, out_additional_features: int = ..., bias: bool = ..., partially_freeze: bool = ..., device=..., dtype=...) -> None:
        """
        out_additional_features: int. Number of additional trainable dimensions. Only makes sense when
        `partially_freeze=True`. partially_freeze: bool. If True, the regular `weight` will be frozen and extra
        parameters (if any) will be trainable. If False, default to the regular behavior of nn.Linear.
        """
        ...

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ...

    def extra_repr(self) -> str:
        """Overwriting `nn.Linear.extra_repr` to include new parameters."""
        ...



class IdeficsRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None:
        """
        IdeficsRMSNorm is equivalent to T5LayerNorm
        """
        ...

    def forward(self, hidden_states):
        ...



class IdeficsEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=..., base=..., device=...) -> None:
        ...

    def forward(self, x, seq_len=...): # -> tuple[Any, Any]:
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

class IdeficsMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str) -> None:
        ...

    def forward(self, x): # -> Any:
        ...



class IdeficsAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = ..., is_cross_attention: bool = ..., config: PretrainedConfig = ..., qk_layer_norms: bool = ...) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., past_key_value: Optional[Tuple[torch.Tensor]] = ..., output_attentions: bool = ..., use_cache: bool = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        ...



class IdeficsDecoderLayer(nn.Module):
    def __init__(self, config: IdeficsConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., past_key_value: Optional[Tuple[torch.Tensor]] = ..., output_attentions: Optional[bool] = ..., use_cache: Optional[bool] = ...) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        ...



class IdeficsGatedCrossAttentionLayer(nn.Module):
    def __init__(self, config: IdeficsConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., image_hidden_states: Optional[torch.Tensor] = ..., image_attention_mask: Optional[torch.Tensor] = ..., cross_attention_gate: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., use_cache: Optional[bool] = ..., past_key_value: Optional[Tuple[torch.Tensor]] = ...) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            image_attention_mask (`torch.FloatTensor`, *optional*): image attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            cross_attention_gate (`torch.FloatTensor`, *optional*):
                gate of size `(batch, seq_len)` used to zero-out cross-attention output for tokens attending no images.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        ...



LLAMA_START_DOCSTRING = ...
@add_start_docstrings("The bare LLaMA Model outputting raw hidden-states without any specific head on top.", LLAMA_START_DOCSTRING)
class IdeficsPreTrainedModel(PreTrainedModel):
    config_class = IdeficsConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_sdpa = ...


LLAMA_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare LLaMA Model outputting raw hidden-states without any specific head on top.", LLAMA_START_DOCSTRING)
class IdeficsModel(IdeficsPreTrainedModel):
    """
    Transformer decoder consisting of `config.num_hidden_layers` layers. Each layer is a [`IdeficsDecoderLayer`]

    Args:
        config: IdeficsConfig
    """
    def __init__(self, config: IdeficsConfig) -> None:
        ...

    def freeze_relevant_params(self, config=...): # -> None:
        ...

    def freeze_text_layers(self, module_exceptions=...): # -> None:
        ...

    def freeze_vision_layers(self, module_exceptions=...): # -> None:
        ...

    def get_input_embeddings(self): # -> IdeficsDecoupledEmbedding | Module:
        ...

    def set_input_embeddings(self, value): # -> None:
        ...

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(self, input_ids: torch.LongTensor = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., past_key_values: Optional[List[torch.FloatTensor]] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., pixel_values: Optional[torch.FloatTensor] = ..., image_encoder_embeddings: Optional[torch.FloatTensor] = ..., perceiver_embeddings: Optional[torch.FloatTensor] = ..., image_attention_mask: Optional[torch.Tensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., interpolate_pos_encoding: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, IdeficsBaseModelOutputWithPast]:
        ...



class IdeficsForVisionText2Text(IdeficsPreTrainedModel):
    _keys_to_ignore_on_load_missing = ...
    _tied_weights_keys = ...
    def __init__(self, config, vision_model=...) -> None:
        ...

    def get_input_embeddings(self): # -> IdeficsDecoupledEmbedding | Module:
        ...

    def set_input_embeddings(self, value): # -> None:
        ...

    def get_output_embeddings(self): # -> IdeficsDecoupledLinear:
        ...

    def set_output_embeddings(self, new_embeddings): # -> None:
        ...

    def set_decoder(self, decoder): # -> None:
        ...

    def get_decoder(self): # -> IdeficsModel:
        ...

    def tie_weights(self): # -> None:
        """
        Overwrite `transformers.modeling_utils.PreTrainedModel.tie_weights` to handle the case of
        IdeficsDecoupledLinear and IdeficsDecoupledEmbedding.
        """
        ...

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=IdeficsCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., past_key_values: Optional[List[torch.FloatTensor]] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., pixel_values: Optional[torch.FloatTensor] = ..., image_encoder_embeddings: Optional[torch.FloatTensor] = ..., perceiver_embeddings: Optional[torch.FloatTensor] = ..., image_attention_mask: Optional[torch.Tensor] = ..., labels: Optional[torch.LongTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., interpolate_pos_encoding: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, IdeficsCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoProcessor, IdeficsForVisionText2Text

        >>> model = IdeficsForVisionText2Text.from_pretrained("HuggingFaceM4/idefics-9b")
        >>> processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics-9b")

        >>> dogs_image_url_1 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_nlvr2/raw/main/image1.jpeg"
        >>> dogs_image_url_2 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_nlvr2/raw/main/image2.jpeg"

        >>> prompts = [
        ...     [
        ...         "User:",
        ...         dogs_image_url_1,
        ...         "Describe this image.\nAssistant: An image of two dogs.\n",
        ...         "User:",
        ...         dogs_image_url_2,
        ...         "Describe this image.\nAssistant:",
        ...     ]
        ... ]
        >>> inputs = processor(prompts, return_tensors="pt")
        >>> generate_ids = model.generate(**inputs, max_new_tokens=6)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True)
        ```"""
        ...

    def prepare_inputs_for_generation(self, input_ids, past=..., **kwargs): # -> dict[str, Any]:
        ...
