"""
This type stub file was generated by pyright.
"""

import torch
from typing import Optional, Tuple, Union
from torch import nn
from transformers.generation import GenerationConfig
from ...generation import GenerationMixin
from ...modeling_outputs import Seq2SeqLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_pop2piano import Pop2PianoConfig

"""PyTorch Pop2Piano model."""
logger = ...
_load_pop2piano_layer_norm = ...
_load_pop2piano_layer_norm = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
POP2PIANO_INPUTS_DOCSTRING = ...
class Pop2PianoLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None:
        """
        Construct a layernorm module in the Pop2Piano style. No bias and no subtraction of mean.
        """
        ...
    
    def forward(self, hidden_states):
        ...
    


if not _load_pop2piano_layer_norm:
    Pop2PianoLayerNorm = ...
class Pop2PianoDenseActDense(nn.Module):
    def __init__(self, config: Pop2PianoConfig) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class Pop2PianoDenseGatedActDense(nn.Module):
    def __init__(self, config: Pop2PianoConfig) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class Pop2PianoLayerFF(nn.Module):
    def __init__(self, config: Pop2PianoConfig) -> None:
        ...
    
    def forward(self, hidden_states):
        ...
    


class Pop2PianoAttention(nn.Module):
    def __init__(self, config: Pop2PianoConfig, has_relative_attention_bias=..., layer_idx: Optional[int] = ...) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def compute_bias(self, query_length, key_length, device=..., cache_position=...): # -> Any:
        """Compute binned relative position bias"""
        ...
    
    def forward(self, hidden_states, mask=..., key_value_states=..., position_bias=..., past_key_value=..., layer_head_mask=..., query_length=..., use_cache=..., output_attentions=..., cache_position=...): # -> tuple[Any, Any | None, Any | Tensor, Any | Tensor] | tuple[Any, Any | None, Any | Tensor]:
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        ...
    


class Pop2PianoLayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=..., layer_idx: Optional[int] = ...) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., position_bias=..., layer_head_mask=..., past_key_value=..., use_cache=..., output_attentions=..., cache_position=...): # -> Any:
        ...
    


class Pop2PianoLayerCrossAttention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = ...) -> None:
        ...
    
    def forward(self, hidden_states, key_value_states, attention_mask=..., position_bias=..., layer_head_mask=..., past_key_value=..., use_cache=..., query_length=..., output_attentions=..., cache_position=...): # -> Any:
        ...
    


class Pop2PianoBlock(nn.Module):
    def __init__(self, config, has_relative_attention_bias=..., layer_idx: Optional[int] = ...) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., position_bias=..., encoder_hidden_states=..., encoder_attention_mask=..., encoder_decoder_position_bias=..., layer_head_mask=..., cross_attn_layer_head_mask=..., past_key_value=..., use_cache=..., output_attentions=..., return_dict=..., cache_position=...): # -> Any:
        ...
    


class Pop2PianoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Pop2PianoConfig
    base_model_prefix = ...
    is_parallelizable = ...
    supports_gradient_checkpointing = ...
    _supports_cache_class = ...
    _supports_static_cache = ...
    _no_split_modules = ...
    _keep_in_fp32_modules = ...


class Pop2PianoStack(Pop2PianoPreTrainedModel):
    def __init__(self, config, embed_tokens=...) -> None:
        ...
    
    def get_input_embeddings(self): # -> None:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    def forward(self, input_ids=..., attention_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., inputs_embeds=..., head_mask=..., cross_attn_head_mask=..., past_key_values=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., cache_position=...):
        ...
    


class Pop2PianoConcatEmbeddingToMel(nn.Module):
    """Embedding Matrix for `composer` tokens."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, feature, index_value, embedding_offset): # -> Tensor:
        ...
    


Pop2Piano_START_DOCSTRING = ...
@add_start_docstrings("""Pop2Piano Model with a `language modeling` head on top.""", Pop2Piano_START_DOCSTRING)
class Pop2PianoForConditionalGeneration(Pop2PianoPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config: Pop2PianoConfig) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def get_encoder(self): # -> Pop2PianoStack:
        ...
    
    def get_decoder(self): # -> Pop2PianoStack:
        ...
    
    def get_mel_conditioner_outputs(self, input_features: torch.FloatTensor, composer: str, generation_config: GenerationConfig, attention_mask: torch.FloatTensor = ...): # -> tuple[FloatTensor, Any | FloatTensor]:
        """
        This method is used to concatenate mel conditioner tokens at the front of the input_features in order to
        control the type of MIDI token generated by the model.

        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                input features extracted from the feature extractor.
            composer (`str`):
                composer token which determines the type of MIDI tokens to be generated.
            generation_config (`~generation.GenerationConfig`):
                The generation is used to get the composer-feature_token pair.
            attention_mask (``, *optional*):
                For batched generation `input_features` are padded to have the same shape across all examples.
                `attention_mask` helps to determine which areas were padded and which were not.
                - 1 for tokens that are **not padded**,
                - 0 for tokens that are **padded**.
        """
        ...
    
    @add_start_docstrings_to_model_forward(POP2PIANO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., decoder_input_ids: Optional[torch.LongTensor] = ..., decoder_attention_mask: Optional[torch.BoolTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., decoder_head_mask: Optional[torch.FloatTensor] = ..., cross_attn_head_mask: Optional[torch.Tensor] = ..., encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = ..., past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., input_features: Optional[torch.FloatTensor] = ..., decoder_inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., cache_position: Optional[torch.LongTensor] = ...) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        Returns:
        """
        ...
    
    @torch.no_grad()
    def generate(self, input_features, attention_mask=..., composer=..., generation_config=..., **kwargs): # -> GenerateOutput | LongTensor:
        """
        Generates token ids for midi outputs.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`. For an overview of generation
        strategies and code examples, check out the [following guide](./generation_strategies).

        </Tip>

        Parameters:
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                This is the featurized version of audio generated by `Pop2PianoFeatureExtractor`.
            attention_mask:
                For batched generation `input_features` are padded to have the same shape across all examples.
                `attention_mask` helps to determine which areas were padded and which were not.
                - 1 for tokens that are **not padded**,
                - 0 for tokens that are **padded**.
            composer (`str`, *optional*, defaults to `"composer1"`):
                This value is passed to `Pop2PianoConcatEmbeddingToMel` to generate different embeddings for each
                `"composer"`. Please make sure that the composet value is present in `composer_to_feature_token` in
                `generation_config`. For an example please see
                https://huggingface.co/sweetcocoa/pop2piano/blob/main/generation_config.json .
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            kwargs:
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.
        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.
                Since Pop2Piano is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:
                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        ...
    
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor): # -> Tensor:
        ...
    


