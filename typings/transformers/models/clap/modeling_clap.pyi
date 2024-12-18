"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_clap import ClapAudioConfig, ClapConfig, ClapTextConfig

"""PyTorch CLAP model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
def interpolate(hidden_states, ratio):
    """
    Interpolate data in time domain. This is used to compensate the resolution reduction in downsampling of a CNN.

    Args:
        hidden_states (`torch.FloatTensor` of shape (batch_size, time_length, classes_num)):
            Input hidden states
        ratio (`int`):
            The ratio of the length of the output to the length of the input.
    """
    ...

def window_partition(hidden_states, window_size):
    """
    Returns the resized hidden states. The output shape should be `(batch_size * num_windows, window_size, window_size,
    num_channels)`

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch_size, height, width, num_channels)`):
            Input hidden states
        window_size (`int`):
            Window size
    """
    ...

def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    Args:
        windows (`torch.FloatTensor` of shape `(num_windows * batch_size, window_size, window_size, num_channels)`):
            Input windows
        window_size (`int`):
            Window size
        height (`int`):
            Height of the resized audio
        width (`int`):
            Width of the resized audio
    """
    ...

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=...):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    ...

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    ...

@dataclass
class ClapTextModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    text_embeds: Optional[torch.FloatTensor] = ...
    last_hidden_state: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = ...
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = ...


@dataclass
class ClapAudioModelOutput(ModelOutput):
    """
    ClapAudio model output to mimic the output of the original implementation.

    Args:
        audio_embeds (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            The Audio embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """
    audio_embeds: Optional[torch.FloatTensor] = ...
    last_hidden_state: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = ...
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = ...


@dataclass
class ClapOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for audio-text similarity.
        logits_per_audio (`torch.FloatTensor` of shape `(audio_batch_size, text_batch_size)`):
            The scaled dot product scores between `audio_embeds` and `text_embeds`. This represents the audio-text
            similarity scores.
        logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, audio_batch_size)`):
            The scaled dot product scores between `text_embeds` and `audio_embeds`. This represents the text-audio
            similarity scores.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`ClapTextModel`].
        audio_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The audio embeddings obtained by applying the projection layer to the pooled output of [`ClapAudioModel`].
        text_model_output (`BaseModelOutputWithPooling`):
            The output of the [`ClapTextModel`].
        audio_model_output (`BaseModelOutputWithPooling`):
            The output of the [`ClapAudioModel`].
    """
    loss: Optional[torch.FloatTensor] = ...
    logits_per_audio: torch.FloatTensor = ...
    logits_per_text: torch.FloatTensor = ...
    text_embeds: torch.FloatTensor = ...
    audio_embeds: torch.FloatTensor = ...
    text_model_output: BaseModelOutputWithPooling = ...
    audio_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> Tuple[Any]:
        ...
    


class ClapDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks). This is a slightly
    refactored version of the `SwinDropPath` implementation.
    """
    def __init__(self, drop_prob=...) -> None:
        ...
    
    def forward(self, hidden_states):
        ...
    


class ClapAudioAFFBlock(nn.Module):
    r"""
    ATTENTIONAL FEATURE FUSION Block from CLAP, since in CLAP we are always in 2D mode, it is not needed to implement
    the 1D version.
    """
    def __init__(self, config: ClapAudioConfig) -> None:
        ...
    
    def forward(self, hidden_states, residual):
        ...
    


class ClapAudioPatchEmbed(nn.Module):
    """
    This module converts the hidden states reshaped as an image to patch embeddings ready to be passed to the
    Transformer block.
    """
    def __init__(self, config: ClapAudioConfig) -> None:
        ...
    
    def forward(self, hidden_states, is_longer_idx=...): # -> Any:
        ...
    


class ClapAudioSelfAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size) -> None:
        ...
    
    def transpose_for_scores(self, x):
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...
    


class ClapAudioSelfOutput(nn.Module):
    def __init__(self, config, dim) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...
    


class ClapAudioAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...
    


class ClapAudioIntermediate(nn.Module):
    def __init__(self, config, dim) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class ClapAudioOutput(nn.Module):
    def __init__(self, config, dim) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class ClapAudioLayer(nn.Module):
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=...) -> None:
        ...
    
    def set_shift_and_window_size(self, input_resolution): # -> None:
        ...
    
    def get_attn_mask(self, height, width, dtype, device): # -> Tensor | None:
        ...
    
    def maybe_pad(self, hidden_states, height, width): # -> tuple[Tensor, tuple[Literal[0], Literal[0], Literal[0], Any, Literal[0], Any]]:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int], head_mask: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., always_partition: Optional[bool] = ...) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
    


class ClapAudioStage(nn.Module):
    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int], head_mask: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., always_partition: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...
    


class ClapAudioPatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """
    def __init__(self, input_resolution: Tuple[int], dim: int, norm_layer: nn.Module = ...) -> None:
        ...
    
    def maybe_pad(self, input_feature, height, width): # -> Tensor:
        ...
    
    def forward(self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]) -> torch.Tensor:
        ...
    


class ClapAudioEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def reshape_mel2img(self, normalized_input_features):
        """
        The input is 4 normalized log mel spectrograms. It is reshape to the common shape of images. Each channel
        should represent 1 of the 4 crops of the spectrogram. For more details, refer to the [`ClapFeatureExtractor`].
        """
        ...
    
    def forward(self, input_features, is_longer: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., output_hidden_states_before_downsampling: Optional[bool] = ..., always_partition: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, ClapAudioModelOutput]:
        ...
    


CLAP_START_DOCSTRING = ...
CLAP_TEXT_INPUTS_DOCSTRING = ...
CLAP_AUDIO_INPUTS_DOCSTRING = ...
CLAP_INPUTS_DOCSTRING = ...
class ClapProjectionLayer(nn.Module):
    def __init__(self, config: Union[ClapAudioConfig, ClapTextConfig]) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class ClapTextEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=..., past_key_values_length=...): # -> Any:
        ...
    
    def create_position_ids_from_inputs_embeds(self, inputs_embeds): # -> Tensor:
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        ...
    


class ClapTextSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None:
        ...
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...
    


class ClapTextSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...
    


CLAP_TEXT_SELF_ATTENTION_CLASSES = ...
class ClapTextAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...
    


class ClapTextIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class ClapTextOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...
    


class ClapTextLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...
    
    def feed_forward_chunk(self, attention_output): # -> Any:
        ...
    


class ClapTextEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        ...
    


class ClapTextPooler(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class ClapPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ClapConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...


class ClapAudioModel(ClapPreTrainedModel):
    config_class = ClapAudioConfig
    main_input_name = ...
    def __init__(self, config: ClapAudioConfig) -> None:
        ...
    
    def get_input_embeddings(self) -> nn.Module:
        ...
    
    @add_start_docstrings_to_model_forward(CLAP_AUDIO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=ClapAudioConfig)
    def forward(self, input_features: Optional[torch.FloatTensor] = ..., is_longer: Optional[torch.BoolTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoProcessor, ClapAudioModel

        >>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused")
        >>> processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")

        >>> inputs = processor(audios=audio_sample, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        ```"""
        ...
    


class ClapTextModel(ClapPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """
    config_class = ClapTextConfig
    def __init__(self, config, add_pooling_layer=...) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    def forward(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., token_type_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., encoder_hidden_states: Optional[torch.Tensor] = ..., encoder_attention_mask: Optional[torch.Tensor] = ..., past_key_values: Optional[List[torch.FloatTensor]] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        ...
    


@add_start_docstrings(CLAP_START_DOCSTRING)
class ClapModel(ClapPreTrainedModel):
    config_class = ClapConfig
    def __init__(self, config: ClapConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CLAP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`ClapTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ClapModel

        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        >>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

        >>> inputs = tokenizer(["the sound of a cat", "the sound of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        ...
    
    @add_start_docstrings_to_model_forward(CLAP_AUDIO_INPUTS_DOCSTRING)
    def get_audio_features(self, input_features: Optional[torch.Tensor] = ..., is_longer: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> torch.FloatTensor:
        r"""
        Returns:
            audio_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The audio embeddings obtained by
            applying the projection layer to the pooled output of [`ClapAudioModel`].

        Examples:

        ```python
        >>> from transformers import AutoFeatureExtractor, ClapModel
        >>> import torch

        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
        >>> random_audio = torch.rand((16_000))
        >>> inputs = feature_extractor(random_audio, return_tensors="pt")
        >>> audio_features = model.get_audio_features(**inputs)
        ```"""
        ...
    
    @add_start_docstrings_to_model_forward(CLAP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ClapOutput, config_class=ClapConfig)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., input_features: Optional[torch.FloatTensor] = ..., is_longer: Optional[torch.BoolTensor] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., return_loss: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, ClapOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoProcessor, ClapModel

        >>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        >>> processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

        >>> input_text = ["Sound of a dog", "Sound of vaccum cleaner"]

        >>> inputs = processor(text=input_text, audios=audio_sample, return_tensors="pt", padding=True)

        >>> outputs = model(**inputs)
        >>> logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
        >>> probs = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities
        ```"""
        ...
    


@add_start_docstrings("""
    CLAP Text Model with a projection layer on top (a linear layer on top of the pooled output).
    """, CLAP_START_DOCSTRING)
class ClapTextModelWithProjection(ClapPreTrainedModel):
    config_class = ClapTextConfig
    def __init__(self, config: ClapTextConfig) -> None:
        ...
    
    def get_input_embeddings(self) -> nn.Module:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CLAP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ClapTextModelOutput, config_class=ClapTextConfig)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, ClapTextModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ClapTextModelWithProjection

        >>> model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused")
        >>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

        >>> inputs = tokenizer(["a sound of a cat", "a sound of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> text_embeds = outputs.text_embeds
        ```"""
        ...
    


@add_start_docstrings("""
    CLAP Audio Model with a projection layer on top (a linear layer on top of the pooled output).
    """, CLAP_START_DOCSTRING)
class ClapAudioModelWithProjection(ClapPreTrainedModel):
    config_class = ClapAudioConfig
    main_input_name = ...
    def __init__(self, config: ClapAudioConfig) -> None:
        ...
    
    def get_input_embeddings(self) -> nn.Module:
        ...
    
    @add_start_docstrings_to_model_forward(CLAP_AUDIO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ClapAudioModelOutput, config_class=ClapAudioConfig)
    def forward(self, input_features: Optional[torch.FloatTensor] = ..., is_longer: Optional[torch.BoolTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, ClapAudioModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import ClapAudioModelWithProjection, ClapProcessor

        >>> model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused")
        >>> processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

        >>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> inputs = processor(audios=audio_sample, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> audio_embeds = outputs.audio_embeds
        ```"""
        ...
    


