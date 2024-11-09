"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn

from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_instructblip import InstructBlipConfig, InstructBlipQFormerConfig, InstructBlipVisionConfig

""" PyTorch InstructBLIP model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
@dataclass
class InstructBlipForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`InstructBlipForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """
    loss: Optional[Tuple[torch.FloatTensor]] = ...
    logits: Optional[Tuple[torch.FloatTensor]] = ...
    vision_outputs: Optional[torch.FloatTensor] = ...
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = ...
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = ...
    def to_tuple(self) -> Tuple[Any]:
        ...



class InstructBlipVisionEmbeddings(nn.Module):
    def __init__(self, config: InstructBlipVisionConfig) -> None:
        ...

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        ...



class InstructBlipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        ...



class InstructBlipMLP(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...



class InstructBlipEncoderLayer(nn.Module):
    def __init__(self, config: InstructBlipConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, output_attentions: Optional[bool] = ...) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        ...



class InstructBlipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = InstructBlipConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _keep_in_fp32_modules = ...


INSTRUCTBLIP_START_DOCSTRING = ...
INSTRUCTBLIP_VISION_INPUTS_DOCSTRING = ...
INSTRUCTBLIP_INPUTS_DOCSTRING = ...
class InstructBlipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InstructBlipEncoderLayer`].

    Args:
        config (`InstructBlipConfig`):
            The corresponding vision configuration for the `InstructBlipEncoder`.
    """
    def __init__(self, config: InstructBlipConfig) -> None:
        ...

    def forward(self, inputs_embeds, attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        ...



class InstructBlipVisionModel(InstructBlipPreTrainedModel):
    main_input_name = ...
    config_class = InstructBlipVisionConfig
    def __init__(self, config: InstructBlipVisionConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(INSTRUCTBLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=InstructBlipVisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        ...

    def get_input_embeddings(self): # -> InstructBlipVisionEmbeddings:
        ...



class InstructBlipQFormerMultiHeadAttention(nn.Module):
    def __init__(self, config, is_cross_attention=...) -> None:
        ...

    def save_attn_gradients(self, attn_gradients): # -> None:
        ...

    def get_attn_gradients(self):
        ...

    def save_attention_map(self, attention_map): # -> None:
        ...

    def get_attention_map(self):
        ...

    def transpose_for_scores(self, x):
        ...

    def forward(self, hidden_states, attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., past_key_value=..., output_attentions=...): # -> tuple[Tensor | Any, ...] | tuple[Tensor | tuple[Any | Tensor, Any | Tensor], ...]:
        ...



class InstructBlipQFormerSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...



class InstructBlipQFormerAttention(nn.Module):
    def __init__(self, config, is_cross_attention=...) -> None:
        ...

    def prune_heads(self, heads): # -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...



class InstructBlipQFormerIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...



class InstructBlipQFormerOutput(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...



class InstructBlipQFormerLayer(nn.Module):
    def __init__(self, config, layer_idx) -> None:
        ...

    def forward(self, hidden_states, attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., past_key_value=..., output_attentions=..., query_length=...): # -> Any:
        ...

    def feed_forward_chunk(self, attention_output): # -> Any:
        ...

    def feed_forward_chunk_query(self, attention_output): # -> Any:
        ...



class InstructBlipQFormerEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states, attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., past_key_values=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., query_length=...): # -> tuple[Any, ...] | BaseModelOutputWithPastAndCrossAttentions:
        ...



class InstructBlipQFormerEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""
    def __init__(self, config) -> None:
        ...

    def forward(self, input_ids=..., position_ids=..., query_embeds=..., past_key_values_length=...): # -> Any:
        ...



class InstructBlipQFormerModel(InstructBlipPreTrainedModel):
    """
    Querying Transformer (Q-Former), used in InstructBLIP. Slightly modified from BLIP-2 as it also takes the
    instruction as input.
    """
    def __init__(self, config: InstructBlipQFormerConfig) -> None:
        ...

    def get_input_embeddings(self): # -> Embedding:
        ...

    def set_input_embeddings(self, value): # -> None:
        ...

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, input_shape: Tuple[int], device: torch.device, has_query: bool = ...) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device: (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        ...

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.FloatTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., query_embeds: Optional[torch.Tensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.FloatTensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of:
            shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and
            value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are
            used, the user can optionally input only the last `decoder_input_ids` (those that don't have their past key
            value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
            `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        ...



@add_start_docstrings("""
    InstructBLIP Model for generating text given an image and an optional text prompt. The model consists of a vision
    encoder, Querying Transformer (Q-Former) and a language model.

    One can optionally pass `input_ids` to the model, which serve as a text prompt, to make the language model continue
    the prompt. Otherwise, the language model starts generating text from the [BOS] (beginning-of-sequence) token.
    """, INSTRUCTBLIP_START_DOCSTRING)
class InstructBlipForConditionalGeneration(InstructBlipPreTrainedModel):
    config_class = InstructBlipConfig
    main_input_name = ...
    def __init__(self, config: InstructBlipConfig) -> None:
        ...

    def get_input_embeddings(self):
        ...

    def set_input_embeddings(self, value): # -> None:
        ...

    def set_output_embeddings(self, new_embeddings): # -> None:
        ...

    def get_output_embeddings(self) -> nn.Module:
        ...

    def get_encoder(self):
        ...

    def get_decoder(self):
        ...

    @add_start_docstrings_to_model_forward(INSTRUCTBLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=InstructBlipForConditionalGenerationModelOutput, config_class=InstructBlipVisionConfig)
    def forward(self, pixel_values: torch.FloatTensor, qformer_input_ids: torch.FloatTensor, qformer_attention_mask: Optional[torch.LongTensor] = ..., input_ids: Optional[torch.FloatTensor] = ..., attention_mask: Optional[torch.LongTensor] = ..., decoder_input_ids: Optional[torch.LongTensor] = ..., decoder_attention_mask: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., labels: Optional[torch.LongTensor] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, InstructBlipForConditionalGenerationModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
            1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        >>> processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> prompt = "What is unusual about this image?"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        >>> outputs = model.generate(
        ...     **inputs,
        ...     do_sample=False,
        ...     num_beams=5,
        ...     max_length=256,
        ...     min_length=1,
        ...     top_p=0.9,
        ...     repetition_penalty=1.5,
        ...     length_penalty=1.0,
        ...     temperature=1,
        ... )
        >>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV, which is parked in the middle of a busy city street. This is an unconventional approach to ironing clothes, as it requires the man to balance himself and his ironing equipment on top of the vehicle while navigating through traffic. Additionally, the presence of taxis and other vehicles in the scene further emphasizes the unusual nature of this situation.
        ```"""
        ...

    @torch.no_grad()
    def generate(self, pixel_values: torch.FloatTensor, qformer_input_ids: Optional[torch.LongTensor] = ..., qformer_attention_mask: Optional[torch.LongTensor] = ..., input_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.LongTensor] = ..., **generate_kwargs) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            qformer_input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt to be fed to the Q-Former module.
            qformer_attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices.

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        ...
