"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_blip_2 import Blip2Config, Blip2QFormerConfig, Blip2VisionConfig

"""PyTorch BLIP-2 model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
@dataclass
class Blip2ForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`Blip2ForConditionalGeneration`].

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
    


class Blip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Blip2VisionConfig) -> None:
        ...
    
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        ...
    
    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding: bool = ...) -> torch.Tensor:
        ...
    


class Blip2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        ...
    


class Blip2MLP(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class Blip2EncoderLayer(nn.Module):
    def __init__(self, config: Blip2Config) -> None:
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
    


class Blip2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Blip2Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _keep_in_fp32_modules = ...


BLIP_2_START_DOCSTRING = ...
BLIP_2_VISION_INPUTS_DOCSTRING = ...
BLIP_2_TEXT_INPUTS_DOCSTRING = ...
BLIP_2_INPUTS_DOCSTRING = ...
class Blip2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Blip2EncoderLayer`].

    Args:
        config (`Blip2Config`):
            The corresponding vision configuration for the `Blip2Encoder`.
    """
    def __init__(self, config: Blip2Config) -> None:
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
    


class Blip2VisionModel(Blip2PreTrainedModel):
    main_input_name = ...
    config_class = Blip2VisionConfig
    def __init__(self, config: Blip2VisionConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(BLIP_2_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Blip2VisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., interpolate_pos_encoding: bool = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        ...
    
    def get_input_embeddings(self): # -> Blip2VisionEmbeddings:
        ...
    


class Blip2QFormerMultiHeadAttention(nn.Module):
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
    


class Blip2QFormerSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...
    


class Blip2QFormerAttention(nn.Module):
    def __init__(self, config, is_cross_attention=...) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...
    


class Blip2QFormerIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class Blip2QFormerOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...
    


class Blip2QFormerLayer(nn.Module):
    def __init__(self, config, layer_idx) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., past_key_value=..., output_attentions=..., query_length=...): # -> Any:
        ...
    
    def feed_forward_chunk(self, attention_output): # -> Any:
        ...
    
    def feed_forward_chunk_query(self, attention_output): # -> Any:
        ...
    


class Blip2QFormerEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., past_key_values=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., query_length=...): # -> tuple[Any, ...] | BaseModelOutputWithPastAndCrossAttentions:
        ...
    


class Blip2QFormerModel(Blip2PreTrainedModel):
    """
    Querying Transformer (Q-Former), used in BLIP-2.
    """
    def __init__(self, config: Blip2QFormerConfig) -> None:
        ...
    
    def get_input_embeddings(self): # -> Any:
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
            device (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        ...
    
    def forward(self, query_embeds: torch.FloatTensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
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
        use_cache (`bool`, `optional`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        ...
    


@add_start_docstrings("""
    BLIP-2 Model for generating text and image features. The model consists of a vision encoder, Querying Transformer
    (Q-Former) and a language model.
    """, BLIP_2_START_DOCSTRING)
class Blip2Model(Blip2PreTrainedModel):
    config_class = Blip2Config
    main_input_name = ...
    def __init__(self, config: Blip2Config) -> None:
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
    
    @add_start_docstrings_to_model_forward(BLIP_2_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., decoder_input_ids: Optional[torch.Tensor] = ..., decoder_attention_mask: Optional[torch.Tensor] = ..., labels: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...):
        r"""
        Returns:
            text_outputs (`CausalLMOutputWithPast`, or `tuple(torch.FloatTensor)` if `return_dict=False`):
                The language model outputs. If `return_dict=True`, the output is a [`CausalLMOutputWithPast`] that
                contains the language model logits, the past key values and the hidden states if
                `output_hidden_states=True`.
        Examples:
        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, Blip2Model

        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")

        >>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> inputs = tokenizer(["a photo of a cat"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        ...
    
    @add_start_docstrings_to_model_forward(BLIP_2_VISION_INPUTS_DOCSTRING)
    def get_image_features(self, pixel_values: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., interpolate_pos_encoding: bool = ...): # -> Any:
        r"""
        Returns:
            vision_outputs (`BaseModelOutputWithPooling` or tuple of `torch.FloatTensor`):
                The vision model outputs. If `return_dict=True`, the output is a [`BaseModelOutputWithPooling`] that
                contains the image features, the pooled image features and the hidden states if
                `output_hidden_states=True`.
        Examples:
        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Blip2Model

        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")

        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> image_outputs = model.get_image_features(**inputs)
        ```"""
        ...
    
    @add_start_docstrings_to_model_forward(BLIP_2_INPUTS_DOCSTRING)
    def get_qformer_features(self, pixel_values: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., interpolate_pos_encoding: bool = ...): # -> Any:
        r"""
        Returns:
            vision_outputs (`BaseModelOutputWithPooling` or tuple of `torch.FloatTensor`):
                The vision model outputs. If `return_dict=True`, the output is a [`BaseModelOutputWithPooling`] that
                contains the image features, the pooled image features and the hidden states if
                `output_hidden_states=True`.
        Examples:
        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2Model

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> qformer_outputs = model.get_qformer_features(**inputs)
        ```"""
        ...
    
    @add_start_docstrings_to_model_forward(BLIP_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Blip2ForConditionalGenerationModelOutput, config_class=Blip2VisionConfig)
    def forward(self, pixel_values: torch.FloatTensor, input_ids: torch.FloatTensor, attention_mask: Optional[torch.LongTensor] = ..., decoder_input_ids: Optional[torch.LongTensor] = ..., decoder_attention_mask: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., labels: Optional[torch.LongTensor] = ..., return_dict: Optional[bool] = ..., interpolate_pos_encoding: bool = ...) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2Model
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> prompt = "Question: how many cats are there? Answer:"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

        >>> outputs = model(**inputs)
        ```"""
        ...
    


@add_start_docstrings("""
    BLIP-2 Model for generating text given an image and an optional text prompt. The model consists of a vision
    encoder, Querying Transformer (Q-Former) and a language model.

    One can optionally pass `input_ids` to the model, which serve as a text prompt, to make the language model continue
    the prompt. Otherwise, the language model starts generating text from the [BOS] (beginning-of-sequence) token.

    <Tip>

    Note that Flan-T5 checkpoints cannot be cast to float16. They are pre-trained using bfloat16.

    </Tip>
    """, BLIP_2_START_DOCSTRING)
class Blip2ForConditionalGeneration(Blip2PreTrainedModel):
    config_class = Blip2Config
    main_input_name = ...
    def __init__(self, config: Blip2Config) -> None:
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
    
    @add_start_docstrings_to_model_forward(BLIP_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Blip2ForConditionalGenerationModelOutput, config_class=Blip2VisionConfig)
    def forward(self, pixel_values: torch.FloatTensor, input_ids: torch.FloatTensor, attention_mask: Optional[torch.LongTensor] = ..., decoder_input_ids: Optional[torch.LongTensor] = ..., decoder_attention_mask: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., labels: Optional[torch.LongTensor] = ..., return_dict: Optional[bool] = ..., interpolate_pos_encoding: bool = ...) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        r"""
        Returns:

        Examples:

        Prepare processor, model and image input

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2ForConditionalGeneration
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2ForConditionalGeneration.from_pretrained(
        ...     "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
        ... )  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        ```

        Image captioning (without providing a text prompt):

        ```python
        >>> inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

        >>> generated_ids = model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        two cats laying on a couch
        ```

        Visual question answering (prompt = question):

        ```python
        >>> prompt = "Question: how many cats are there? Answer:"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

        >>> generated_ids = model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        two
        ```

        Note that int8 inference is also supported through [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).
        This greatly reduces the amount of memory used by the model while maintaining the same performance.

        ```python
        >>> model = Blip2ForConditionalGeneration.from_pretrained(
        ...     "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.bfloat16
        ... )  # doctest: +IGNORE_RESULT

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)

        >>> generated_ids = model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        two
        ```"""
        ...
    
    @torch.no_grad()
    def generate(self, pixel_values: torch.FloatTensor, input_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.LongTensor] = ..., interpolate_pos_encoding: bool = ..., **generate_kwargs) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        ...
    


