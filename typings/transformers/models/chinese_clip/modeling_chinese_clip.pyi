"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_chinese_clip import ChineseCLIPConfig, ChineseCLIPTextConfig, ChineseCLIPVisionConfig

"""PyTorch Chinese-CLIP model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    ...

def chinese_clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    ...

@dataclass
class ChineseCLIPOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            [`ChineseCLIPTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`ChineseCLIPVisionModel`].
        text_model_output(`BaseModelOutputWithPoolingAndCrossAttentions`):
            The output of the [`ChineseCLIPTextModel`].
        vision_model_output(`BaseModelOutputWithPoolingAndCrossAttentions`):
            The output of the [`ChineseCLIPVisionModel`].
    """
    loss: Optional[torch.FloatTensor] = ...
    logits_per_image: torch.FloatTensor = ...
    logits_per_text: torch.FloatTensor = ...
    text_embeds: torch.FloatTensor = ...
    image_embeds: torch.FloatTensor = ...
    text_model_output: BaseModelOutputWithPoolingAndCrossAttentions = ...
    vision_model_output: BaseModelOutputWithPoolingAndCrossAttentions = ...
    def to_tuple(self) -> Tuple[Any]:
        ...
    


class ChineseCLIPTextEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., past_key_values_length: int = ...) -> torch.Tensor:
        ...
    


class ChineseCLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: ChineseCLIPVisionConfig) -> None:
        ...
    
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """
        ...
    
    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=...) -> torch.Tensor:
        ...
    


class ChineseCLIPTextSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None:
        ...
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...
    


class ChineseCLIPTextSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...
    


CHINESE_CLIP_TEXT_SELF_ATTENTION_CLASSES = ...
class ChineseCLIPTextAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...
    


class ChineseCLIPVisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        ...
    


class ChineseCLIPTextIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class ChineseCLIPTextOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...
    


class ChineseCLIPVisionMLP(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class ChineseCLIPTextLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...
    
    def feed_forward_chunk(self, attention_output): # -> Any:
        ...
    


class ChineseCLIPVisionLayer(nn.Module):
    def __init__(self, config: ChineseCLIPConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, output_attentions: Optional[bool] = ...) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        ...
    


class ChineseCLIPTextPooler(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class ChineseCLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ChineseCLIPConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...


CHINESE_CLIP_START_DOCSTRING = ...
CHINESE_CLIP_TEXT_INPUTS_DOCSTRING = ...
CHINESE_CLIP_VISION_INPUTS_DOCSTRING = ...
CHINESE_CLIP_INPUTS_DOCSTRING = ...
class ChineseCLIPTextEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        ...
    


class ChineseCLIPVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`ChineseCLIPVisionEncoderLayer`].

    Args:
        config: ChineseCLIPConfig
    """
    def __init__(self, config: ChineseCLIPConfig) -> None:
        ...
    
    def forward(self, inputs_embeds, output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
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
    


class ChineseCLIPVisionTransformer(nn.Module):
    def __init__(self, config: ChineseCLIPVisionConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CHINESE_CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=ChineseCLIPVisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., interpolate_pos_encoding: bool = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        ...
    


@add_start_docstrings("The text model from CHINESE_CLIP without any head or projection on top.", CHINESE_CLIP_START_DOCSTRING)
class ChineseCLIPTextModel(ChineseCLIPPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """
    config_class = ChineseCLIPTextConfig
    _no_split_modules = ...
    def __init__(self, config, add_pooling_layer=...) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CHINESE_CLIP_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
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
    


@add_start_docstrings("""The vision model from CHINESE_CLIP without any head or projection on top.""", CHINESE_CLIP_START_DOCSTRING)
class ChineseCLIPVisionModel(ChineseCLIPPreTrainedModel):
    config_class = ChineseCLIPVisionConfig
    main_input_name = ...
    _no_split_modules = ...
    def __init__(self, config: ChineseCLIPVisionConfig) -> None:
        ...
    
    def get_input_embeddings(self) -> nn.Module:
        ...
    
    @add_start_docstrings_to_model_forward(CHINESE_CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=ChineseCLIPVisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., interpolate_pos_encoding: bool = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, ChineseCLIPVisionModel

        >>> model = ChineseCLIPVisionModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        >>> processor = CLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

        >>> url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        ...
    


@add_start_docstrings(CHINESE_CLIP_START_DOCSTRING)
class ChineseCLIPModel(ChineseCLIPPreTrainedModel):
    config_class = ChineseCLIPConfig
    def __init__(self, config: ChineseCLIPConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CHINESE_CLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., token_type_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the final [CLS] hidden state of Text-Transformer.

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ChineseCLIPModel

        >>> model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        >>> tokenizer = AutoTokenizer.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

        >>> inputs = tokenizer(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        >>> text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        ```"""
        ...
    
    @add_start_docstrings_to_model_forward(CHINESE_CLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(self, pixel_values: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., interpolate_pos_encoding: bool = ..., return_dict: Optional[bool] = ...) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the final [CLS] hidden state of Vision-Transformer.

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ChineseCLIPModel

        >>> model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        >>> processor = AutoProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

        >>> url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        >>> image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        ```"""
        ...
    
    @add_start_docstrings_to_model_forward(CHINESE_CLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ChineseCLIPOutput, config_class=ChineseCLIPConfig)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., pixel_values: Optional[torch.FloatTensor] = ..., attention_mask: Optional[torch.Tensor] = ..., token_type_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., return_loss: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., interpolate_pos_encoding: bool = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, ChineseCLIPOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ChineseCLIPModel

        >>> model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        >>> processor = AutoProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

        >>> url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"], images=image, return_tensors="pt", padding=True)

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        ...
    


