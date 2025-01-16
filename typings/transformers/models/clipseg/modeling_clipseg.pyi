"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_clipseg import CLIPSegConfig, CLIPSegTextConfig, CLIPSegVisionConfig

"""PyTorch CLIPSeg model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    ...

def clipseg_loss(similarity: torch.Tensor) -> torch.Tensor:
    ...

@dataclass
class CLIPSegOutput(ModelOutput):
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
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPSegTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPSegVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPSegTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPSegVisionModel`].
    """
    loss: Optional[torch.FloatTensor] = ...
    logits_per_image: torch.FloatTensor = ...
    logits_per_text: torch.FloatTensor = ...
    text_embeds: torch.FloatTensor = ...
    image_embeds: torch.FloatTensor = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> Tuple[Any]:
        ...
    


@dataclass
class CLIPSegDecoderOutput(ModelOutput):
    """
    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, height, width)`):
            Classification scores for each pixel.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """
    logits: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class CLIPSegImageSegmentationOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        ...
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`CLIPSegVisionModel`].
    """
    loss: Optional[torch.FloatTensor] = ...
    logits: torch.FloatTensor = ...
    conditional_embeddings: torch.FloatTensor = ...
    pooled_output: torch.FloatTensor = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    decoder_output: CLIPSegDecoderOutput = ...
    def to_tuple(self) -> Tuple[Any]:
        ...
    


class CLIPSegVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPSegVisionConfig) -> None:
        ...
    
    def interpolate_position_embeddings(self, new_size): # -> Tensor:
        ...
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        ...
    


class CLIPSegTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPSegTextConfig) -> None:
        ...
    
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ...) -> torch.Tensor:
        ...
    


class CLIPSegAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., causal_attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        ...
    


class CLIPSegMLP(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class CLIPSegEncoderLayer(nn.Module):
    def __init__(self, config: CLIPSegConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, causal_attention_mask: torch.Tensor, output_attentions: Optional[bool] = ...) -> Tuple[torch.FloatTensor]:
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
    


class CLIPSegPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = CLIPSegConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...


CLIPSEG_START_DOCSTRING = ...
CLIPSEG_TEXT_INPUTS_DOCSTRING = ...
CLIPSEG_VISION_INPUTS_DOCSTRING = ...
CLIPSEG_INPUTS_DOCSTRING = ...
class CLIPSegEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPSegEncoderLayer`].

    Args:
        config: CLIPSegConfig
    """
    def __init__(self, config: CLIPSegConfig) -> None:
        ...
    
    def forward(self, inputs_embeds, attention_mask: Optional[torch.Tensor] = ..., causal_attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

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
    


class CLIPSegTextTransformer(nn.Module):
    def __init__(self, config: CLIPSegTextConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CLIPSEG_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPSegTextConfig)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        ...
    


class CLIPSegTextModel(CLIPSegPreTrainedModel):
    config_class = CLIPSegTextConfig
    _no_split_modules = ...
    def __init__(self, config: CLIPSegTextConfig) -> None:
        ...
    
    def get_input_embeddings(self) -> nn.Module:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CLIPSEG_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPSegTextConfig)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPSegTextModel

        >>> tokenizer = AutoTokenizer.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegTextModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        ...
    


class CLIPSegVisionTransformer(nn.Module):
    def __init__(self, config: CLIPSegVisionConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CLIPSEG_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPSegVisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        ...
    


class CLIPSegVisionModel(CLIPSegPreTrainedModel):
    config_class = CLIPSegVisionConfig
    main_input_name = ...
    def __init__(self, config: CLIPSegVisionConfig) -> None:
        ...
    
    def get_input_embeddings(self) -> nn.Module:
        ...
    
    @add_start_docstrings_to_model_forward(CLIPSEG_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPSegVisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPSegVisionModel

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegVisionModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        ...
    


@add_start_docstrings(CLIPSEG_START_DOCSTRING)
class CLIPSegModel(CLIPSegPreTrainedModel):
    config_class = CLIPSegConfig
    def __init__(self, config: CLIPSegConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CLIPSEG_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPSegTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPSegModel

        >>> tokenizer = AutoTokenizer.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        ...
    
    @add_start_docstrings_to_model_forward(CLIPSEG_VISION_INPUTS_DOCSTRING)
    def get_image_features(self, pixel_values: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPSegVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPSegModel

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        ...
    
    @add_start_docstrings_to_model_forward(CLIPSEG_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPSegOutput, config_class=CLIPSegConfig)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., pixel_values: Optional[torch.FloatTensor] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., return_loss: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, CLIPSegOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPSegModel

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        ...
    


class CLIPSegDecoderLayer(nn.Module):
    """
    CLIPSeg decoder layer, which is identical to `CLIPSegEncoderLayer`, except that normalization is applied after
    self-attention/MLP, rather than before.
    """
    def __init__(self, config: CLIPSegConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, causal_attention_mask: torch.Tensor, output_attentions: Optional[bool] = ...) -> Tuple[torch.FloatTensor]:
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
    


class CLIPSegDecoder(CLIPSegPreTrainedModel):
    def __init__(self, config: CLIPSegConfig) -> None:
        ...
    
    def forward(self, hidden_states: Tuple[torch.Tensor], conditional_embeddings: torch.Tensor, output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...): # -> tuple[Any, ...] | CLIPSegDecoderOutput:
        ...
    


@add_start_docstrings("""
    CLIPSeg model with a Transformer-based decoder on top for zero-shot and one-shot image segmentation.
    """, CLIPSEG_START_DOCSTRING)
class CLIPSegForImageSegmentation(CLIPSegPreTrainedModel):
    config_class = CLIPSegConfig
    def __init__(self, config: CLIPSegConfig) -> None:
        ...
    
    def get_conditional_embeddings(self, batch_size: int = ..., input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., conditional_pixel_values: Optional[torch.Tensor] = ...): # -> FloatTensor:
        ...
    
    @add_start_docstrings_to_model_forward(CLIPSEG_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPSegImageSegmentationOutput, config_class=CLIPSegTextConfig)
    def forward(self, input_ids: Optional[torch.FloatTensor] = ..., pixel_values: Optional[torch.FloatTensor] = ..., conditional_pixel_values: Optional[torch.FloatTensor] = ..., conditional_embeddings: Optional[torch.FloatTensor] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., labels: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, CLIPSegOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, CLIPSegForImageSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = ["a cat", "a remote", "a blanket"]
        >>> inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> print(logits.shape)
        torch.Size([3, 352, 352])
        ```"""
        ...
    


