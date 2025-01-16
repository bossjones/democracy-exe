"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput, ModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_bridgetower import BridgeTowerConfig, BridgeTowerTextConfig, BridgeTowerVisionConfig

"""PyTorch BridgeTower Model"""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
BRIDGETOWER_START_DOCSTRING = ...
BRIDGETOWER_INPUTS_DOCSTRING = ...
@dataclass
class BridgeTowerModelOutput(ModelOutput):
    """
    Output type of [`BridgeTowerModel`].

    Args:
        text_features (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, hidden_size)`):
            Sequence of hidden-states at the text output of the last layer of the model.
        image_features (`torch.FloatTensor` of shape `(batch_size, image_sequence_length, hidden_size)`):
            Sequence of hidden-states at the image output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size x 2)`):
            Concatenation of last layer hidden-state of the first token of the text and image sequence (classification
            token), respectively, after further processing through layers used for auxiliary pretraining tasks.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    text_features: torch.FloatTensor = ...
    image_features: torch.FloatTensor = ...
    pooler_output: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class BridgeTowerContrastiveOutput(ModelOutput):
    """
    Output type of ['BridgeTowerForContrastiveLearning']

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`:
            Image-text contrastive loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        text_embeds (`torch.FloatTensor)`, *optional*, returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        image_embeds (`torch.FloatTensor)`, *optional*, returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        cross_embeds  (`torch.FloatTensor)`, *optional*, returned when model is initialized with `with_projection=True`):
            The text-image cross-modal embeddings obtained by applying the projection layer to the pooler_output.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
    """
    loss: Optional[torch.FloatTensor] = ...
    logits: torch.FloatTensor = ...
    text_embeds: Optional[Tuple[torch.FloatTensor]] = ...
    image_embeds: Optional[Tuple[torch.FloatTensor]] = ...
    cross_embeds: Optional[Tuple[torch.FloatTensor]] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


class BridgeTowerResidualAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def attention(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor): # -> Any:
        ...
    
    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor = ...): # -> Tensor:
        ...
    


class BridgeTowerTransformer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = ...): # -> list[Any]:
        ...
    


class BridgeTowerVisionEmbeddings(nn.Module):
    def __init__(self, config: BridgeTowerVisionConfig) -> None:
        ...
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        ...
    


class BridgeTowerVisionTransformer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, pixel_values: torch.Tensor, attention_mask): # -> Any | Tensor:
        ...
    
    def forward_pre(self, pixel_values: torch.Tensor): # -> Any:
        ...
    
    def forward_post(self, hidden_state: torch.Tensor): # -> Any:
        ...
    


class BridgeTowerLinkTower(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, cross_modal_hidden_states, attention_mask): # -> Any:
        ...
    


class BridgeTowerSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...
    


class BridgeTowerIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class BridgeTowerOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...
    


class BridgeTowerPooler(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class BridgeTowerSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None:
        ...
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...
    


BRIDGE_TOWER_SELF_ATTENTION_CLASSES = ...
class BridgeTowerAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...
    


class BridgeTowerBertCrossLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, encoder_hidden_states, attention_mask=..., head_mask=..., encoder_attention_mask=..., past_key_value=..., output_attentions=...): # -> Any:
        ...
    
    def feed_forward_chunk(self, attention_output): # -> Any:
        ...
    


class BridgeTowerTextLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...
    
    def feed_forward_chunk(self, attention_output): # -> Any:
        ...
    


class BridgeTowerTextEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        ...
    


class BridgeTowerTextEmbeddings(nn.Module):
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
    


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=...):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    ...

class BridgeTowerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BridgeTowerConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...


class BridgeTowerVisionModel(BridgeTowerPreTrainedModel):
    config_class = BridgeTowerVisionConfig
    def __init__(self, config) -> None:
        ...
    
    @property
    def dtype(self): # -> dtype:
        ...
    
    def forward(self, image, image_mask=...): # -> Any:
        ...
    


class BridgeTowerTextModel(BridgeTowerPreTrainedModel):
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
    config_class = BridgeTowerTextConfig
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
    


@add_start_docstrings("The bare BridgeTower Model transformer outputting BridgeTowerModelOutput object without any specific head on" " top.", BRIDGETOWER_START_DOCSTRING)
class BridgeTowerModel(BridgeTowerPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(BRIDGETOWER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BridgeTowerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., pixel_values: Optional[torch.FloatTensor] = ..., pixel_mask: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., image_embeds: Optional[torch.FloatTensor] = ..., image_token_type_idx: Optional[int] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[torch.LongTensor] = ...) -> Union[Tuple[torch.Tensor], BridgeTowerModelOutput]:
        r"""
        output_hidden_states (`bool`, *optional*):
            If set to `True`, hidden states are returned as a list containing the hidden states of text, image, and
            cross-modal components respectively. i.e. `(hidden_states_text, hidden_states_image,
            hidden_states_cross_modal)` where each element is a list of the hidden states of the corresponding
            modality. `hidden_states_txt/img` are a list of tensors corresponding to unimodal hidden states and
            `hidden_states_cross_modal` is a list of tuples containing `cross_modal_text_hidden_states` and
            `cross_modal_image_hidden_states` of each brdige layer.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels are currently not supported.
        Returns:

        Examples:

        ```python
        >>> from transformers import BridgeTowerProcessor, BridgeTowerModel
        >>> from PIL import Image
        >>> import requests

        >>> # prepare image and text
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "hello world"
        >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base")
        >>> model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")

        >>> inputs = processor(image, text, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> outputs.keys()
        odict_keys(['text_features', 'image_features', 'pooler_output'])
        ```"""
        ...
    
    def get_cls_features(self, text_features, image_features): # -> Tensor:
        ...
    


class BridgeTowerPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class BridgeTowerMLMHead(nn.Module):
    def __init__(self, config, weight=...) -> None:
        ...
    
    def forward(self, x): # -> Any:
        ...
    


class BridgeTowerITMHead(nn.Module):
    def __init__(self, hidden_size) -> None:
        ...
    
    def forward(self, x): # -> Any:
        ...
    


@add_start_docstrings("""
    BridgeTower Model with a language modeling head on top as done during pretraining.
    """, BRIDGETOWER_START_DOCSTRING)
class BridgeTowerForMaskedLM(BridgeTowerPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(BRIDGETOWER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., pixel_values: Optional[torch.FloatTensor] = ..., pixel_mask: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., image_embeds: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[torch.LongTensor] = ...) -> Union[MaskedLMOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        Returns:

        Examples:

        ```python
        >>> from transformers import BridgeTowerProcessor, BridgeTowerForMaskedLM
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000360943.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> text = "a <mask> looking out of the window"

        >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
        >>> model = BridgeTowerForMaskedLM.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")

        >>> # prepare inputs
        >>> encoding = processor(image, text, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**encoding)

        >>> results = processor.decode(outputs.logits.argmax(dim=-1).squeeze(0).tolist())

        >>> print(results)
        .a cat looking out of the window.
        ```"""
        ...
    


@add_start_docstrings("""
    BridgeTower Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the
    [CLS] token) for image-to-text matching.
    """, BRIDGETOWER_START_DOCSTRING)
class BridgeTowerForImageAndTextRetrieval(BridgeTowerPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(BRIDGETOWER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., pixel_values: Optional[torch.FloatTensor] = ..., pixel_mask: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., image_embeds: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[torch.LongTensor] = ...) -> Union[SequenceClassifierOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*):
            Labels for computing the image-text matching loss. 0 means the pairs don't match and 1 means they match.
            The pairs with 0 will be skipped for calculation.
        Returns:

        Examples:

        ```python
        >>> from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval
        >>> import requests
        >>> from PIL import Image

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

        >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
        >>> model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")

        >>> # forward pass
        >>> scores = dict()
        >>> for text in texts:
        ...     # prepare inputs
        ...     encoding = processor(image, text, return_tensors="pt")
        ...     outputs = model(**encoding)
        ...     scores[text] = outputs.logits[0, 1].item()
        ```"""
        ...
    


class BridgeTowerContrastiveHead(nn.Module):
    def __init__(self, hidden_size, embed_size) -> None:
        ...
    
    def forward(self, x): # -> Any:
        ...
    


@add_start_docstrings("""
    BridgeTower Model with a image-text contrastive head on top computing image-text contrastive loss.
    """, BRIDGETOWER_START_DOCSTRING)
class BridgeTowerForContrastiveLearning(BridgeTowerPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(BRIDGETOWER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BridgeTowerContrastiveOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., pixel_values: Optional[torch.FloatTensor] = ..., pixel_mask: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., image_embeds: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., return_loss: Optional[bool] = ...) -> Union[BridgeTowerContrastiveOutput, Tuple[torch.FloatTensor]]:
        r"""
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        Returns:

        Examples:

        ```python
        >>> from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
        >>> import requests
        >>> from PIL import Image
        >>> import torch

        >>> image_urls = [
        ...     "https://farm4.staticflickr.com/3395/3428278415_81c3e27f15_z.jpg",
        ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
        ... ]
        >>> texts = ["two dogs in a car", "two cats sleeping on a couch"]
        >>> images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]

        >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
        >>> model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")

        >>> inputs = processor(images, texts, padding=True, return_tensors="pt")
        >>> loss = model(**inputs, return_loss=True).loss

        >>> inputs = processor(images, texts[::-1], padding=True, return_tensors="pt")
        >>> loss_swapped = model(**inputs, return_loss=True).loss

        >>> print("Loss", round(loss.item(), 4))
        Loss 0.0019

        >>> print("Loss with swapped images", round(loss_swapped.item(), 4))
        Loss with swapped images 2.126
        ```"""
        ...
    


