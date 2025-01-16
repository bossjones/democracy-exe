"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor, nn
from ...modeling_outputs import BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, is_accelerate_available, is_scipy_available, is_timm_available, is_vision_available, replace_return_docstrings
from .configuration_table_transformer import TableTransformerConfig

"""PyTorch Table Transformer model."""
if is_scipy_available():
    ...
if is_timm_available():
    ...
if is_vision_available():
    ...
if is_accelerate_available():
    ...
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
@dataclass
class TableTransformerDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Base class for outputs of the TABLE_TRANSFORMER decoder. This class adds one attribute to BaseModelOutputWithCrossAttentions,
    namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them
    gone through a layernorm. This is useful when training the model with auxiliary decoding losses.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, num_queries, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
    """
    intermediate_hidden_states: Optional[torch.FloatTensor] = ...


@dataclass
class TableTransformerModelOutput(Seq2SeqModelOutput):
    """
    Base class for outputs of the TABLE_TRANSFORMER encoder-decoder model. This class adds one attribute to Seq2SeqModelOutput,
    namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them
    gone through a layernorm. This is useful when training the model with auxiliary decoding losses.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of each
            layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, sequence_length, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
    """
    intermediate_hidden_states: Optional[torch.FloatTensor] = ...


@dataclass
class TableTransformerObjectDetectionOutput(ModelOutput):
    """
    Output type of [`TableTransformerForObjectDetection`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~TableTransformerImageProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxilary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of each
            layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = ...
    loss_dict: Optional[Dict] = ...
    logits: torch.FloatTensor = ...
    pred_boxes: torch.FloatTensor = ...
    auxiliary_outputs: Optional[List[Dict]] = ...
    last_hidden_state: Optional[torch.FloatTensor] = ...
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    encoder_last_hidden_state: Optional[torch.FloatTensor] = ...
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = ...


class TableTransformerFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """
    def __init__(self, n) -> None:
        ...
    
    def forward(self, x):
        ...
    


def replace_batch_norm(model): # -> None:
    r"""
    Recursively replace all `torch.nn.BatchNorm2d` with `TableTransformerFrozenBatchNorm2d`.

    Args:
        model (torch.nn.Module):
            input model
    """
    ...

class TableTransformerConvEncoder(nn.Module):
    """
    Convolutional backbone, using either the AutoBackbone API or one from the timm library.

    nn.BatchNorm2d layers are replaced by TableTransformerFrozenBatchNorm2d as defined above.

    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor): # -> list[Any]:
        ...
    


class TableTransformerConvModel(nn.Module):
    """
    This module adds 2D position embeddings to all intermediate feature maps of the convolutional encoder.
    """
    def __init__(self, conv_encoder, position_embedding) -> None:
        ...
    
    def forward(self, pixel_values, pixel_mask): # -> tuple[Any, list[Any]]:
        ...
    


class TableTransformerSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """
    def __init__(self, embedding_dim=..., temperature=..., normalize=..., scale=...) -> None:
        ...
    
    def forward(self, pixel_values, pixel_mask): # -> Tensor:
        ...
    


class TableTransformerLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    def __init__(self, embedding_dim=...) -> None:
        ...
    
    def forward(self, pixel_values, pixel_mask=...): # -> Tensor:
        ...
    


def build_position_encoding(config): # -> TableTransformerSinePositionEmbedding | TableTransformerLearnedPositionEmbedding:
    ...

class TableTransformerAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the TABLE_TRANSFORMER paper).
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = ..., bias: bool = ...) -> None:
        ...
    
    def with_pos_embed(self, tensor: torch.Tensor, object_queries: Optional[Tensor]): # -> Tensor:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., object_queries: Optional[torch.Tensor] = ..., key_value_states: Optional[torch.Tensor] = ..., spatial_position_embeddings: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        ...
    


class TableTransformerEncoderLayer(nn.Module):
    def __init__(self, config: TableTransformerConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, object_queries: torch.Tensor = ..., output_attentions: bool = ...): # -> tuple[Tensor, Any] | tuple[Tensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            object_queries (`torch.FloatTensor`, *optional*): object queries, to be added to hidden_states.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        ...
    


class TableTransformerDecoderLayer(nn.Module):
    def __init__(self, config: TableTransformerConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., object_queries: Optional[torch.Tensor] = ..., query_position_embeddings: Optional[torch.Tensor] = ..., encoder_hidden_states: Optional[torch.Tensor] = ..., encoder_attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ...): # -> tuple[Tensor, Any, Any | None] | tuple[Tensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            object_queries (`torch.FloatTensor`, *optional*):
                object queries that are added to the queries and keys
            in the cross-attention layer.
            query_position_embeddings (`torch.FloatTensor`, *optional*):
                object queries that are added to the queries and keys
            in the self-attention layer.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        ...
    


class TableTransformerPreTrainedModel(PreTrainedModel):
    config_class = TableTransformerConfig
    base_model_prefix = ...
    main_input_name = ...
    _no_split_modules = ...


TABLE_TRANSFORMER_START_DOCSTRING = ...
TABLE_TRANSFORMER_INPUTS_DOCSTRING = ...
class TableTransformerEncoder(TableTransformerPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`TableTransformerEncoderLayer`].

    The encoder updates the flattened feature map through multiple self-attention layers.

    Small tweak for Table Transformer:

    - object_queries are added to the forward pass.

    Args:
        config: TableTransformerConfig
    """
    def __init__(self, config: TableTransformerConfig) -> None:
        ...
    
    def forward(self, inputs_embeds=..., attention_mask=..., object_queries=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> tuple[Any, ...] | BaseModelOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.

            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:

                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).

                [What are attention masks?](../glossary#attention-mask)

            object_queries (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.

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
    


class TableTransformerDecoder(TableTransformerPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TableTransformerDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some small tweaks for TABLE_TRANSFORMER:

    - object_queries and query_position_embeddings are added to the forward pass.
    - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

    Args:
        config: TableTransformerConfig
    """
    def __init__(self, config: TableTransformerConfig) -> None:
        ...
    
    def forward(self, inputs_embeds=..., attention_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., object_queries=..., query_position_embeddings=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The query embeddings that are passed into the decoder.

            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on certain queries. Mask values selected in `[0, 1]`:

                - 1 for queries that are **not masked**,
                - 0 for queries that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            object_queries (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Object queries that are added to the queries and keys in each cross-attention layer.
            query_position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                , *optional*): Position embeddings that are added to the values and keys in each self-attention layer.

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
    


@add_start_docstrings("""
    The bare Table Transformer Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states without any specific head on top.
    """, TABLE_TRANSFORMER_START_DOCSTRING)
class TableTransformerModel(TableTransformerPreTrainedModel):
    def __init__(self, config: TableTransformerConfig) -> None:
        ...
    
    def get_encoder(self): # -> TableTransformerEncoder:
        ...
    
    def get_decoder(self): # -> TableTransformerDecoder:
        ...
    
    def freeze_backbone(self): # -> None:
        ...
    
    def unfreeze_backbone(self): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(TABLE_TRANSFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TableTransformerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, pixel_mask: Optional[torch.FloatTensor] = ..., decoder_attention_mask: Optional[torch.FloatTensor] = ..., encoder_outputs: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., decoder_inputs_embeds: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.FloatTensor], TableTransformerModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TableTransformerModel
        >>> from huggingface_hub import hf_hub_download
        >>> from PIL import Image

        >>> file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")
        >>> image = Image.open(file_path).convert("RGB")

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        >>> model = TableTransformerModel.from_pretrained("microsoft/table-transformer-detection")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> # the last hidden states are the final query embeddings of the Transformer decoder
        >>> # these are of shape (batch_size, num_queries, hidden_size)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 15, 256]
        ```"""
        ...
    


@add_start_docstrings("""
    Table Transformer Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on
    top, for tasks such as COCO detection.
    """, TABLE_TRANSFORMER_START_DOCSTRING)
class TableTransformerForObjectDetection(TableTransformerPreTrainedModel):
    def __init__(self, config: TableTransformerConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(TABLE_TRANSFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TableTransformerObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, pixel_mask: Optional[torch.FloatTensor] = ..., decoder_attention_mask: Optional[torch.FloatTensor] = ..., encoder_outputs: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., decoder_inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[List[Dict]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.FloatTensor], TableTransformerObjectDetectionOutput]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns:

        Examples:

        ```python
        >>> from huggingface_hub import hf_hub_download
        >>> from transformers import AutoImageProcessor, TableTransformerForObjectDetection
        >>> import torch
        >>> from PIL import Image

        >>> file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")
        >>> image = Image.open(file_path).convert("RGB")

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        >>> model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
        ...     0
        ... ]

        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected table with confidence 1.0 at location [202.1, 210.59, 1119.22, 385.09]
        ```"""
        ...
    


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
    """
    ...

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = ..., gamma: float = ...):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    ...

class TableTransformerLoss(nn.Module):
    """
    This class computes the losses for TableTransformerForObjectDetection/TableTransformerForSegmentation. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair
    of matched ground-truth / prediction (supervise class and box).

    A note on the `num_classes` argument (copied from original repo in table_transformer.py): "the naming of the `num_classes`
    parameter of the criterion is somewhat misleading. It indeed corresponds to `max_obj_id` + 1, where `max_obj_id` is
    the maximum id for a class in your dataset. For example, COCO has a `max_obj_id` of 90, so we pass `num_classes` to
    be 91. As another example, for a dataset that has a single class with `id` 1, you should pass `num_classes` to be 2
    (`max_obj_id` + 1). For more details on this, check the following discussion
    https://github.com/facebookresearch/table_transformer/issues/108#issuecomment-650269223"


    Args:
        matcher (`TableTransformerHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        eos_coef (`float`):
            Relative classification weight applied to the no-object category.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """
    def __init__(self, matcher, num_classes, eos_coef, losses) -> None:
        ...
    
    def loss_labels(self, outputs, targets, indices, num_boxes): # -> dict[str, Tensor]:
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        ...
    
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes): # -> dict[str, Tensor]:
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        ...
    
    def loss_boxes(self, outputs, targets, indices, num_boxes): # -> dict[Any, Any]:
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        ...
    
    def loss_masks(self, outputs, targets, indices, num_boxes): # -> dict[str, Any]:
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        ...
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        ...
    
    def forward(self, outputs, targets): # -> dict[Any, Any]:
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        ...
    


class TableTransformerMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/table_transformer/blob/master/models/table_transformer.py

    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers) -> None:
        ...
    
    def forward(self, x): # -> Tensor | Any:
        ...
    


class TableTransformerHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Args:
        class_cost:
            The relative weight of the classification error in the matching cost.
        bbox_cost:
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost:
            The relative weight of the giou loss of the bounding box in the matching cost.
    """
    def __init__(self, class_cost: float = ..., bbox_cost: float = ..., giou_cost: float = ...) -> None:
        ...
    
    @torch.no_grad()
    def forward(self, outputs, targets): # -> list[tuple[Tensor, Tensor]]:
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
            targets (`List[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                  ground-truth
                 objects in the target) containing the class labels
                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        ...
    


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    ...

def box_iou(boxes1, boxes2): # -> tuple[Any, Any]:
    ...

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    ...

class NestedTensor:
    def __init__(self, tensors, mask: Optional[Tensor]) -> None:
        ...
    
    def to(self, device): # -> NestedTensor:
        ...
    
    def decompose(self): # -> tuple[Any, Tensor | None]:
        ...
    
    def __repr__(self): # -> str:
        ...
    


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]): # -> NestedTensor:
    ...

