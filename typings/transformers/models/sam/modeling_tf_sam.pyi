"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import tensorflow as tf

from ...modeling_tf_outputs import TFBaseModelOutput
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, keras, unpack_inputs
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig

"""
TensorFlow SAM model. This file was mostly generated by auto-translation from the PyTorch original. In the event of a
discrepancy, the original file should be regarded as the 'reference' version.
"""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
@dataclass
class TFSamVisionEncoderOutput(ModelOutput):
    """
    Base class for sam vision model's outputs that also contains image embeddings obtained by applying the projection
    layer to the pooler_output.

    Args:
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    image_embeds: tf.Tensor | None = ...
    last_hidden_state: tf.Tensor = ...
    hidden_states: Tuple[tf.Tensor, ...] | None = ...
    attentions: Tuple[tf.Tensor, ...] | None = ...


@dataclass
class TFSamImageSegmentationOutput(ModelOutput):
    """
    Base class for Segment-Anything model's output

    Args:
        iou_scores (`tf.Tensor` of shape `(batch_size, num_masks)`):
            The iou scores of the predicted masks.
        pred_masks (`tf.Tensor` of shape `(batch_size, num_masks, height, width)`):
            The predicted low resolutions masks. Needs to be post-processed by the processor
        vision_hidden_states  (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the vision model at the output of each layer plus the optional initial embedding outputs.
        vision_attentions  (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        mask_decoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    iou_scores: tf.Tensor = ...
    pred_masks: tf.Tensor = ...
    vision_hidden_states: Tuple[tf.Tensor, ...] | None = ...
    vision_attentions: Tuple[tf.Tensor, ...] | None = ...
    mask_decoder_attentions: Tuple[tf.Tensor, ...] | None = ...


class TFSamPatchEmbeddings(keras.layers.Layer):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """
    def __init__(self, config, **kwargs) -> None:
        ...

    def call(self, pixel_values):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFSamMLPBlock(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFSamLayerNorm(keras.layers.Layer):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=..., data_format=..., **kwargs) -> None:
        ...

    def build(self, input_shape): # -> None:
        ...

    def call(self, x: tf.Tensor) -> tf.Tensor:
        ...



class TFSamAttention(keras.layers.Layer):
    """
    SAM's attention layer that allows for downscaling the size of the embedding after projection to queries, keys, and
    values.
    """
    def __init__(self, config, downsample_rate=..., **kwargs) -> None:
        ...

    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFSamTwoWayAttentionBlock(keras.layers.Layer):
    def __init__(self, config, attention_downsample_rate: int = ..., skip_first_layer_pe: bool = ..., **kwargs) -> None:
        """
        A transformer block with four layers:
            (1) self-attention of sparse inputs (2) cross attention of sparse inputs -> dense inputs (3) mlp block on
            sparse inputs (4) cross attention of dense inputs -> sparse inputs

        Arguments:
            config (`SamMaskDecoderConfig`):
                The configuration file used to instantiate the block
            attention_downsample_rate (*optionalk*, int, defaults to 2):
                The downsample ratio of the block used to reduce the inner dim of the attention.
            skip_first_layer_pe (*optional*, bool, defaults to `False`):
                Whether or not to skip the addition of the query_point_embedding on the first layer.
        """
        ...

    def call(self, queries: tf.Tensor, keys: tf.Tensor, query_point_embedding: tf.Tensor, key_point_embedding: tf.Tensor, output_attentions: bool = ...): # -> tuple[Any, Any, Any] | tuple[Any, Any, None]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFSamTwoWayTransformer(keras.layers.Layer):
    def __init__(self, config: SamMaskDecoderConfig, **kwargs) -> None:
        ...

    def call(self, point_embeddings: tf.Tensor, image_embeddings: tf.Tensor, image_positional_embeddings: tf.Tensor, output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, TFBaseModelOutput]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFSamFeedForward(keras.layers.Layer):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = ..., **kwargs) -> None:
        ...

    def call(self, hidden_states):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFSamMaskDecoder(keras.layers.Layer):
    def __init__(self, config: SamMaskDecoderConfig, **kwargs) -> None:
        ...

    def build(self, input_shape=...): # -> None:
        ...

    def call(self, image_embeddings: tf.Tensor, image_positional_embeddings: tf.Tensor, sparse_prompt_embeddings: tf.Tensor, dense_prompt_embeddings: tf.Tensor, multimask_output: bool, output_attentions: Optional[bool] = ...) -> Tuple[tf.Tensor, tf.Tensor]:
        ...



class TFSamPositionalEmbedding(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...

    def build(self, input_shape): # -> None:
        ...

    def call(self, input_coords, input_shape=...):
        """Positionally encode points that are normalized to [0,1]."""
        ...



class TFSamMaskEmbedding(keras.layers.Layer):
    def __init__(self, config: SamPromptEncoderConfig, **kwargs) -> None:
        ...

    def call(self, masks):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFSamPromptEncoder(keras.layers.Layer):
    def __init__(self, config: SamPromptEncoderConfig, shared_patch_embedding, **kwargs) -> None:
        ...

    def build(self, input_shape=...): # -> None:
        ...

    def call(self, batch_size: Optional[int], input_points: Optional[Tuple[tf.Tensor, tf.Tensor]], input_labels: tf.Tensor | None, input_boxes: tf.Tensor | None, input_masks: tf.Tensor | None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Args:
            points (`tf.Tensor`, *optional*):
                point coordinates and labels to embed.
            boxes (`tf.Tensor`, *optional*):
                boxes to embed
            masks (`tf.Tensor`, *optional*):
                masks to embed
        """
        ...



class TFSamVisionAttention(keras.layers.Layer):
    """Multi-head Attention block with relative position embeddings."""
    def __init__(self, config, window_size, **kwargs) -> None:
        ...

    def build(self, input_shape=...): # -> None:
        ...

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: tf.Tensor) -> tf.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.

        Args:
            q_size (int):
                size of the query.
            k_size (int):
                size of key k.
            rel_pos (`tf.Tensor`):
                relative position embeddings (L, channel).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        ...

    def add_decomposed_rel_pos(self, attn: tf.Tensor, query: tf.Tensor, rel_pos_h: tf.Tensor, rel_pos_w: tf.Tensor, q_size: Tuple[int, int], k_size: Tuple[int, int]) -> tf.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Args:
            attn (`tf.Tensor`):
                attention map.
            query (`tf.Tensor`):
                query q in the attention layer with shape (batch_size, query_height * query_width, channel).
            rel_pos_h (`tf.Tensor`):
                relative position embeddings (Lh, channel) for height axis.
            rel_pos_w (`tf.Tensor`):
                relative position embeddings (Lw, channel) for width axis.
            q_size (tuple):
                spatial sequence size of query q with (query_height, query_width).
            k_size (tuple):
                spatial sequence size of key k with (key_height, key_width).

        Returns:
            attn (`tf.Tensor`):
                attention map with added relative positional embeddings.
        """
        ...

    def call(self, hidden_states: tf.Tensor, output_attentions=..., training=...) -> tf.Tensor:
        ...



class TFSamVisionLayer(keras.layers.Layer):
    def __init__(self, config, window_size, **kwargs) -> None:
        ...

    def window_partition(self, hidden_states: tf.Tensor, window_size: int) -> Tuple[tf.Tensor, Tuple[int, int]]:
        ...

    def window_unpartition(self, windows: tf.Tensor, window_size: int, padding_shape: Tuple[int, int], original_shape: Tuple[int, int]) -> tf.Tensor:
        ...

    def call(self, hidden_states: tf.Tensor, output_attentions: Optional[bool] = ..., training: Optional[bool] = ...) -> Tuple[tf.Tensor]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFSamVisionNeck(keras.layers.Layer):
    def __init__(self, config: SamVisionConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFSamVisionEncoder(keras.layers.Layer):
    def __init__(self, config: SamVisionConfig, **kwargs) -> None:
        ...

    def build(self, input_shape=...): # -> None:
        ...

    def get_input_embeddings(self): # -> TFSamPatchEmbeddings:
        ...

    def call(self, pixel_values: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[Tuple, TFSamVisionEncoderOutput]:
        ...



class TFSamPreTrainedModel(TFPreTrainedModel):
    config_class = SamConfig
    base_model_prefix = ...
    main_input_name = ...


SAM_START_DOCSTRING = ...
SAM_INPUTS_DOCSTRING = ...
@add_start_docstrings("Segment Anything Model (SAM) for generating segmentation masks, given an input image and ", " optional 2D location and bounding boxes.", SAM_START_DOCSTRING)
class TFSamModel(TFSamPreTrainedModel):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, **kwargs) -> None:
        ...

    def get_input_embeddings(self): # -> TFSamPatchEmbeddings:
        ...

    def get_image_wide_positional_embeddings(self):
        ...

    def get_image_embeddings(self, pixel_values, output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...):
        r"""
        Returns the image embeddings by passing the pixel values through the vision encoder.

        Args:
            pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Input pixel values
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.TFModelOutput`] instead of a plain tuple.

        """
        ...

    def get_prompt_embeddings(self, input_points: tf.Tensor | None = ..., input_labels: tf.Tensor | None = ..., input_boxes: tf.Tensor | None = ..., input_masks: tf.Tensor | None = ...):
        r"""
        Returns the prompt embeddings by passing the input points, labels, boxes and masks through the prompt encoder.

        Args:
            input_points (`tf.Tensor` of shape `(batch_size, point_batch_size, num_points_per_image, 2)`):
                Optional input points for the prompt encoder. The padding of the point is automatically done by the
                processor. `point_batch_size` refers to the number of masks that we want the model to predict per
                point. The model will output `point_batch_size` times 3 masks in total.
            input_labels (`tf.Tensor` of shape `(batch_size, point_batch_size, num_points_per_image)`):
                Optional input labels for the prompt encoder. The padding of the labels is automatically done by the
                processor, or can be fed by the user.
            input_boxes (`tf.Tensor` of shape `(batch_size, num_boxes_per_image, 4)`):
                Optional input boxes for the prompt encoder. The padding of the boxes is automatically done by the
                processor. users can also pass manually the input boxes.
            input_masks (`tf.Tensor` of shape `(batch_size, image_size, image_size)`):
                Optional input masks for the prompt encoder.
        """
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(SAM_INPUTS_DOCSTRING)
    def call(self, pixel_values: TFModelInputType | None = ..., input_points: tf.Tensor | None = ..., input_labels: tf.Tensor | None = ..., input_boxes: tf.Tensor | None = ..., input_masks: tf.Tensor | None = ..., image_embeddings: tf.Tensor | None = ..., multimask_output: bool = ..., output_attentions: bool | None = ..., output_hidden_states: bool | None = ..., return_dict: bool | None = ..., training: bool = ..., **kwargs) -> TFSamImageSegmentationOutput | Tuple[tf.Tensor]:
        ...

    def serving_output(self, output: TFSamImageSegmentationOutput) -> TFSamImageSegmentationOutput:
        ...

    def build(self, input_shape=...): # -> None:
        ...
