"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_data2vec_vision import Data2VecVisionConfig

"""PyTorch Data2VecVision model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...
@dataclass
class Data2VecVisionModelOutputWithPooling(BaseModelOutputWithPooling):
    """
    Class for outputs of [`Data2VecVisionModel`].

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Average of the last layer hidden states of the patch tokens (excluding the *[CLS]* token) if
            *config.use_mean_pooling* is set to True. If set to False, then the final hidden state of the *[CLS]* token
            will be returned.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    ...


def drop_path(input: torch.Tensor, drop_prob: float = ..., training: bool = ...) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    ...

class Data2VecVisionDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: Optional[float] = ...) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    
    def extra_repr(self) -> str:
        ...
    


class Data2VecVisionEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.

    """
    def __init__(self, config: Data2VecVisionConfig) -> None:
        ...
    
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows the model to interpolate the pre-trained position encodings so that it can be used on
        higher resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        ...
    
    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor] = ..., interpolate_pos_encoding: bool = ...) -> torch.Tensor:
        ...
    


class Data2VecVisionPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, pixel_values: torch.Tensor, position_embedding: Optional[torch.Tensor] = ...) -> torch.Tensor:
        ...
    


class Data2VecVisionSelfAttention(nn.Module):
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = ...) -> None:
        ...
    
    def transpose_for_scores(self, x):
        ...
    
    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ..., relative_position_bias: Optional[Data2VecVisionRelativePositionBias] = ..., interpolate_pos_encoding: bool = ..., resolution: Optional[Tuple[int]] = ...) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        ...
    


class Data2VecVisionSelfOutput(nn.Module):
    """
    The residual connection is defined in Data2VecVisionLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """
    def __init__(self, config: Data2VecVisionConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, gamma=...) -> torch.Tensor:
        ...
    


class Data2VecVisionAttention(nn.Module):
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = ...) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ..., relative_position_bias: Optional[Data2VecVisionRelativePositionBias] = ..., interpolate_pos_encoding: bool = ..., resolution: Optional[Tuple[int]] = ...) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        ...
    


class Data2VecVisionIntermediate(nn.Module):
    def __init__(self, config: Data2VecVisionConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class Data2VecVisionOutput(nn.Module):
    def __init__(self, config: Data2VecVisionConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class Data2VecVisionLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = ..., drop_path_rate: float = ...) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ..., relative_position_bias: Optional[Data2VecVisionRelativePositionBias] = ..., interpolate_pos_encoding: bool = ..., resolution: Optional[Tuple[int]] = ...) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        ...
    


class Data2VecVisionRelativePositionBias(nn.Module):
    def __init__(self, config: Data2VecVisionConfig, window_size: tuple) -> None:
        ...
    
    def generate_relative_position_index(self, window_size: Tuple[int, int]) -> torch.Tensor:
        """
        This method creates the relative position index, modified to support arbitrary window sizes,
        as introduced in [MiDaS v3.1](https://arxiv.org/abs/2307.14460).
        """
        ...
    
    def forward(self, window_size, interpolate_pos_encoding: bool = ..., dim_size=...) -> torch.Tensor:
        """
        Modification of timm.models.beit.py: Attention._get_rel_pos_bias to support arbitrary window sizes.
        """
        ...
    


class Data2VecVisionEncoder(nn.Module):
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = ...) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., interpolate_pos_encoding: bool = ..., resolution: Optional[Tuple[int]] = ..., return_dict: bool = ...) -> Union[tuple, BaseModelOutput]:
        ...
    


class Data2VecVisionPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Data2VecVisionConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _keys_to_ignore_on_load_unexpected = ...


DATA2VEC_VISION_START_DOCSTRING = ...
DATA2VEC_VISION_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Data2VecVision Model transformer outputting raw hidden-states without any specific head on top.", DATA2VEC_VISION_START_DOCSTRING)
class Data2VecVisionModel(Data2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig, add_pooling_layer: bool = ...) -> None:
        ...
    
    def get_input_embeddings(self): # -> Data2VecVisionPatchEmbeddings:
        ...
    
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=Data2VecVisionModelOutputWithPooling, config_class=_CONFIG_FOR_DOC, modality="vision", expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor] = ..., head_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., interpolate_pos_encoding: bool = ..., return_dict: Optional[bool] = ...) -> Union[tuple, Data2VecVisionModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        ...
    


class Data2VecVisionPooler(nn.Module):
    def __init__(self, config: Data2VecVisionConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


@add_start_docstrings("""
    Data2VecVision Model transformer with an image classification head on top (a linear layer on top of the average of
    the final hidden states of the patch tokens) e.g. for ImageNet.
    """, DATA2VEC_VISION_START_DOCSTRING)
class Data2VecVisionForImageClassification(Data2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=ImageClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def forward(self, pixel_values: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., labels: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., interpolate_pos_encoding: bool = ..., return_dict: Optional[bool] = ...) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


class Data2VecVisionConvModule(nn.Module):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int], str] = ..., bias: bool = ..., dilation: Union[int, Tuple[int, int]] = ...) -> None:
        ...
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ...
    


class Data2VecVisionPyramidPoolingBlock(nn.Module):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None:
        ...
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ...
    


class Data2VecVisionPyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate.

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """
    def __init__(self, pool_scales: Tuple[int, ...], in_channels: int, channels: int, align_corners: bool) -> None:
        ...
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        ...
    


class Data2VecVisionUperHead(nn.Module):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """
    def __init__(self, config: Data2VecVisionConfig) -> None:
        ...
    
    def psp_forward(self, inputs): # -> Any:
        ...
    
    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class Data2VecVisionFCNHead(nn.Module):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is implemented of
    [FCNNet](https://arxiv.org/abs/1411.4038>).

    Args:
        config (Data2VecVisionConfig): Configuration.
        in_channels
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        dilation (int): The dilation rate for convs in the head. Default: 1.


    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """
    def __init__(self, config: Data2VecVisionConfig, in_index: int = ..., kernel_size: int = ..., dilation: Union[int, Tuple[int, int]] = ...) -> None:
        ...
    
    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


@add_start_docstrings("""
    Data2VecVision Model transformer with a semantic segmentation head on top e.g. for ADE20k, CityScapes.
    """, DATA2VEC_VISION_START_DOCSTRING)
class Data2VecVisionForSemanticSegmentation(Data2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig) -> None:
        ...
    
    def compute_loss(self, logits, auxiliary_logits, labels): # -> Any:
        ...
    
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., labels: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., interpolate_pos_encoding: bool = ..., return_dict: Optional[bool] = ...) -> Union[tuple, SemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, Data2VecVisionForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
        >>> model = Data2VecVisionForSemanticSegmentation.from_pretrained("facebook/data2vec-vision-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> # logits are of shape (batch_size, num_labels, height, width)
        >>> logits = outputs.logits
        ```"""
        ...
    


