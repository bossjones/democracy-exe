"""
This type stub file was generated by pyright.
"""

from typing import Optional, Tuple, Union

import torch
from torch import nn

from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_efficientnet import EfficientNetConfig

""" PyTorch EfficientNet model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...
EFFICIENTNET_START_DOCSTRING = ...
EFFICIENTNET_INPUTS_DOCSTRING = ...
def round_filters(config: EfficientNetConfig, num_channels: int): # -> int:
    r"""
    Round number of filters based on depth multiplier.
    """
    ...

def correct_pad(kernel_size: Union[int, Tuple], adjust: bool = ...): # -> tuple[int | Any, int | Any, int | Any, int | Any]:
    r"""
    Utility function to get the tuple padding value for the depthwise convolution.

    Args:
        kernel_size (`int` or `tuple`):
            Kernel size of the convolution layers.
        adjust (`bool`, *optional*, defaults to `True`):
            Adjusts padding value to apply to right and bottom sides of the input.
    """
    ...

class EfficientNetEmbeddings(nn.Module):
    r"""
    A module that corresponds to the stem module of the original work.
    """
    def __init__(self, config: EfficientNetConfig) -> None:
        ...

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        ...



class EfficientNetDepthwiseConv2d(nn.Conv2d):
    def __init__(self, in_channels, depth_multiplier=..., kernel_size=..., stride=..., padding=..., dilation=..., bias=..., padding_mode=...) -> None:
        ...



class EfficientNetExpansionLayer(nn.Module):
    r"""
    This corresponds to the expansion phase of each block in the original implementation.
    """
    def __init__(self, config: EfficientNetConfig, in_dim: int, out_dim: int, stride: int) -> None:
        ...

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        ...



class EfficientNetDepthwiseLayer(nn.Module):
    r"""
    This corresponds to the depthwise convolution phase of each block in the original implementation.
    """
    def __init__(self, config: EfficientNetConfig, in_dim: int, stride: int, kernel_size: int, adjust_padding: bool) -> None:
        ...

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        ...



class EfficientNetSqueezeExciteLayer(nn.Module):
    r"""
    This corresponds to the Squeeze and Excitement phase of each block in the original implementation.
    """
    def __init__(self, config: EfficientNetConfig, in_dim: int, expand_dim: int, expand: bool = ...) -> None:
        ...

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        ...



class EfficientNetFinalBlockLayer(nn.Module):
    r"""
    This corresponds to the final phase of each block in the original implementation.
    """
    def __init__(self, config: EfficientNetConfig, in_dim: int, out_dim: int, stride: int, drop_rate: float, id_skip: bool) -> None:
        ...

    def forward(self, embeddings: torch.FloatTensor, hidden_states: torch.FloatTensor) -> torch.Tensor:
        ...



class EfficientNetBlock(nn.Module):
    r"""
    This corresponds to the expansion and depthwise convolution phase of each block in the original implementation.

    Args:
        config ([`EfficientNetConfig`]):
            Model configuration class.
        in_dim (`int`):
            Number of input channels.
        out_dim (`int`):
            Number of output channels.
        stride (`int`):
            Stride size to be used in convolution layers.
        expand_ratio (`int`):
            Expand ratio to set the output dimensions for the expansion and squeeze-excite layers.
        kernel_size (`int`):
            Kernel size for the depthwise convolution layer.
        drop_rate (`float`):
            Dropout rate to be used in the final phase of each block.
        id_skip (`bool`):
            Whether to apply dropout and sum the final hidden states with the input embeddings during the final phase
            of each block. Set to `True` for the first block of each stage.
        adjust_padding (`bool`):
            Whether to apply padding to only right and bottom side of the input kernel before the depthwise convolution
            operation, set to `True` for inputs with odd input sizes.
    """
    def __init__(self, config: EfficientNetConfig, in_dim: int, out_dim: int, stride: int, expand_ratio: int, kernel_size: int, drop_rate: float, id_skip: bool, adjust_padding: bool) -> None:
        ...

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        ...



class EfficientNetEncoder(nn.Module):
    r"""
    Forward propogates the embeddings through each EfficientNet block.

    Args:
        config ([`EfficientNetConfig`]):
            Model configuration class.
    """
    def __init__(self, config: EfficientNetConfig) -> None:
        ...

    def forward(self, hidden_states: torch.FloatTensor, output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> BaseModelOutputWithNoAttention:
        ...



class EfficientNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = EfficientNetConfig
    base_model_prefix = ...
    main_input_name = ...
    _no_split_modules = ...


@add_start_docstrings("The bare EfficientNet model outputting raw features without any specific head on top.", EFFICIENTNET_START_DOCSTRING)
class EfficientNetModel(EfficientNetPreTrainedModel):
    def __init__(self, config: EfficientNetConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(EFFICIENTNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPoolingAndNoAttention, config_class=_CONFIG_FOR_DOC, modality="vision", expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: torch.FloatTensor = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        ...



@add_start_docstrings("""
    EfficientNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g.
    for ImageNet.
    """, EFFICIENTNET_START_DOCSTRING)
class EfficientNetForImageClassification(EfficientNetPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    @add_start_docstrings_to_model_forward(EFFICIENTNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=ImageClassifierOutputWithNoAttention, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def forward(self, pixel_values: torch.FloatTensor = ..., labels: Optional[torch.LongTensor] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
