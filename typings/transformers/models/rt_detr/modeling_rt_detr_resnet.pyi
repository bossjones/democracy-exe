"""
This type stub file was generated by pyright.
"""

from typing import Optional
from torch import Tensor, nn
from ...modeling_outputs import BackboneOutput, BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...utils.backbone_utils import BackboneMixin
from .configuration_rt_detr_resnet import RTDetrResNetConfig

"""
PyTorch RTDetr specific ResNet model. The main difference between hugginface ResNet model is that this RTDetrResNet model forces to use shortcut at the first layer in the resnet-18/34 models.
See https://github.com/lyuwenyu/RT-DETR/blob/5b628eaa0a2fc25bdafec7e6148d5296b144af85/rtdetr_pytorch/src/nn/backbone/presnet.py#L126 for details.
"""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
class RTDetrResNetConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = ..., stride: int = ..., activation: str = ...) -> None:
        ...
    
    def forward(self, input: Tensor) -> Tensor:
        ...
    


class RTDetrResNetEmbeddings(nn.Module):
    """
    ResNet Embeddings (stem) composed of a deep aggressive convolution.
    """
    def __init__(self, config: RTDetrResNetConfig) -> None:
        ...
    
    def forward(self, pixel_values: Tensor) -> Tensor:
        ...
    


class RTDetrResNetShortCut(nn.Module):
    """
    ResNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = ...) -> None:
        ...
    
    def forward(self, input: Tensor) -> Tensor:
        ...
    


class RTDetrResNetBasicLayer(nn.Module):
    """
    A classic ResNet's residual layer composed by two `3x3` convolutions.
    See https://github.com/lyuwenyu/RT-DETR/blob/5b628eaa0a2fc25bdafec7e6148d5296b144af85/rtdetr_pytorch/src/nn/backbone/presnet.py#L34.
    """
    def __init__(self, config: RTDetrResNetConfig, in_channels: int, out_channels: int, stride: int = ..., should_apply_shortcut: bool = ...) -> None:
        ...
    
    def forward(self, hidden_state):
        ...
    


class RTDetrResNetBottleNeckLayer(nn.Module):
    """
    A classic RTDetrResNet's bottleneck layer composed by three `3x3` convolutions.

    The first `1x1` convolution reduces the input by a factor of `reduction` in order to make the second `3x3`
    convolution faster. The last `1x1` convolution remaps the reduced features to `out_channels`. If
    `downsample_in_bottleneck` is true, downsample will be in the first layer instead of the second layer.
    """
    def __init__(self, config: RTDetrResNetConfig, in_channels: int, out_channels: int, stride: int = ...) -> None:
        ...
    
    def forward(self, hidden_state):
        ...
    


class RTDetrResNetStage(nn.Module):
    """
    A RTDetrResNet stage composed by stacked layers.
    """
    def __init__(self, config: RTDetrResNetConfig, in_channels: int, out_channels: int, stride: int = ..., depth: int = ...) -> None:
        ...
    
    def forward(self, input: Tensor) -> Tensor:
        ...
    


class RTDetrResNetEncoder(nn.Module):
    def __init__(self, config: RTDetrResNetConfig) -> None:
        ...
    
    def forward(self, hidden_state: Tensor, output_hidden_states: bool = ..., return_dict: bool = ...) -> BaseModelOutputWithNoAttention:
        ...
    


class RTDetrResNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = RTDetrResNetConfig
    base_model_prefix = ...
    main_input_name = ...
    _no_split_modules = ...


RTDETR_RESNET_START_DOCSTRING = ...
RTDETR_RESNET_INPUTS_DOCSTRING = ...
@add_start_docstrings("""
    ResNet backbone, to be used with frameworks like RTDETR.
    """, RTDETR_RESNET_START_DOCSTRING)
class RTDetrResNetBackbone(RTDetrResNetPreTrainedModel, BackboneMixin):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(RTDETR_RESNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Tensor, output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import RTDetrResNetConfig, RTDetrResNetBackbone
        >>> import torch

        >>> config = RTDetrResNetConfig()
        >>> model = RTDetrResNetBackbone(config)

        >>> pixel_values = torch.randn(1, 3, 224, 224)

        >>> with torch.no_grad():
        ...     outputs = model(pixel_values)

        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 2048, 7, 7]
        ```"""
        ...
    


