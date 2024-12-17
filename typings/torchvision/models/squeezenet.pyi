"""
This type stub file was generated by pyright.
"""

import torch
import torch.nn as nn
from typing import Any, Optional
from ._api import WeightsEnum, register_model
from ._utils import handle_legacy_interface

__all__ = ["SqueezeNet", "SqueezeNet1_0_Weights", "SqueezeNet1_1_Weights", "squeezenet1_0", "squeezenet1_1"]
class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
    


class SqueezeNet(nn.Module):
    def __init__(self, version: str = ..., num_classes: int = ..., dropout: float = ...) -> None:
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
    


_COMMON_META = ...
class SqueezeNet1_0_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    DEFAULT = ...


class SqueezeNet1_1_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    DEFAULT = ...


@register_model()
@handle_legacy_interface(weights=("pretrained", SqueezeNet1_0_Weights.IMAGENET1K_V1))
def squeezenet1_0(*, weights: Optional[SqueezeNet1_0_Weights] = ..., progress: bool = ..., **kwargs: Any) -> SqueezeNet:
    """SqueezeNet model architecture from the `SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        weights (:class:`~torchvision.models.SqueezeNet1_0_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.SqueezeNet1_0_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.squeezenet.SqueezeNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.SqueezeNet1_0_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", SqueezeNet1_1_Weights.IMAGENET1K_V1))
def squeezenet1_1(*, weights: Optional[SqueezeNet1_1_Weights] = ..., progress: bool = ..., **kwargs: Any) -> SqueezeNet:
    """SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.

    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        weights (:class:`~torchvision.models.SqueezeNet1_1_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.SqueezeNet1_1_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.squeezenet.SqueezeNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.SqueezeNet1_1_Weights
        :members:
    """
    ...

