"""
This type stub file was generated by pyright.
"""

from typing import Any, Dict, Optional

from torch import Tensor, nn

from .._api import WeightsEnum, register_model
from .._utils import handle_legacy_interface
from ..mobilenetv3 import MobileNet_V3_Large_Weights

__all__ = ["LRASPP", "LRASPP_MobileNet_V3_Large_Weights", "lraspp_mobilenet_v3_large"]
class LRASPP(nn.Module):
    """
    Implements a Lite R-ASPP Network for semantic segmentation from
    `"Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "high" for the high level feature map and "low" for the low level feature map.
        low_channels (int): the number of channels of the low level features.
        high_channels (int): the number of channels of the high level features.
        num_classes (int, optional): number of output classes of the model (including the background).
        inter_channels (int, optional): the number of channels for intermediate computations.
    """
    def __init__(self, backbone: nn.Module, low_channels: int, high_channels: int, num_classes: int, inter_channels: int = ...) -> None:
        ...

    def forward(self, input: Tensor) -> Dict[str, Tensor]:
        ...



class LRASPPHead(nn.Module):
    def __init__(self, low_channels: int, high_channels: int, num_classes: int, inter_channels: int) -> None:
        ...

    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        ...



class LRASPP_MobileNet_V3_Large_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = ...
    DEFAULT = ...


@register_model()
@handle_legacy_interface(weights=("pretrained", LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1), weights_backbone=("pretrained_backbone", MobileNet_V3_Large_Weights.IMAGENET1K_V1))
def lraspp_mobilenet_v3_large(*, weights: Optional[LRASPP_MobileNet_V3_Large_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., weights_backbone: Optional[MobileNet_V3_Large_Weights] = ..., **kwargs: Any) -> LRASPP:
    """Constructs a Lite R-ASPP Network model with a MobileNetV3-Large backbone from
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_ paper.

    .. betastatus:: segmentation module

    Args:
        weights (:class:`~torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background).
        aux_loss (bool, optional): If True, it uses an auxiliary loss.
        weights_backbone (:class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The pretrained
            weights for the backbone.
        **kwargs: parameters passed to the ``torchvision.models.segmentation.LRASPP``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/lraspp.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights
        :members:
    """
    ...
