"""
This type stub file was generated by pyright.
"""

from typing import List, Optional

import torch
from torch import Tensor, nn

from .image_list import ImageList

class AnchorGenerator(nn.Module):
    """
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map. This module assumes aspect ratio = height / width for
    each anchor.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Args:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """
    __annotations__ = ...
    def __init__(self, sizes=..., aspect_ratios=...) -> None:
        ...

    def generate_anchors(self, scales: List[int], aspect_ratios: List[float], dtype: torch.dtype = ..., device: torch.device = ...) -> Tensor:
        ...

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device): # -> None:
        ...

    def num_anchors_per_location(self) -> List[int]:
        ...

    def grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[Tensor]]) -> List[Tensor]:
        ...

    def forward(self, image_list: ImageList, feature_maps: List[Tensor]) -> List[Tensor]:
        ...



class DefaultBoxGenerator(nn.Module):
    """
    This module generates the default boxes of SSD for a set of feature maps and image sizes.

    Args:
        aspect_ratios (List[List[int]]): A list with all the aspect ratios used in each feature map.
        min_ratio (float): The minimum scale :math:`\text{s}_{\text{min}}` of the default boxes used in the estimation
            of the scales of each feature map. It is used only if the ``scales`` parameter is not provided.
        max_ratio (float): The maximum scale :math:`\text{s}_{\text{max}}`  of the default boxes used in the estimation
            of the scales of each feature map. It is used only if the ``scales`` parameter is not provided.
        scales (List[float]], optional): The scales of the default boxes. If not provided it will be estimated using
            the ``min_ratio`` and ``max_ratio`` parameters.
        steps (List[int]], optional): It's a hyper-parameter that affects the tiling of default boxes. If not provided
            it will be estimated from the data.
        clip (bool): Whether the standardized values of default boxes should be clipped between 0 and 1. The clipping
            is applied while the boxes are encoded in format ``(cx, cy, w, h)``.
    """
    def __init__(self, aspect_ratios: List[List[int]], min_ratio: float = ..., max_ratio: float = ..., scales: Optional[List[float]] = ..., steps: Optional[List[int]] = ..., clip: bool = ...) -> None:
        ...

    def num_anchors_per_location(self) -> List[int]:
        ...

    def __repr__(self) -> str:
        ...

    def forward(self, image_list: ImageList, feature_maps: List[Tensor]) -> List[Tensor]:
        ...
