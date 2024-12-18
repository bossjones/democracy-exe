"""
This type stub file was generated by pyright.
"""

import torch
from typing import Any, Callable, Dict, List, Optional, Tuple
from torch import Tensor, nn
from .._api import WeightsEnum, register_model
from .._utils import handle_legacy_interface
from ..resnet import ResNet50_Weights
from .anchor_utils import AnchorGenerator

__all__ = ["FCOS", "FCOS_ResNet50_FPN_Weights", "fcos_resnet50_fpn"]
class FCOSHead(nn.Module):
    """
    A regression and classification head for use in FCOS.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer of head. Default: 4.
    """
    __annotations__ = ...
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int, num_convs: Optional[int] = ...) -> None:
        ...
    
    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor], matched_idxs: List[Tensor]) -> Dict[str, Tensor]:
        ...
    
    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        ...
    


class FCOSClassificationHead(nn.Module):
    """
    A classification head for use in FCOS.

    Args:
        in_channels (int): number of channels of the input feature.
        num_anchors (int): number of anchors to be predicted.
        num_classes (int): number of classes to be predicted.
        num_convs (Optional[int]): number of conv layer. Default: 4.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
        norm_layer: Module specifying the normalization layer to use.
    """
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int, num_convs: int = ..., prior_probability: float = ..., norm_layer: Optional[Callable[..., nn.Module]] = ...) -> None:
        ...
    
    def forward(self, x: List[Tensor]) -> Tensor:
        ...
    


class FCOSRegressionHead(nn.Module):
    """
    A regression head for use in FCOS, which combines regression branch and center-ness branch.
    This can obtain better performance.

    Reference: `FCOS: A simple and strong anchor-free object detector <https://arxiv.org/abs/2006.09214>`_.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 4.
        norm_layer: Module specifying the normalization layer to use.
    """
    def __init__(self, in_channels: int, num_anchors: int, num_convs: int = ..., norm_layer: Optional[Callable[..., nn.Module]] = ...) -> None:
        ...
    
    def forward(self, x: List[Tensor]) -> Tuple[Tensor, Tensor]:
        ...
    


class FCOS(nn.Module):
    """
    Implements FCOS.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or an OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
        min_size (int): Images are rescaled before feeding them to the backbone:
            we attempt to preserve the aspect ratio and scale the shorter edge
            to ``min_size``. If the resulting longer edge exceeds ``max_size``,
            then downscale so that the longer edge does not exceed ``max_size``.
            This may result in the shorter edge beeing lower than ``min_size``.
        max_size (int): See ``min_size``.
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps. For FCOS, only set one anchor for per position of each level, the width and height equal to
            the stride of feature map, and set aspect ratio = 1.0, so the center of anchor is equivalent to the point
            in FCOS paper.
        head (nn.Module): Module run on top of the feature pyramid.
            Defaults to a module containing a classification and regression module.
        center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        topk_candidates (int): Number of best detections to keep before NMS.

    Example:

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import FCOS
        >>> from torchvision.models.detection.anchor_utils import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        >>> # FCOS needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280,
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the network generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(
        >>>     sizes=((8,), (16,), (32,), (64,), (128,)),
        >>>     aspect_ratios=((1.0,),)
        >>> )
        >>>
        >>> # put the pieces together inside a FCOS model
        >>> model = FCOS(
        >>>     backbone,
        >>>     num_classes=80,
        >>>     anchor_generator=anchor_generator,
        >>> )
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """
    __annotations__ = ...
    def __init__(self, backbone: nn.Module, num_classes: int, min_size: int = ..., max_size: int = ..., image_mean: Optional[List[float]] = ..., image_std: Optional[List[float]] = ..., anchor_generator: Optional[AnchorGenerator] = ..., head: Optional[nn.Module] = ..., center_sampling_radius: float = ..., score_thresh: float = ..., nms_thresh: float = ..., detections_per_img: int = ..., topk_candidates: int = ..., **kwargs) -> None:
        ...
    
    @torch.jit.unused
    def eager_outputs(self, losses: Dict[str, Tensor], detections: List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        ...
    
    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor], num_anchors_per_level: List[int]) -> Dict[str, Tensor]:
        ...
    
    def postprocess_detections(self, head_outputs: Dict[str, List[Tensor]], anchors: List[List[Tensor]], image_shapes: List[Tuple[int, int]]) -> List[Dict[str, Tensor]]:
        ...
    
    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = ...) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        ...
    


class FCOS_ResNet50_FPN_Weights(WeightsEnum):
    COCO_V1 = ...
    DEFAULT = ...


@register_model()
@handle_legacy_interface(weights=("pretrained", FCOS_ResNet50_FPN_Weights.COCO_V1), weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1))
def fcos_resnet50_fpn(*, weights: Optional[FCOS_ResNet50_FPN_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., weights_backbone: Optional[ResNet50_Weights] = ..., trainable_backbone_layers: Optional[int] = ..., **kwargs: Any) -> FCOS:
    """
    Constructs a FCOS model with a ResNet-50-FPN backbone.

    .. betastatus:: detection module

    Reference: `FCOS: Fully Convolutional One-Stage Object Detection <https://arxiv.org/abs/1904.01355>`_.
               `FCOS: A simple and strong anchor-free object detector <https://arxiv.org/abs/2006.09214>`_.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each detection
        - scores (``Tensor[N]``): the scores of each detection

    For more details on the output, you may refer to :ref:`instance_seg_output`.

    Example:

        >>> model = torchvision.models.detection.fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        weights (:class:`~torchvision.models.detection.FCOS_ResNet50_FPN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.detection.FCOS_ResNet50_FPN_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int, optional): number of output classes of the model (including the background)
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The pretrained weights for
            the backbone.
        trainable_backbone_layers (int, optional): number of trainable (not frozen) resnet layers starting
            from final block. Valid values are between 0 and 5, with 5 meaning all backbone layers are
            trainable. If ``None`` is passed (the default) this value is set to 3. Default: None
        **kwargs: parameters passed to the ``torchvision.models.detection.FCOS``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/fcos.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.detection.FCOS_ResNet50_FPN_Weights
        :members:
    """
    ...

