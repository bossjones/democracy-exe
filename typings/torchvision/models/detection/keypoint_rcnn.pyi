"""
This type stub file was generated by pyright.
"""

from typing import Any, Optional
from torch import nn
from .._api import WeightsEnum, register_model
from .._utils import handle_legacy_interface
from ..resnet import ResNet50_Weights
from .faster_rcnn import FasterRCNN

__all__ = ["KeypointRCNN", "KeypointRCNN_ResNet50_FPN_Weights", "keypointrcnn_resnet50_fpn"]
class KeypointRCNN(FasterRCNN):
    """
    Implements Keypoint R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
            ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - keypoints (FloatTensor[N, K, 3]): the K keypoints location for each of the N instances, in the
          format [x, y, visibility], where visibility=0 means that the keypoint is not visible.

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN, and the keypoint loss.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
            ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction
        - keypoints (FloatTensor[N, K, 3]): the locations of the predicted keypoints, in [x, y, v] format.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
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
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): only return proposals with an objectness score greater than rpn_score_thresh
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        keypoint_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
             the locations indicated by the bounding boxes, which will be used for the keypoint head.
        keypoint_head (nn.Module): module that takes the cropped feature maps as input
        keypoint_predictor (nn.Module): module that takes the output of the keypoint_head and returns the
            heatmap logits

    Example::

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import KeypointRCNN
        >>> from torchvision.models.detection.anchor_utils import AnchorGenerator
        >>>
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        >>> # KeypointRCNN needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280,
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the RPN generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
        >>>                                    aspect_ratios=((0.5, 1.0, 2.0),))
        >>>
        >>> # let's define what are the feature maps that we will
        >>> # use to perform the region of interest cropping, as well as
        >>> # the size of the crop after rescaling.
        >>> # if your backbone returns a Tensor, featmap_names is expected to
        >>> # be ['0']. More generally, the backbone should return an
        >>> # OrderedDict[Tensor], and in featmap_names you can choose which
        >>> # feature maps to use.
        >>> roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        >>>                                                 output_size=7,
        >>>                                                 sampling_ratio=2)
        >>>
        >>> keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        >>>                                                          output_size=14,
        >>>                                                          sampling_ratio=2)
        >>> # put the pieces together inside a KeypointRCNN model
        >>> model = KeypointRCNN(backbone,
        >>>                      num_classes=2,
        >>>                      rpn_anchor_generator=anchor_generator,
        >>>                      box_roi_pool=roi_pooler,
        >>>                      keypoint_roi_pool=keypoint_roi_pooler)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """
    def __init__(self, backbone, num_classes=..., min_size=..., max_size=..., image_mean=..., image_std=..., rpn_anchor_generator=..., rpn_head=..., rpn_pre_nms_top_n_train=..., rpn_pre_nms_top_n_test=..., rpn_post_nms_top_n_train=..., rpn_post_nms_top_n_test=..., rpn_nms_thresh=..., rpn_fg_iou_thresh=..., rpn_bg_iou_thresh=..., rpn_batch_size_per_image=..., rpn_positive_fraction=..., rpn_score_thresh=..., box_roi_pool=..., box_head=..., box_predictor=..., box_score_thresh=..., box_nms_thresh=..., box_detections_per_img=..., box_fg_iou_thresh=..., box_bg_iou_thresh=..., box_batch_size_per_image=..., box_positive_fraction=..., bbox_reg_weights=..., keypoint_roi_pool=..., keypoint_head=..., keypoint_predictor=..., num_keypoints=..., **kwargs) -> None:
        ...
    


class KeypointRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers) -> None:
        ...
    


class KeypointRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_keypoints) -> None:
        ...
    
    def forward(self, x):
        ...
    


_COMMON_META = ...
class KeypointRCNN_ResNet50_FPN_Weights(WeightsEnum):
    COCO_LEGACY = ...
    COCO_V1 = ...
    DEFAULT = ...


@register_model()
@handle_legacy_interface(weights=("pretrained", lambda kwargs: KeypointRCNN_ResNet50_FPN_Weights.COCO_LEGACY if kwargs["pretrained"] == "legacy" else KeypointRCNN_ResNet50_FPN_Weights.COCO_V1), weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1))
def keypointrcnn_resnet50_fpn(*, weights: Optional[KeypointRCNN_ResNet50_FPN_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., num_keypoints: Optional[int] = ..., weights_backbone: Optional[ResNet50_Weights] = ..., trainable_backbone_layers: Optional[int] = ..., **kwargs: Any) -> KeypointRCNN:
    """
    Constructs a Keypoint R-CNN model with a ResNet-50-FPN backbone.

    .. betastatus:: detection module

    Reference: `Mask R-CNN <https://arxiv.org/abs/1703.06870>`__.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
        - keypoints (``FloatTensor[N, K, 3]``): the ``K`` keypoints location for each of the ``N`` instances, in the
          format ``[x, y, visibility]``, where ``visibility=0`` means that the keypoint is not visible.

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN, and the keypoint loss.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detected instances:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each instance
        - scores (``Tensor[N]``): the scores or each instance
        - keypoints (``FloatTensor[N, K, 3]``): the locations of the predicted keypoints, in ``[x, y, v]`` format.

    For more details on the output, you may refer to :ref:`instance_seg_output`.

    Keypoint R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.

    Example::

        >>> model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "keypoint_rcnn.onnx", opset_version = 11)

    Args:
        weights (:class:`~torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int, optional): number of output classes of the model (including the background)
        num_keypoints (int, optional): number of keypoints
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights for the backbone.
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. If ``None`` is
            passed (the default) this value is set to 3.

    .. autoclass:: torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights
        :members:
    """
    ...

