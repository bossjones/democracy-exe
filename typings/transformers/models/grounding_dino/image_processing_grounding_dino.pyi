"""
This type stub file was generated by pyright.
"""

import pathlib
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import ExplicitEnum, TensorType, is_scipy_available, is_torch_available, is_vision_available

"""Image processor class for Deformable DETR."""
if is_torch_available():
    ...
if is_vision_available():
    ...
if is_scipy_available():
    ...
logger = ...
AnnotationType = Dict[str, Union[int, str, List[Dict]]]
class AnnotationFormat(ExplicitEnum):
    COCO_DETECTION = ...
    COCO_PANOPTIC = ...


SUPPORTED_ANNOTATION_FORMATS = ...
def get_size_with_aspect_ratio(image_size, size, max_size=...) -> Tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size.

    Args:
        image_size (`Tuple[int, int]`):
            The input image size.
        size (`int`):
            The desired output size.
        max_size (`int`, *optional*):
            The maximum allowed output size.
    """
    ...

def get_resize_output_image_size(input_image: np.ndarray, size: Union[int, Tuple[int, int], List[int]], max_size: Optional[int] = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ...) -> Tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size. If the desired output size
    is a tuple or list, the output image size is returned as is. If the desired output size is an integer, the output
    image size is computed by keeping the aspect ratio of the input image size.

    Args:
        input_image (`np.ndarray`):
            The image to resize.
        size (`int` or `Tuple[int, int]` or `List[int]`):
            The desired output size.
        max_size (`int`, *optional*):
            The maximum allowed output size.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.
    """
    ...

def get_numpy_to_framework_fn(arr) -> Callable:
    """
    Returns a function that converts a numpy array to the framework of the input array.

    Args:
        arr (`np.ndarray`): The array to convert.
    """
    ...

def safe_squeeze(arr: np.ndarray, axis: Optional[int] = ...) -> np.ndarray:
    """
    Squeezes an array, but only if the axis specified has dim 1.
    """
    ...

def normalize_annotation(annotation: Dict, image_size: Tuple[int, int]) -> Dict:
    ...

def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    ...

def get_max_height_width(images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = ...) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    ...

def make_pixel_mask(image: np.ndarray, output_size: Tuple[int, int], input_data_format: Optional[Union[str, ChannelDimension]] = ...) -> np.ndarray:
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

    Args:
        image (`np.ndarray`):
            Image to make the pixel mask for.
        output_size (`Tuple[int, int]`):
            Output size of the mask.
    """
    ...

def convert_coco_poly_to_mask(segmentations, height: int, width: int) -> np.ndarray:
    """
    Convert a COCO polygon annotation to a mask.

    Args:
        segmentations (`List[List[float]]`):
            List of polygons, each polygon represented by a list of x-y coordinates.
        height (`int`):
            Height of the mask.
        width (`int`):
            Width of the mask.
    """
    ...

def prepare_coco_detection_annotation(image, target, return_segmentation_masks: bool = ..., input_data_format: Optional[Union[ChannelDimension, str]] = ...): # -> dict[Any, Any]:
    """
    Convert the target in COCO format into the format expected by GroundingDino.
    """
    ...

def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """
    Compute the bounding boxes around the provided panoptic segmentation masks.

    Args:
        masks: masks in format `[number_masks, height, width]` where N is the number of masks

    Returns:
        boxes: bounding boxes in format `[number_masks, 4]` in xyxy format
    """
    ...

def prepare_coco_panoptic_annotation(image: np.ndarray, target: Dict, masks_path: Union[str, pathlib.Path], return_masks: bool = ..., input_data_format: Union[ChannelDimension, str] = ...) -> Dict:
    """
    Prepare a coco panoptic annotation for GroundingDino.
    """
    ...

def get_segmentation_image(masks: np.ndarray, input_size: Tuple, target_size: Tuple, stuff_equiv_classes, deduplicate=...): # -> ndarray[Any, Any]:
    ...

def get_mask_area(seg_img: np.ndarray, target_size: Tuple[int, int], n_classes: int) -> np.ndarray:
    ...

def score_labels_from_class_probabilities(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ...

def post_process_panoptic_sample(out_logits: np.ndarray, masks: np.ndarray, boxes: np.ndarray, processed_size: Tuple[int, int], target_size: Tuple[int, int], is_thing_map: Dict, threshold=...) -> Dict:
    """
    Converts the output of [`DetrForSegmentation`] into panoptic segmentation predictions for a single sample.

    Args:
        out_logits (`torch.Tensor`):
            The logits for this sample.
        masks (`torch.Tensor`):
            The predicted segmentation masks for this sample.
        boxes (`torch.Tensor`):
            The prediced bounding boxes for this sample. The boxes are in the normalized format `(center_x, center_y,
            width, height)` and values between `[0, 1]`, relative to the size the image (disregarding padding).
        processed_size (`Tuple[int, int]`):
            The processed size of the image `(height, width)`, as returned by the preprocessing step i.e. the size
            after data augmentation but before batching.
        target_size (`Tuple[int, int]`):
            The target size of the image, `(height, width)` corresponding to the requested final size of the
            prediction.
        is_thing_map (`Dict`):
            A dictionary mapping class indices to a boolean value indicating whether the class is a thing or not.
        threshold (`float`, *optional*, defaults to 0.85):
            The threshold used to binarize the segmentation masks.
    """
    ...

def resize_annotation(annotation: Dict[str, Any], orig_size: Tuple[int, int], target_size: Tuple[int, int], threshold: float = ..., resample: PILImageResampling = ...): # -> dict[Any, Any]:
    """
    Resizes an annotation to a target size.

    Args:
        annotation (`Dict[str, Any]`):
            The annotation dictionary.
        orig_size (`Tuple[int, int]`):
            The original size of the input image.
        target_size (`Tuple[int, int]`):
            The target size of the image, as returned by the preprocessing `resize` step.
        threshold (`float`, *optional*, defaults to 0.5):
            The threshold used to binarize the segmentation masks.
        resample (`PILImageResampling`, defaults to `PILImageResampling.NEAREST`):
            The resampling filter to use when resizing the masks.
    """
    ...

def binary_mask_to_rle(mask): # -> list[Any]:
    """
    Converts given binary mask of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        mask (`torch.Tensor` or `numpy.array`):
            A binary mask tensor of shape `(height, width)` where 0 denotes background and 1 denotes the target
            segment_id or class_id.
    Returns:
        `List`: Run-length encoded list of the binary mask. Refer to COCO API for more information about the RLE
        format.
    """
    ...

def convert_segmentation_to_rle(segmentation): # -> list[Any]:
    """
    Converts given segmentation map of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        segmentation (`torch.Tensor` or `numpy.array`):
            A segmentation map of shape `(height, width)` where each value denotes a segment or class id.
    Returns:
        `List[List]`: A list of lists, where each list is the run-length encoding of a segment / class id.
    """
    ...

def remove_low_and_no_objects(masks, scores, labels, object_mask_threshold, num_labels): # -> tuple[Any, Any, Any]:
    """
    Binarize the given masks using `object_mask_threshold`, it returns the associated values of `masks`, `scores` and
    `labels`.

    Args:
        masks (`torch.Tensor`):
            A tensor of shape `(num_queries, height, width)`.
        scores (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        labels (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        object_mask_threshold (`float`):
            A number between 0 and 1 used to binarize the masks.
    Raises:
        `ValueError`: Raised when the first dimension doesn't match in all input tensors.
    Returns:
        `Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]`: The `masks`, `scores` and `labels` without the region
        < `object_mask_threshold`.
    """
    ...

def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=..., overlap_mask_area_threshold=...): # -> tuple[Any | Literal[False], Any]:
    ...

def compute_segments(mask_probs, pred_scores, pred_labels, mask_threshold: float = ..., overlap_mask_area_threshold: float = ..., label_ids_to_fuse: Optional[Set[int]] = ..., target_size: Tuple[int, int] = ...): # -> tuple[Tensor, list[Dict[Any, Any]]]:
    ...

class GroundingDinoImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Grounding DINO image processor.

    Args:
        format (`str`, *optional*, defaults to `AnnotationFormat.COCO_DETECTION`):
            Data format of the annotations. One of "coco_detection" or "coco_panoptic".
        do_resize (`bool`, *optional*, defaults to `True`):
            Controls whether to resize the image's (height, width) dimensions to the specified `size`. Can be
            overridden by the `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 800, "longest_edge": 1333}`):
            Size of the image's (height, width) dimensions after resizing. Can be overridden by the `size` parameter in
            the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Controls whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method. Controls whether to normalize the image. Can be overridden by the `do_normalize`
            parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean values to use when normalizing the image. Can be a single value or a list of values, one for each
            channel. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Standard deviation values to use when normalizing the image. Can be a single value or a list of values, one
            for each channel. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_annotations (`bool`, *optional*, defaults to `True`):
            Controls whether to convert the annotations to the format expected by the DETR model. Converts the
            bounding boxes to the format `(center_x, center_y, width, height)` and in the range `[0, 1]`.
            Can be overridden by the `do_convert_annotations` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Controls whether to pad the image to the largest image in a batch and create a pixel mask. Can be
            overridden by the `do_pad` parameter in the `preprocess` method.
    """
    model_input_names = ...
    def __init__(self, format: Union[str, AnnotationFormat] = ..., do_resize: bool = ..., size: Dict[str, int] = ..., resample: PILImageResampling = ..., do_rescale: bool = ..., rescale_factor: Union[int, float] = ..., do_normalize: bool = ..., image_mean: Union[float, List[float]] = ..., image_std: Union[float, List[float]] = ..., do_convert_annotations: Optional[bool] = ..., do_pad: bool = ..., **kwargs) -> None:
        ...

    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs): # -> tuple[Self, dict[str, Any]] | Self:
        """
        Overrides the `from_dict` method from the base class to make sure parameters are updated if image processor is
        created using from_dict and kwargs e.g. `GroundingDinoImageProcessor.from_pretrained(checkpoint, size=600,
        max_size=800)`
        """
        ...

    def prepare_annotation(self, image: np.ndarray, target: Dict, format: Optional[AnnotationFormat] = ..., return_segmentation_masks: bool = ..., masks_path: Optional[Union[str, pathlib.Path]] = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ...) -> Dict:
        """
        Prepare an annotation for feeding into GroundingDino model.
        """
        ...

    def prepare(self, image, target, return_segmentation_masks=..., masks_path=...): # -> tuple[Any, Dict[Any, Any]]:
        ...

    def convert_coco_poly_to_mask(self, *args, **kwargs): # -> ndarray[Any, Any]:
        ...

    def prepare_coco_detection(self, *args, **kwargs): # -> dict[Any, Any]:
        ...

    def prepare_coco_panoptic(self, *args, **kwargs): # -> Dict[Any, Any]:
        ...

    def resize(self, image: np.ndarray, size: Dict[str, int], resample: PILImageResampling = ..., data_format: Optional[ChannelDimension] = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ..., **kwargs) -> np.ndarray:
        """
        Resize the image to the given size. Size can be `min_size` (scalar) or `(height, width)` tuple. If size is an
        int, smaller edge of the image will be matched to this number.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary containing the size to resize to. Can contain the keys `shortest_edge` and `longest_edge` or
                `height` and `width`.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use if resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        ...

    def resize_annotation(self, annotation, orig_size, size, resample: PILImageResampling = ...) -> Dict:
        """
        Resize the annotation to match the resized image. If size is an int, smaller edge of the mask will be matched
        to this number.
        """
        ...

    def rescale(self, image: np.ndarray, rescale_factor: float, data_format: Optional[Union[str, ChannelDimension]] = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ...) -> np.ndarray:
        """
        Rescale the image by the given factor. image = image * rescale_factor.

        Args:
            image (`np.ndarray`):
                Image to rescale.
            rescale_factor (`float`):
                The value to use for rescaling.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. If unset, is inferred from the input image. Can be
                one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """
        ...

    def normalize_annotation(self, annotation: Dict, image_size: Tuple[int, int]) -> Dict:
        """
        Normalize the boxes in the annotation from `[top_left_x, top_left_y, bottom_right_x, bottom_right_y]` to
        `[center_x, center_y, width, height]` format and from absolute to relative pixel values.
        """
        ...

    def pad(self, images: List[np.ndarray], annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = ..., constant_values: Union[float, Iterable[float]] = ..., return_pixel_mask: bool = ..., return_tensors: Optional[Union[str, TensorType]] = ..., data_format: Optional[ChannelDimension] = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ..., update_bboxes: bool = ...) -> BatchFeature:
        """
        Pads a batch of images to the bottom and right of the image with zeros to the size of largest height and width
        in the batch and optionally returns their corresponding pixel mask.

        Args:
            images (List[`np.ndarray`]):
                Images to pad.
            annotations (`AnnotationType` or `List[AnnotationType]`, *optional*):
                Annotations to transform according to the padding that is applied to the images.
            constant_values (`float` or `Iterable[float]`, *optional*):
                The value to use for the padding if `mode` is `"constant"`.
            return_pixel_mask (`bool`, *optional*, defaults to `True`):
                Whether to return a pixel mask.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
            update_bboxes (`bool`, *optional*, defaults to `True`):
                Whether to update the bounding boxes in the annotations to match the padded images. If the
                bounding boxes have not been converted to relative coordinates and `(centre_x, centre_y, width, height)`
                format, the bounding boxes will not be updated.
        """
        ...

    def preprocess(self, images: ImageInput, annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = ..., return_segmentation_masks: bool = ..., masks_path: Optional[Union[str, pathlib.Path]] = ..., do_resize: Optional[bool] = ..., size: Optional[Dict[str, int]] = ..., resample=..., do_rescale: Optional[bool] = ..., rescale_factor: Optional[Union[int, float]] = ..., do_normalize: Optional[bool] = ..., do_convert_annotations: Optional[bool] = ..., image_mean: Optional[Union[float, List[float]]] = ..., image_std: Optional[Union[float, List[float]]] = ..., do_pad: Optional[bool] = ..., format: Optional[Union[str, AnnotationFormat]] = ..., return_tensors: Optional[Union[TensorType, str]] = ..., data_format: Union[str, ChannelDimension] = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ..., **kwargs) -> BatchFeature:
        """
        Preprocess an image or a batch of images so that it can be used by the model.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects a single or batch of images with pixel values ranging
                from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            annotations (`AnnotationType` or `List[AnnotationType]`, *optional*):
                List of annotations associated with the image or batch of images. If annotation is for object
                detection, the annotations should be a dictionary with the following keys:
                - "image_id" (`int`): The image id.
                - "annotations" (`List[Dict]`): List of annotations for an image. Each annotation should be a
                  dictionary. An image can have no annotations, in which case the list should be empty.
                If annotation is for segmentation, the annotations should be a dictionary with the following keys:
                - "image_id" (`int`): The image id.
                - "segments_info" (`List[Dict]`): List of segments for an image. Each segment should be a dictionary.
                  An image can have no segments, in which case the list should be empty.
                - "file_name" (`str`): The file name of the image.
            return_segmentation_masks (`bool`, *optional*, defaults to self.return_segmentation_masks):
                Whether to return segmentation masks.
            masks_path (`str` or `pathlib.Path`, *optional*):
                Path to the directory containing the segmentation masks.
            do_resize (`bool`, *optional*, defaults to self.do_resize):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to self.size):
                Size of the image after resizing.
            resample (`PILImageResampling`, *optional*, defaults to self.resample):
                Resampling filter to use when resizing the image.
            do_rescale (`bool`, *optional*, defaults to self.do_rescale):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to self.rescale_factor):
                Rescale factor to use when rescaling the image.
            do_normalize (`bool`, *optional*, defaults to self.do_normalize):
                Whether to normalize the image.
            do_convert_annotations (`bool`, *optional*, defaults to self.do_convert_annotations):
                Whether to convert the annotations to the format expected by the model. Converts the bounding
                boxes from the format `(top_left_x, top_left_y, width, height)` to `(center_x, center_y, width, height)`
                and in relative coordinates.
            image_mean (`float` or `List[float]`, *optional*, defaults to self.image_mean):
                Mean to use when normalizing the image.
            image_std (`float` or `List[float]`, *optional*, defaults to self.image_std):
                Standard deviation to use when normalizing the image.
            do_pad (`bool`, *optional*, defaults to self.do_pad):
                Whether to pad the image. If `True` will pad the images in the batch to the largest image in the batch
                and create a pixel mask. Padding will be applied to the bottom and right of the image with zeros.
            format (`str` or `AnnotationFormat`, *optional*, defaults to self.format):
                Format of the annotations.
            return_tensors (`str` or `TensorType`, *optional*, defaults to self.return_tensors):
                Type of tensors to return. If `None`, will return the list of images.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        ...

    def post_process_object_detection(self, outputs, threshold: float = ..., target_sizes: Union[TensorType, List[Tuple]] = ...): # -> list[Any]:
        """
        Converts the raw output of [`GroundingDinoForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.

        Args:
            outputs ([`GroundingDinoObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        ...
