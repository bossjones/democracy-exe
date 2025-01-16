"""
This type stub file was generated by pyright.
"""

import numpy as np
import PIL
from typing import Dict, Iterable, List, Optional, Tuple, Union
from ...image_processing_utils import BaseImageProcessor
from ...image_transforms import PaddingMode
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType, filter_out_non_signature_kwargs, is_torch_available, is_vision_available

"""Image processor class for ZoeDepth."""
if is_vision_available():
    ...
if is_torch_available():
    ...
logger = ...
def get_resize_output_image_size(input_image: np.ndarray, output_size: Union[int, Iterable[int]], keep_aspect_ratio: bool, multiple: int, input_data_format: Optional[Union[str, ChannelDimension]] = ...) -> Tuple[int, int]:
    ...

class ZoeDepthImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ZoeDepth image processor.

    Args:
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to apply pad the input.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overidden by `do_rescale` in
            `preprocess`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overidden by `rescale_factor` in `preprocess`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions. Can be overidden by `do_resize` in `preprocess`.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 384, "width": 512}`):
            Size of the image after resizing. Size of the image after resizing. If `keep_aspect_ratio` is `True`,
            the image is resized by choosing the smaller of the height and width scaling factors and using it for both dimensions.
            If `ensure_multiple_of` is also set, the image is further resized to a size that is a multiple of this value.
            Can be overidden by `size` in `preprocess`.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Defines the resampling filter to use if resizing the image. Can be overidden by `resample` in `preprocess`.
        keep_aspect_ratio (`bool`, *optional*, defaults to `True`):
            If `True`, the image is resized by choosing the smaller of the height and width scaling factors and using it for
            both dimensions. This ensures that the image is scaled down as little as possible while still fitting within the
            desired output size. In case `ensure_multiple_of` is also set, the image is further resized to a size that is a
            multiple of this value by flooring the height and width to the nearest multiple of this value.
            Can be overidden by `keep_aspect_ratio` in `preprocess`.
        ensure_multiple_of (`int`, *optional*, defaults to 32):
            If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Works by flooring
            the height and width to the nearest multiple of this value.

            Works both with and without `keep_aspect_ratio` being set to `True`. Can be overidden by `ensure_multiple_of`
            in `preprocess`.
    """
    model_input_names = ...
    def __init__(self, do_pad: bool = ..., do_rescale: bool = ..., rescale_factor: Union[int, float] = ..., do_normalize: bool = ..., image_mean: Optional[Union[float, List[float]]] = ..., image_std: Optional[Union[float, List[float]]] = ..., do_resize: bool = ..., size: Dict[str, int] = ..., resample: PILImageResampling = ..., keep_aspect_ratio: bool = ..., ensure_multiple_of: int = ..., **kwargs) -> None:
        ...
    
    def resize(self, image: np.ndarray, size: Dict[str, int], keep_aspect_ratio: bool = ..., ensure_multiple_of: int = ..., resample: PILImageResampling = ..., data_format: Optional[Union[str, ChannelDimension]] = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ...) -> np.ndarray:
        """
        Resize an image to target size `(size["height"], size["width"])`. If `keep_aspect_ratio` is `True`, the image
        is resized to the largest possible size such that the aspect ratio is preserved. If `ensure_multiple_of` is
        set, the image is resized to a size that is a multiple of this value.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Target size of the output image.
            keep_aspect_ratio (`bool`, *optional*, defaults to `False`):
                If `True`, the image is resized to the largest possible size such that the aspect ratio is preserved.
            ensure_multiple_of (`int`, *optional*, defaults to 1):
                The image is resized to a size that is a multiple of this value.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Defines the resampling filter to use if resizing the image. Otherwise, the image is resized to size
                specified in `size`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        ...
    
    def pad_image(self, image: np.array, mode: PaddingMode = ..., data_format: Optional[Union[str, ChannelDimension]] = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ...): # -> ndarray[Any, Any]:
        """
        Pad an image as done in the original ZoeDepth implementation.

        Padding fixes the boundary artifacts in the output depth map.
        Boundary artifacts are sometimes caused by the fact that the model is trained on NYU raw dataset
        which has a black or white border around the image. This function pads the input image and crops
        the prediction back to the original size / view.

        Args:
            image (`np.ndarray`):
                Image to pad.
            mode (`PaddingMode`):
                The padding mode to use. Can be one of:
                    - `"constant"`: pads with a constant value.
                    - `"reflect"`: pads with the reflection of the vector mirrored on the first and last values of the
                    vector along each axis.
                    - `"replicate"`: pads with the replication of the last value on the edge of the array along each axis.
                    - `"symmetric"`: pads with the reflection of the vector mirrored along the edge of the array.
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
    
    @filter_out_non_signature_kwargs()
    def preprocess(self, images: ImageInput, do_pad: bool = ..., do_rescale: bool = ..., rescale_factor: float = ..., do_normalize: bool = ..., image_mean: Optional[Union[float, List[float]]] = ..., image_std: Optional[Union[float, List[float]]] = ..., do_resize: bool = ..., size: int = ..., keep_aspect_ratio: bool = ..., ensure_multiple_of: int = ..., resample: PILImageResampling = ..., return_tensors: Optional[Union[str, TensorType]] = ..., data_format: ChannelDimension = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ...) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the input image.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. If `keep_aspect_ratio` is `True`, he image is resized by choosing the smaller of
                the height and width scaling factors and using it for both dimensions. If `ensure_multiple_of` is also set,
                the image is further resized to a size that is a multiple of this value.
            keep_aspect_ratio (`bool`, *optional*, defaults to `self.keep_aspect_ratio`):
                If `True` and `do_resize=True`, the image is resized by choosing the smaller of the height and width scaling factors and using it for
                both dimensions. This ensures that the image is scaled down as little as possible while still fitting within the
                desired output size. In case `ensure_multiple_of` is also set, the image is further resized to a size that is a
                multiple of this value by flooring the height and width to the nearest multiple of this value.
            ensure_multiple_of (`int`, *optional*, defaults to `self.ensure_multiple_of`):
                If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Works by flooring
                the height and width to the nearest multiple of this value.

                Works both with and without `keep_aspect_ratio` being set to `True`. Can be overidden by `ensure_multiple_of` in `preprocess`.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
                has an effect if `do_resize` is set to `True`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        ...
    


