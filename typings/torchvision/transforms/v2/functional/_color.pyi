"""
This type stub file was generated by pyright.
"""

from typing import List

import PIL.Image
import torch
from torchvision import tv_tensors

from ._utils import _register_kernel_internal

def rgb_to_grayscale(inpt: torch.Tensor, num_output_channels: int = ...) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.Grayscale` for details."""
    ...

to_grayscale = ...
@_register_kernel_internal(rgb_to_grayscale, torch.Tensor)
@_register_kernel_internal(rgb_to_grayscale, tv_tensors.Image)
def rgb_to_grayscale_image(image: torch.Tensor, num_output_channels: int = ...) -> torch.Tensor:
    ...

def grayscale_to_rgb(inpt: torch.Tensor) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.GrayscaleToRgb` for details."""
    ...

@_register_kernel_internal(grayscale_to_rgb, torch.Tensor)
@_register_kernel_internal(grayscale_to_rgb, tv_tensors.Image)
def grayscale_to_rgb_image(image: torch.Tensor) -> torch.Tensor:
    ...

@_register_kernel_internal(grayscale_to_rgb, PIL.Image.Image)
def grayscale_to_rgb_image_pil(image: PIL.Image.Image) -> PIL.Image.Image:
    ...

def adjust_brightness(inpt: torch.Tensor, brightness_factor: float) -> torch.Tensor:
    """Adjust brightness."""
    ...

@_register_kernel_internal(adjust_brightness, torch.Tensor)
@_register_kernel_internal(adjust_brightness, tv_tensors.Image)
def adjust_brightness_image(image: torch.Tensor, brightness_factor: float) -> torch.Tensor:
    ...

@_register_kernel_internal(adjust_brightness, tv_tensors.Video)
def adjust_brightness_video(video: torch.Tensor, brightness_factor: float) -> torch.Tensor:
    ...

def adjust_saturation(inpt: torch.Tensor, saturation_factor: float) -> torch.Tensor:
    """Adjust saturation."""
    ...

@_register_kernel_internal(adjust_saturation, torch.Tensor)
@_register_kernel_internal(adjust_saturation, tv_tensors.Image)
def adjust_saturation_image(image: torch.Tensor, saturation_factor: float) -> torch.Tensor:
    ...

_adjust_saturation_image_pil = ...
@_register_kernel_internal(adjust_saturation, tv_tensors.Video)
def adjust_saturation_video(video: torch.Tensor, saturation_factor: float) -> torch.Tensor:
    ...

def adjust_contrast(inpt: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    """See :class:`~torchvision.transforms.RandomAutocontrast`"""
    ...

@_register_kernel_internal(adjust_contrast, torch.Tensor)
@_register_kernel_internal(adjust_contrast, tv_tensors.Image)
def adjust_contrast_image(image: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    ...

_adjust_contrast_image_pil = ...
@_register_kernel_internal(adjust_contrast, tv_tensors.Video)
def adjust_contrast_video(video: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    ...

def adjust_sharpness(inpt: torch.Tensor, sharpness_factor: float) -> torch.Tensor:
    """See :class:`~torchvision.transforms.RandomAdjustSharpness`"""
    ...

@_register_kernel_internal(adjust_sharpness, torch.Tensor)
@_register_kernel_internal(adjust_sharpness, tv_tensors.Image)
def adjust_sharpness_image(image: torch.Tensor, sharpness_factor: float) -> torch.Tensor:
    ...

_adjust_sharpness_image_pil = ...
@_register_kernel_internal(adjust_sharpness, tv_tensors.Video)
def adjust_sharpness_video(video: torch.Tensor, sharpness_factor: float) -> torch.Tensor:
    ...

def adjust_hue(inpt: torch.Tensor, hue_factor: float) -> torch.Tensor:
    """Adjust hue"""
    ...

@_register_kernel_internal(adjust_hue, torch.Tensor)
@_register_kernel_internal(adjust_hue, tv_tensors.Image)
def adjust_hue_image(image: torch.Tensor, hue_factor: float) -> torch.Tensor:
    ...

_adjust_hue_image_pil = ...
@_register_kernel_internal(adjust_hue, tv_tensors.Video)
def adjust_hue_video(video: torch.Tensor, hue_factor: float) -> torch.Tensor:
    ...

def adjust_gamma(inpt: torch.Tensor, gamma: float, gain: float = ...) -> torch.Tensor:
    """Adjust gamma."""
    ...

@_register_kernel_internal(adjust_gamma, torch.Tensor)
@_register_kernel_internal(adjust_gamma, tv_tensors.Image)
def adjust_gamma_image(image: torch.Tensor, gamma: float, gain: float = ...) -> torch.Tensor:
    ...

_adjust_gamma_image_pil = ...
@_register_kernel_internal(adjust_gamma, tv_tensors.Video)
def adjust_gamma_video(video: torch.Tensor, gamma: float, gain: float = ...) -> torch.Tensor:
    ...

def posterize(inpt: torch.Tensor, bits: int) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.RandomPosterize` for details."""
    ...

@_register_kernel_internal(posterize, torch.Tensor)
@_register_kernel_internal(posterize, tv_tensors.Image)
def posterize_image(image: torch.Tensor, bits: int) -> torch.Tensor:
    ...

_posterize_image_pil = ...
@_register_kernel_internal(posterize, tv_tensors.Video)
def posterize_video(video: torch.Tensor, bits: int) -> torch.Tensor:
    ...

def solarize(inpt: torch.Tensor, threshold: float) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.RandomSolarize` for details."""
    ...

@_register_kernel_internal(solarize, torch.Tensor)
@_register_kernel_internal(solarize, tv_tensors.Image)
def solarize_image(image: torch.Tensor, threshold: float) -> torch.Tensor:
    ...

_solarize_image_pil = ...
@_register_kernel_internal(solarize, tv_tensors.Video)
def solarize_video(video: torch.Tensor, threshold: float) -> torch.Tensor:
    ...

def autocontrast(inpt: torch.Tensor) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.RandomAutocontrast` for details."""
    ...

@_register_kernel_internal(autocontrast, torch.Tensor)
@_register_kernel_internal(autocontrast, tv_tensors.Image)
def autocontrast_image(image: torch.Tensor) -> torch.Tensor:
    ...

_autocontrast_image_pil = ...
@_register_kernel_internal(autocontrast, tv_tensors.Video)
def autocontrast_video(video: torch.Tensor) -> torch.Tensor:
    ...

def equalize(inpt: torch.Tensor) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.RandomEqualize` for details."""
    ...

@_register_kernel_internal(equalize, torch.Tensor)
@_register_kernel_internal(equalize, tv_tensors.Image)
def equalize_image(image: torch.Tensor) -> torch.Tensor:
    ...

_equalize_image_pil = ...
@_register_kernel_internal(equalize, tv_tensors.Video)
def equalize_video(video: torch.Tensor) -> torch.Tensor:
    ...

def invert(inpt: torch.Tensor) -> torch.Tensor:
    """See :func:`~torchvision.transforms.v2.RandomInvert`."""
    ...

@_register_kernel_internal(invert, torch.Tensor)
@_register_kernel_internal(invert, tv_tensors.Image)
def invert_image(image: torch.Tensor) -> torch.Tensor:
    ...

_invert_image_pil = ...
@_register_kernel_internal(invert, tv_tensors.Video)
def invert_video(video: torch.Tensor) -> torch.Tensor:
    ...

def permute_channels(inpt: torch.Tensor, permutation: List[int]) -> torch.Tensor:
    """Permute the channels of the input according to the given permutation.

    This function supports plain :class:`~torch.Tensor`'s, :class:`PIL.Image.Image`'s, and
    :class:`torchvision.tv_tensors.Image` and :class:`torchvision.tv_tensors.Video`.

    Example:
        >>> rgb_image = torch.rand(3, 256, 256)
        >>> bgr_image = F.permutate_channels(rgb_image, permutation=[2, 1, 0])

    Args:
        permutation (List[int]): Valid permutation of the input channel indices. The index of the element determines the
            channel index in the input and the value determines the channel index in the output. For example,
            ``permutation=[2, 0 , 1]``

            - takes ``ìnpt[..., 0, :, :]`` and puts it at ``output[..., 2, :, :]``,
            - takes ``ìnpt[..., 1, :, :]`` and puts it at ``output[..., 0, :, :]``, and
            - takes ``ìnpt[..., 2, :, :]`` and puts it at ``output[..., 1, :, :]``.

    Raises:
        ValueError: If ``len(permutation)`` doesn't match the number of channels in the input.
    """
    ...

@_register_kernel_internal(permute_channels, torch.Tensor)
@_register_kernel_internal(permute_channels, tv_tensors.Image)
def permute_channels_image(image: torch.Tensor, permutation: List[int]) -> torch.Tensor:
    ...

@_register_kernel_internal(permute_channels, tv_tensors.Video)
def permute_channels_video(video: torch.Tensor, permutation: List[int]) -> torch.Tensor:
    ...
