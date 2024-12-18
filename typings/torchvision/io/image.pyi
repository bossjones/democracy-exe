"""
This type stub file was generated by pyright.
"""

import torch
from enum import Enum
from typing import List, Union

class ImageReadMode(Enum):
    """Allow automatic conversion to RGB, RGBA, etc while decoding.

    .. note::

        You don't need to use this struct, you can just pass strings to all
        ``mode`` parameters, e.g. ``mode="RGB"``.

    The different available modes are the following.

    - UNCHANGED: loads the image as-is
    - RGB: converts to RGB
    - RGBA: converts to RGB with transparency (also aliased as RGB_ALPHA)
    - GRAY: converts to grayscale
    - GRAY_ALPHA: converts to grayscale with transparency

    .. note::

        Some decoders won't support all possible values, e.g. GRAY and
        GRAY_ALPHA are only supported for PNG and JPEG images.
    """
    UNCHANGED = ...
    GRAY = ...
    GRAY_ALPHA = ...
    RGB = ...
    RGB_ALPHA = ...
    RGBA = ...


def read_file(path: str) -> torch.Tensor:
    """
    Return the bytes contents of a file as a uint8 1D Tensor.

    Args:
        path (str or ``pathlib.Path``): the path to the file to be read

    Returns:
        data (Tensor)
    """
    ...

def write_file(filename: str, data: torch.Tensor) -> None:
    """
    Write the content of an uint8 1D tensor to a file.

    Args:
        filename (str or ``pathlib.Path``): the path to the file to be written
        data (Tensor): the contents to be written to the output file
    """
    ...

def decode_png(input: torch.Tensor, mode: ImageReadMode = ..., apply_exif_orientation: bool = ...) -> torch.Tensor:
    """
    Decodes a PNG image into a 3 dimensional RGB or grayscale Tensor.

    The values of the output tensor are in uint8 in [0, 255] for most cases. If
    the image is a 16-bit png, then the output tensor is uint16 in [0, 65535]
    (supported from torchvision ``0.21``). Since uint16 support is limited in
    pytorch, we recommend calling
    :func:`torchvision.transforms.v2.functional.to_dtype()` with ``scale=True``
    after this function to convert the decoded image into a uint8 or float
    tensor.

    Args:
        input (Tensor[1]): a one dimensional uint8 tensor containing
            the raw bytes of the PNG image.
        mode (str or ImageReadMode): The mode to convert the image to, e.g. "RGB".
            Default is "UNCHANGED".  See :class:`~torchvision.io.ImageReadMode`
            for available modes.
        apply_exif_orientation (bool): apply EXIF orientation transformation to the output tensor.
            Default: False.

    Returns:
        output (Tensor[image_channels, image_height, image_width])
    """
    ...

def encode_png(input: torch.Tensor, compression_level: int = ...) -> torch.Tensor:
    """
    Takes an input tensor in CHW layout and returns a buffer with the contents
    of its corresponding PNG file.

    Args:
        input (Tensor[channels, image_height, image_width]): int8 image tensor of
            ``c`` channels, where ``c`` must 3 or 1.
        compression_level (int): Compression factor for the resulting file, it must be a number
            between 0 and 9. Default: 6

    Returns:
        Tensor[1]: A one dimensional int8 tensor that contains the raw bytes of the
            PNG file.
    """
    ...

def write_png(input: torch.Tensor, filename: str, compression_level: int = ...): # -> None:
    """
    Takes an input tensor in CHW layout (or HW in the case of grayscale images)
    and saves it in a PNG file.

    Args:
        input (Tensor[channels, image_height, image_width]): int8 image tensor of
            ``c`` channels, where ``c`` must be 1 or 3.
        filename (str or ``pathlib.Path``): Path to save the image.
        compression_level (int): Compression factor for the resulting file, it must be a number
            between 0 and 9. Default: 6
    """
    ...

def decode_jpeg(input: Union[torch.Tensor, List[torch.Tensor]], mode: ImageReadMode = ..., device: Union[str, torch.device] = ..., apply_exif_orientation: bool = ...) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Decode JPEG image(s) into 3D RGB or grayscale Tensor(s), on CPU or CUDA.

    The values of the output tensor are uint8 between 0 and 255.

    .. note::
        When using a CUDA device, passing a list of tensors is more efficient than repeated individual calls to ``decode_jpeg``.
        When using CPU the performance is equivalent.
        The CUDA version of this function has explicitly been designed with thread-safety in mind.
        This function does not return partial results in case of an error.

    Args:
        input (Tensor[1] or list[Tensor[1]]): a (list of) one dimensional uint8 tensor(s) containing
            the raw bytes of the JPEG image. The tensor(s) must be on CPU,
            regardless of the ``device`` parameter.
        mode (str or ImageReadMode): The mode to convert the image to, e.g. "RGB".
            Default is "UNCHANGED".  See :class:`~torchvision.io.ImageReadMode`
            for available modes.
        device (str or torch.device): The device on which the decoded image will
            be stored. If a cuda device is specified, the image will be decoded
            with `nvjpeg <https://developer.nvidia.com/nvjpeg>`_. This is only
            supported for CUDA version >= 10.1

            .. betastatus:: device parameter

            .. warning::
                There is a memory leak in the nvjpeg library for CUDA versions < 11.6.
                Make sure to rely on CUDA 11.6 or above before using ``device="cuda"``.
        apply_exif_orientation (bool): apply EXIF orientation transformation to the output tensor.
            Default: False. Only implemented for JPEG format on CPU.

    Returns:
        output (Tensor[image_channels, image_height, image_width] or list[Tensor[image_channels, image_height, image_width]]):
            The values of the output tensor(s) are uint8 between 0 and 255.
            ``output.device`` will be set to the specified ``device``


    """
    ...

def encode_jpeg(input: Union[torch.Tensor, List[torch.Tensor]], quality: int = ...) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Encode RGB tensor(s) into raw encoded jpeg bytes, on CPU or CUDA.

    .. note::
        Passing a list of CUDA tensors is more efficient than repeated individual calls to ``encode_jpeg``.
        For CPU tensors the performance is equivalent.

    Args:
        input (Tensor[channels, image_height, image_width] or List[Tensor[channels, image_height, image_width]]):
            (list of) uint8 image tensor(s) of ``c`` channels, where ``c`` must be 1 or 3
        quality (int): Quality of the resulting JPEG file(s). Must be a number between
            1 and 100. Default: 75

    Returns:
        output (Tensor[1] or list[Tensor[1]]): A (list of) one dimensional uint8 tensor(s) that contain the raw bytes of the JPEG file.
    """
    ...

def write_jpeg(input: torch.Tensor, filename: str, quality: int = ...): # -> None:
    """
    Takes an input tensor in CHW layout and saves it in a JPEG file.

    Args:
        input (Tensor[channels, image_height, image_width]): int8 image tensor of ``c``
            channels, where ``c`` must be 1 or 3.
        filename (str or ``pathlib.Path``): Path to save the image.
        quality (int): Quality of the resulting JPEG file, it must be a number
            between 1 and 100. Default: 75
    """
    ...

def decode_image(input: Union[torch.Tensor, str], mode: ImageReadMode = ..., apply_exif_orientation: bool = ...) -> torch.Tensor:
    """Decode an image into a uint8 tensor, from a path or from raw encoded bytes.

    Currently supported image formats are jpeg, png, gif and webp.

    The values of the output tensor are in uint8 in [0, 255] for most cases.

    If the image is a 16-bit png, then the output tensor is uint16 in [0, 65535]
    (supported from torchvision ``0.21``). Since uint16 support is limited in
    pytorch, we recommend calling
    :func:`torchvision.transforms.v2.functional.to_dtype()` with ``scale=True``
    after this function to convert the decoded image into a uint8 or float
    tensor.

    Args:
        input (Tensor or str or ``pathlib.Path``): The image to decode. If a
            tensor is passed, it must be one dimensional uint8 tensor containing
            the raw bytes of the image. Otherwise, this must be a path to the image file.
        mode (str or ImageReadMode): The mode to convert the image to, e.g. "RGB".
            Default is "UNCHANGED".  See :class:`~torchvision.io.ImageReadMode`
            for available modes.
        apply_exif_orientation (bool): apply EXIF orientation transformation to the output tensor.
           Only applies to JPEG and PNG images. Default: False.

    Returns:
        output (Tensor[image_channels, image_height, image_width])
    """
    ...

def read_image(path: str, mode: ImageReadMode = ..., apply_exif_orientation: bool = ...) -> torch.Tensor:
    """[OBSOLETE] Use :func:`~torchvision.io.decode_image` instead."""
    ...

def decode_gif(input: torch.Tensor) -> torch.Tensor:
    """
    Decode a GIF image into a 3 or 4 dimensional RGB Tensor.

    The values of the output tensor are uint8 between 0 and 255.
    The output tensor has shape ``(C, H, W)`` if there is only one image in the
    GIF, and ``(N, C, H, W)`` if there are ``N`` images.

    Args:
        input (Tensor[1]): a one dimensional contiguous uint8 tensor containing
            the raw bytes of the GIF image.

    Returns:
        output (Tensor[image_channels, image_height, image_width] or Tensor[num_images, image_channels, image_height, image_width])
    """
    ...

def decode_webp(input: torch.Tensor, mode: ImageReadMode = ...) -> torch.Tensor:
    """
    Decode a WEBP image into a 3 dimensional RGB[A] Tensor.

    The values of the output tensor are uint8 between 0 and 255.

    Args:
        input (Tensor[1]): a one dimensional contiguous uint8 tensor containing
            the raw bytes of the WEBP image.
        mode (str or ImageReadMode): The mode to convert the image to, e.g. "RGB".
            Default is "UNCHANGED".  See :class:`~torchvision.io.ImageReadMode`
            for available modes.

    Returns:
        Decoded image (Tensor[image_channels, image_height, image_width])
    """
    ...

