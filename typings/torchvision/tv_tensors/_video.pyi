"""
This type stub file was generated by pyright.
"""

import torch
from typing import Any, Optional, Union
from ._tv_tensor import TVTensor

class Video(TVTensor):
    """:class:`torch.Tensor` subclass for videos with shape ``[..., T, C, H, W]``.

    Args:
        data (tensor-like): Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        dtype (torch.dtype, optional): Desired data type. If omitted, will be inferred from
            ``data``.
        device (torch.device, optional): Desired device. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the video is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
    """
    def __new__(cls, data: Any, *, dtype: Optional[torch.dtype] = ..., device: Optional[Union[torch.device, str, int]] = ..., requires_grad: Optional[bool] = ...) -> Video:
        ...
    
    def __repr__(self, *, tensor_contents: Any = ...) -> str:
        ...
    


