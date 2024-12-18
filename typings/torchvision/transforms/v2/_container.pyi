"""
This type stub file was generated by pyright.
"""

from typing import Any, Callable, List, Optional, Sequence, Union
from torch import nn
from torchvision import transforms as _transforms
from torchvision.transforms.v2 import Transform

class Compose(Transform):
    """Composes several transforms together.

    This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """
    def __init__(self, transforms: Sequence[Callable]) -> None:
        ...
    
    def forward(self, *inputs: Any) -> Any:
        ...
    
    def extra_repr(self) -> str:
        ...
    


class RandomApply(Transform):
    """Apply randomly a list of transformations with a given probability.

    .. note::
        In order to script the transformation, please use ``torch.nn.ModuleList`` as input instead of list/tuple of
        transforms as shown below:

        >>> transforms = transforms.RandomApply(torch.nn.ModuleList([
        >>>     transforms.ColorJitter(),
        >>> ]), p=0.3)
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    Args:
        transforms (sequence or torch.nn.Module): list of transformations
        p (float): probability of applying the list of transforms
    """
    _v1_transform_cls = _transforms.RandomApply
    def __init__(self, transforms: Union[Sequence[Callable], nn.ModuleList], p: float = ...) -> None:
        ...
    
    def forward(self, *inputs: Any) -> Any:
        ...
    
    def extra_repr(self) -> str:
        ...
    


class RandomChoice(Transform):
    """Apply single transformation randomly picked from a list.

    This transform does not support torchscript.

    Args:
        transforms (sequence or torch.nn.Module): list of transformations
        p (list of floats or None, optional): probability of each transform being picked.
            If ``p`` doesn't sum to 1, it is automatically normalized. If ``None``
            (default), all transforms have the same probability.
    """
    def __init__(self, transforms: Sequence[Callable], p: Optional[List[float]] = ...) -> None:
        ...
    
    def forward(self, *inputs: Any) -> Any:
        ...
    


class RandomOrder(Transform):
    """Apply a list of transformations in a random order.

    This transform does not support torchscript.

    Args:
        transforms (sequence or torch.nn.Module): list of transformations
    """
    def __init__(self, transforms: Sequence[Callable]) -> None:
        ...
    
    def forward(self, *inputs: Any) -> Any:
        ...
    


