"""
This type stub file was generated by pyright.
"""

import pathlib
from typing import Any, Callable, Optional, Tuple, Union

from .vision import VisionDataset

class DTD(VisionDataset):
    """`Describable Textures Dataset (DTD) <https://www.robots.ox.ac.uk/~vgg/data/dtd/>`_.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        partition (int, optional): The dataset partition. Should be ``1 <= partition <= 10``. Defaults to ``1``.

            .. note::

                The partition only changes which split each image belongs to. Thus, regardless of the selected
                partition, combining all splits will result in all images.

        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """
    _URL = ...
    _MD5 = ...
    def __init__(self, root: Union[str, pathlib.Path], split: str = ..., partition: int = ..., transform: Optional[Callable] = ..., target_transform: Optional[Callable] = ..., download: bool = ...) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        ...

    def extra_repr(self) -> str:
        ...
