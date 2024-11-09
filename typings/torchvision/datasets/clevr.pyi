"""
This type stub file was generated by pyright.
"""

import pathlib
from typing import Any, Callable, Optional, Tuple, Union

from .vision import VisionDataset

class CLEVRClassification(VisionDataset):
    """`CLEVR <https://cs.stanford.edu/people/jcjohns/clevr/>`_  classification dataset.

    The number of objects in a scene are used as label.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory ``root/clevr`` exists or will be saved to if download is
            set to True.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in them target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory. If
            dataset is already downloaded, it is not downloaded again.
    """
    _URL = ...
    _MD5 = ...
    def __init__(self, root: Union[str, pathlib.Path], split: str = ..., transform: Optional[Callable] = ..., target_transform: Optional[Callable] = ..., download: bool = ...) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        ...

    def extra_repr(self) -> str:
        ...
