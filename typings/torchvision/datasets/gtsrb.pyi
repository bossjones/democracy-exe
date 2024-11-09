"""
This type stub file was generated by pyright.
"""

import pathlib
from typing import Any, Callable, Optional, Tuple, Union

from .vision import VisionDataset

class GTSRB(VisionDataset):
    """`German Traffic Sign Recognition Benchmark (GTSRB) <https://benchmark.ini.rub.de/>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(self, root: Union[str, pathlib.Path], split: str = ..., transform: Optional[Callable] = ..., target_transform: Optional[Callable] = ..., download: bool = ...) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        ...

    def download(self) -> None:
        ...
