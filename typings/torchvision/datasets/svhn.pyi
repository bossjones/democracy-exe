"""
This type stub file was generated by pyright.
"""

from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union
from .vision import VisionDataset

class SVHN(VisionDataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load data from `.mat` format.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset where the data is stored.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    split_list = ...
    def __init__(self, root: Union[str, Path], split: str = ..., transform: Optional[Callable] = ..., target_transform: Optional[Callable] = ..., download: bool = ...) -> None:
        ...
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        ...
    
    def __len__(self) -> int:
        ...
    
    def download(self) -> None:
        ...
    
    def extra_repr(self) -> str:
        ...
    


