"""
This type stub file was generated by pyright.
"""

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
from .vision import VisionDataset

CSV = ...
class CelebA(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (Tensor shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (Tensor shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (Tensor shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

            .. warning::

                To download the dataset `gdown <https://github.com/wkentaro/gdown>`_ is required.
    """
    base_folder = ...
    file_list = ...
    def __init__(self, root: Union[str, Path], split: str = ..., target_type: Union[List[str], str] = ..., transform: Optional[Callable] = ..., target_transform: Optional[Callable] = ..., download: bool = ...) -> None:
        ...
    
    def download(self) -> None:
        ...
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        ...
    
    def __len__(self) -> int:
        ...
    
    def extra_repr(self) -> str:
        ...
    


