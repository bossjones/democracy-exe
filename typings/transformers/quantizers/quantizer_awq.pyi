"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING

from ..utils import is_torch_available
from .base import HfQuantizer

if TYPE_CHECKING:
    ...
if is_torch_available():
    ...
logger = ...
class AwqQuantizer(HfQuantizer):
    """
    4-bit quantization for Activation-aware Weight Quantization(AWQ) (https://arxiv.org/abs/2306.00978)
    """
    requires_calibration = ...
    required_packages = ...
    def __init__(self, quantization_config, **kwargs) -> None:
        ...

    def validate_environment(self, device_map, **kwargs): # -> None:
        ...

    def update_torch_dtype(self, torch_dtype): # -> dtype:
        ...

    @property
    def is_serializable(self): # -> bool:
        ...

    @property
    def is_trainable(self): # -> bool:
        ...
