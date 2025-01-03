"""
This type stub file was generated by pyright.
"""

import torch
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from .base import HfQuantizer
from ..modeling_utils import PreTrainedModel
from ..utils import is_torch_available

if TYPE_CHECKING:
    ...
if is_torch_available():
    ...
logger = ...
class FbgemmFp8HfQuantizer(HfQuantizer):
    """
    FP8 quantization using fbgemm kernels
    """
    requires_parameters_quantization = ...
    requires_calibration = ...
    required_packages = ...
    def __init__(self, quantization_config, **kwargs) -> None:
        ...
    
    def validate_environment(self, *args, **kwargs): # -> None:
        ...
    
    def update_torch_dtype(self, torch_dtype: torch.dtype) -> torch.dtype:
        ...
    
    def check_quantized_param(self, model: PreTrainedModel, param_value: torch.Tensor, param_name: str, state_dict: Dict[str, Any], **kwargs): # -> bool:
        ...
    
    def create_quantized_param(self, model: PreTrainedModel, param_value: torch.Tensor, param_name: str, target_device: torch.device, state_dict: Dict[str, Any], unexpected_keys: Optional[List[str]] = ...): # -> None:
        """
        Quantizes weights into weight and weight_scale
        """
        ...
    
    def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        ...
    
    def is_serializable(self, safe_serialization=...): # -> Literal[True]:
        ...
    
    @property
    def is_trainable(self) -> bool:
        ...
    


