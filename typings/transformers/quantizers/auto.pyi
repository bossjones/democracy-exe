"""
This type stub file was generated by pyright.
"""

from typing import Dict, Optional, Union
from ..utils.quantization_config import QuantizationConfigMixin

AUTO_QUANTIZER_MAPPING = ...
AUTO_QUANTIZATION_CONFIG_MAPPING = ...
class AutoQuantizationConfig:
    """
    The Auto-HF quantization config class that takes care of automatically dispatching to the correct
    quantization config given a quantization config stored in a dictionary.
    """
    @classmethod
    def from_dict(cls, quantization_config_dict: Dict):
        ...
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        ...
    


class AutoHfQuantizer:
    """
     The Auto-HF quantizer class that takes care of automatically instantiating to the correct
    `HfQuantizer` given the `QuantizationConfig`.
    """
    @classmethod
    def from_config(cls, quantization_config: Union[QuantizationConfigMixin, Dict], **kwargs):
        ...
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        ...
    
    @classmethod
    def merge_quantization_configs(cls, quantization_config: Union[dict, QuantizationConfigMixin], quantization_config_from_args: Optional[QuantizationConfigMixin]): # -> GPTQConfig | AwqConfig | FbgemmFp8Config | dict[Any, Any] | QuantizationConfigMixin:
        """
        handles situations where both quantization_config from args and quantization_config from model config are present.
        """
        ...
    


