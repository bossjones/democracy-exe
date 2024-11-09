"""
This type stub file was generated by pyright.
"""

from collections.abc import Mapping
from typing import List

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

""" EfficientNet model configuration"""
logger = ...
class EfficientNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`EfficientNetModel`]. It is used to instantiate an
    EfficientNet model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the EfficientNet
    [google/efficientnet-b7](https://huggingface.co/google/efficientnet-b7) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 600):
            The input image size.
        width_coefficient (`float`, *optional*, defaults to 2.0):
            Scaling coefficient for network width at each stage.
        depth_coefficient (`float`, *optional*, defaults to 3.1):
            Scaling coefficient for network depth at each stage.
        depth_divisor `int`, *optional*, defaults to 8):
            A unit of network width.
        kernel_sizes (`List[int]`, *optional*, defaults to `[3, 3, 5, 3, 5, 5, 3]`):
            List of kernel sizes to be used in each block.
        in_channels (`List[int]`, *optional*, defaults to `[32, 16, 24, 40, 80, 112, 192]`):
            List of input channel sizes to be used in each block for convolutional layers.
        out_channels (`List[int]`, *optional*, defaults to `[16, 24, 40, 80, 112, 192, 320]`):
            List of output channel sizes to be used in each block for convolutional layers.
        depthwise_padding (`List[int]`, *optional*, defaults to `[]`):
            List of block indices with square padding.
        strides (`List[int]`, *optional*, defaults to `[1, 2, 2, 2, 1, 2, 1]`):
            List of stride sizes to be used in each block for convolutional layers.
        num_block_repeats (`List[int]`, *optional*, defaults to `[1, 2, 2, 3, 3, 4, 1]`):
            List of the number of times each block is to repeated.
        expand_ratios (`List[int]`, *optional*, defaults to `[1, 6, 6, 6, 6, 6, 6]`):
            List of scaling coefficient of each block.
        squeeze_expansion_ratio (`float`, *optional*, defaults to 0.25):
            Squeeze expansion ratio.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in each block. If string, `"gelu"`, `"relu"`,
            `"selu", `"gelu_new"`, `"silu"` and `"mish"` are supported.
        hiddem_dim (`int`, *optional*, defaults to 1280):
            The hidden dimension of the layer before the classification head.
        pooling_type (`str` or `function`, *optional*, defaults to `"mean"`):
            Type of final pooling to be applied before the dense classification head. Available options are [`"mean"`,
            `"max"`]
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        batch_norm_eps (`float`, *optional*, defaults to 1e-3):
            The epsilon used by the batch normalization layers.
        batch_norm_momentum (`float`, *optional*, defaults to 0.99):
            The momentum used by the batch normalization layers.
        dropout_rate (`float`, *optional*, defaults to 0.5):
            The dropout rate to be applied before final classifier layer.
        drop_connect_rate (`float`, *optional*, defaults to 0.2):
            The drop rate for skip connections.

    Example:
    ```python
    >>> from transformers import EfficientNetConfig, EfficientNetModel

    >>> # Initializing a EfficientNet efficientnet-b7 style configuration
    >>> configuration = EfficientNetConfig()

    >>> # Initializing a model (with random weights) from the efficientnet-b7 style configuration
    >>> model = EfficientNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    def __init__(self, num_channels: int = ..., image_size: int = ..., width_coefficient: float = ..., depth_coefficient: float = ..., depth_divisor: int = ..., kernel_sizes: List[int] = ..., in_channels: List[int] = ..., out_channels: List[int] = ..., depthwise_padding: List[int] = ..., strides: List[int] = ..., num_block_repeats: List[int] = ..., expand_ratios: List[int] = ..., squeeze_expansion_ratio: float = ..., hidden_act: str = ..., hidden_dim: int = ..., pooling_type: str = ..., initializer_range: float = ..., batch_norm_eps: float = ..., batch_norm_momentum: float = ..., dropout_rate: float = ..., drop_connect_rate: float = ..., **kwargs) -> None:
        ...



class EfficientNetOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        ...

    @property
    def atol_for_validation(self) -> float:
        ...
