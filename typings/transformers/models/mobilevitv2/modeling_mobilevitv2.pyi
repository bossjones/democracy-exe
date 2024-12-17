"""
This type stub file was generated by pyright.
"""

import torch
from typing import Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutputWithNoAttention, BaseModelOutputWithPoolingAndNoAttention, ImageClassifierOutputWithNoAttention, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_mobilevitv2 import MobileViTV2Config

"""PyTorch MobileViTV2 model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...
def make_divisible(value: int, divisor: int = ..., min_value: Optional[int] = ...) -> int:
    """
    Ensure that all layers have a channel count that is divisible by `divisor`. This function is taken from the
    original TensorFlow repo. It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    ...

def clip(value: float, min_val: float = ..., max_val: float = ...) -> float:
    ...

class MobileViTV2ConvLayer(nn.Module):
    def __init__(self, config: MobileViTV2Config, in_channels: int, out_channels: int, kernel_size: int, stride: int = ..., groups: int = ..., bias: bool = ..., dilation: int = ..., use_normalization: bool = ..., use_activation: Union[bool, str] = ...) -> None:
        ...
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        ...
    


class MobileViTV2InvertedResidual(nn.Module):
    """
    Inverted residual block (MobileNetv2): https://arxiv.org/abs/1801.04381
    """
    def __init__(self, config: MobileViTV2Config, in_channels: int, out_channels: int, stride: int, dilation: int = ...) -> None:
        ...
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        ...
    


class MobileViTV2MobileNetLayer(nn.Module):
    def __init__(self, config: MobileViTV2Config, in_channels: int, out_channels: int, stride: int = ..., num_stages: int = ...) -> None:
        ...
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        ...
    


class MobileViTV2LinearSelfAttention(nn.Module):
    """
    This layer applies a self-attention with linear complexity, as described in MobileViTV2 paper:
    https://arxiv.org/abs/2206.02680

    Args:
        config (`MobileVitv2Config`):
             Model configuration object
        embed_dim (`int`):
            `input_channels` from an expected input of size :math:`(batch_size, input_channels, height, width)`
    """
    def __init__(self, config: MobileViTV2Config, embed_dim: int) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class MobileViTV2FFN(nn.Module):
    def __init__(self, config: MobileViTV2Config, embed_dim: int, ffn_latent_dim: int, ffn_dropout: float = ...) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class MobileViTV2TransformerLayer(nn.Module):
    def __init__(self, config: MobileViTV2Config, embed_dim: int, ffn_latent_dim: int, dropout: float = ...) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class MobileViTV2Transformer(nn.Module):
    def __init__(self, config: MobileViTV2Config, n_layers: int, d_model: int) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class MobileViTV2Layer(nn.Module):
    """
    MobileViTV2 layer: https://arxiv.org/abs/2206.02680
    """
    def __init__(self, config: MobileViTV2Config, in_channels: int, out_channels: int, attn_unit_dim: int, n_attn_blocks: int = ..., dilation: int = ..., stride: int = ...) -> None:
        ...
    
    def unfolding(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        ...
    
    def folding(self, patches: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        ...
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        ...
    


class MobileViTV2Encoder(nn.Module):
    def __init__(self, config: MobileViTV2Config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, output_hidden_states: bool = ..., return_dict: bool = ...) -> Union[tuple, BaseModelOutputWithNoAttention]:
        ...
    


class MobileViTV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MobileViTV2Config
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...


MOBILEVITV2_START_DOCSTRING = ...
MOBILEVITV2_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare MobileViTV2 model outputting raw hidden-states without any specific head on top.", MOBILEVITV2_START_DOCSTRING)
class MobileViTV2Model(MobileViTV2PreTrainedModel):
    def __init__(self, config: MobileViTV2Config, expand_output: bool = ...) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(MOBILEVITV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPoolingAndNoAttention, config_class=_CONFIG_FOR_DOC, modality="vision", expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: Optional[torch.Tensor] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[tuple, BaseModelOutputWithPoolingAndNoAttention]:
        ...
    


@add_start_docstrings("""
    MobileViTV2 model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """, MOBILEVITV2_START_DOCSTRING)
class MobileViTV2ForImageClassification(MobileViTV2PreTrainedModel):
    def __init__(self, config: MobileViTV2Config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(MOBILEVITV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=ImageClassifierOutputWithNoAttention, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def forward(self, pixel_values: Optional[torch.Tensor] = ..., output_hidden_states: Optional[bool] = ..., labels: Optional[torch.Tensor] = ..., return_dict: Optional[bool] = ...) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


class MobileViTV2ASPPPooling(nn.Module):
    def __init__(self, config: MobileViTV2Config, in_channels: int, out_channels: int) -> None:
        ...
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        ...
    


class MobileViTV2ASPP(nn.Module):
    """
    ASPP module defined in DeepLab papers: https://arxiv.org/abs/1606.00915, https://arxiv.org/abs/1706.05587
    """
    def __init__(self, config: MobileViTV2Config) -> None:
        ...
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        ...
    


class MobileViTV2DeepLabV3(nn.Module):
    """
    DeepLabv3 architecture: https://arxiv.org/abs/1706.05587
    """
    def __init__(self, config: MobileViTV2Config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


@add_start_docstrings("""
    MobileViTV2 model with a semantic segmentation head on top, e.g. for Pascal VOC.
    """, MOBILEVITV2_START_DOCSTRING)
class MobileViTV2ForSemanticSegmentation(MobileViTV2PreTrainedModel):
    def __init__(self, config: MobileViTV2Config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(MOBILEVITV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.Tensor] = ..., labels: Optional[torch.Tensor] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[tuple, SemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> import requests
        >>> import torch
        >>> from PIL import Image
        >>> from transformers import AutoImageProcessor, MobileViTV2ForSemanticSegmentation

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
        >>> model = MobileViTV2ForSemanticSegmentation.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # logits are of shape (batch_size, num_labels, height, width)
        >>> logits = outputs.logits
        ```"""
        ...
    


