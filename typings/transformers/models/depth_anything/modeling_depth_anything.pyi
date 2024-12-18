"""
This type stub file was generated by pyright.
"""

import torch
from typing import List, Optional, Tuple, Union
from torch import nn
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from .configuration_depth_anything import DepthAnythingConfig

"""PyTorch Depth Anything model."""
logger = ...
_CONFIG_FOR_DOC = ...
DEPTH_ANYTHING_START_DOCSTRING = ...
DEPTH_ANYTHING_INPUTS_DOCSTRING = ...
class DepthAnythingReassembleLayer(nn.Module):
    def __init__(self, config, channels, factor) -> None:
        ...
    
    def forward(self, hidden_state): # -> Any:
        ...
    


class DepthAnythingReassembleStage(nn.Module):
    """
    This class reassembles the hidden states of the backbone into image-like feature representations at various
    resolutions.

    This happens in 3 stages:
    1. Take the patch embeddings and reshape them to image-like feature representations.
    2. Project the channel dimension of the hidden states according to `config.neck_hidden_sizes`.
    3. Resizing the spatial dimensions (height, width).

    Args:
        config (`[DepthAnythingConfig]`):
            Model configuration class defining the model architecture.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: List[torch.Tensor], patch_height=..., patch_width=...) -> List[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length + 1, hidden_size)`):
                List of hidden states from the backbone.
        """
        ...
    


class DepthAnythingPreActResidualLayer(nn.Module):
    """
    ResidualConvUnit, pre-activate residual unit.

    Args:
        config (`[DepthAnythingConfig]`):
            Model configuration class defining the model architecture.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        ...
    


class DepthAnythingFeatureFusionLayer(nn.Module):
    """Feature fusion layer, merges feature maps from different stages.

    Args:
        config (`[DepthAnythingConfig]`):
            Model configuration class defining the model architecture.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_state, residual=..., size=...): # -> Any:
        ...
    


class DepthAnythingFeatureFusionStage(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, size=...): # -> list[Any]:
        ...
    


class DepthAnythingPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DepthAnythingConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...


class DepthAnythingNeck(nn.Module):
    """
    DepthAnythingNeck. A neck is a module that is normally used between the backbone and the head. It takes a list of tensors as
    input and produces another list of tensors as output. For DepthAnything, it includes 2 stages:

    * DepthAnythingReassembleStage
    * DepthAnythingFeatureFusionStage.

    Args:
        config (dict): config dict.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: List[torch.Tensor], patch_height=..., patch_width=...) -> List[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, hidden_size, height, width)`):
                List of hidden states from the backbone.
        """
        ...
    


class DepthAnythingDepthEstimationHead(nn.Module):
    """
    Output head consisting of 3 convolutional layers. It progressively halves the feature dimension and upsamples
    the predictions to the input resolution after the first convolutional layer (details can be found in the DPT paper's
    supplementary material). The final activation function is either ReLU or Sigmoid, depending on the depth estimation
    type (relative or metric). For metric depth estimation, the output is scaled by the maximum depth used during pretraining.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: List[torch.Tensor], patch_height, patch_width) -> torch.Tensor:
        ...
    


@add_start_docstrings("""
    Depth Anything Model with a depth estimation head on top (consisting of 3 convolutional layers) e.g. for KITTI, NYUv2.
    """, DEPTH_ANYTHING_START_DOCSTRING)
class DepthAnythingForDepthEstimation(DepthAnythingPreTrainedModel):
    _no_split_modules = ...
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DEPTH_ANYTHING_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, labels: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], DepthEstimatorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
        >>> model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # interpolate to original size
        >>> post_processed_output = image_processor.post_process_depth_estimation(
        ...     outputs,
        ...     target_sizes=[(image.height, image.width)],
        ... )

        >>> # visualize the prediction
        >>> predicted_depth = post_processed_output[0]["predicted_depth"]
        >>> depth = predicted_depth * 255 / predicted_depth.max()
        >>> depth = depth.detach().cpu().numpy()
        >>> depth = Image.fromarray(depth.astype("uint8"))
        ```"""
        ...
    


