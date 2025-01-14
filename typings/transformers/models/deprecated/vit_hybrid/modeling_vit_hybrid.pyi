"""
This type stub file was generated by pyright.
"""

import torch
from typing import Optional, Set, Tuple, Union
from torch import nn
from ....modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ....modeling_utils import PreTrainedModel
from ....utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_vit_hybrid import ViTHybridConfig

"""PyTorch ViT Hybrid model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...
class ViTHybridEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """
    def __init__(self, config: ViTHybridConfig, use_mask_token: bool = ...) -> None:
        ...
    
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """
        ...
    
    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor] = ..., interpolate_pos_encoding: bool = ...) -> torch.Tensor:
        ...
    


class ViTHybridPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """
    def __init__(self, config, feature_size=...) -> None:
        ...
    
    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = ...) -> torch.Tensor:
        ...
    


class ViTHybridSelfAttention(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None:
        ...
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        ...
    
    def forward(self, hidden_states, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        ...
    


class ViTHybridSdpaSelfAttention(ViTHybridSelfAttention):
    def __init__(self, config: ViTHybridConfig) -> None:
        ...
    
    def forward(self, hidden_states, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        ...
    


class ViTHybridSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTHybridLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """
    def __init__(self, config: ViTHybridConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...
    


class ViTHybridAttention(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None:
        ...
    
    def prune_heads(self, heads: Set[int]) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        ...
    


class ViTHybridSdpaAttention(ViTHybridAttention):
    def __init__(self, config: ViTHybridConfig) -> None:
        ...
    


class ViTHybridIntermediate(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class ViTHybridOutput(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...
    


VIT_HYBRID_ATTENTION_CLASSES = ...
class ViTHybridLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""
    def __init__(self, config: ViTHybridConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        ...
    


class ViTHybridEncoder(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...) -> Union[tuple, BaseModelOutput]:
        ...
    


class ViTHybridPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ViTHybridConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_sdpa = ...


VIT_START_DOCSTRING = ...
VIT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare ViT Hybrid Model transformer outputting raw hidden-states without any specific head on top.", VIT_START_DOCSTRING)
class ViTHybridModel(ViTHybridPreTrainedModel):
    def __init__(self, config: ViTHybridConfig, add_pooling_layer: bool = ..., use_mask_token: bool = ...) -> None:
        ...
    
    def get_input_embeddings(self) -> ViTHybridPatchEmbeddings:
        ...
    
    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC, modality="vision", expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: Optional[torch.Tensor] = ..., bool_masked_pos: Optional[torch.BoolTensor] = ..., head_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., interpolate_pos_encoding: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        ...
    


class ViTHybridPooler(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


@add_start_docstrings("""
    ViT Hybrid Model transformer with an image classification head on top (a linear layer on top of the final hidden
    state of the [CLS] token) e.g. for ImageNet.
    """, VIT_START_DOCSTRING)
class ViTHybridForImageClassification(ViTHybridPreTrainedModel):
    def __init__(self, config: ViTHybridConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=ImageClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def forward(self, pixel_values: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., labels: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., interpolate_pos_encoding: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


