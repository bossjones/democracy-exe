"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutputWithPoolingAndNoAttention, ImageClassifierOutputWithNoAttention, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_levit import LevitConfig

"""PyTorch LeViT model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...
@dataclass
class LevitForImageClassificationWithTeacherOutput(ModelOutput):
    """
    Output type of [`LevitForImageClassificationWithTeacher`].

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores as the average of the `cls_logits` and `distillation_logits`.
        cls_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the
            class token).
        distillation_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the
            distillation token).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
    """
    logits: torch.FloatTensor = ...
    cls_logits: torch.FloatTensor = ...
    distillation_logits: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...


class LevitConvEmbeddings(nn.Module):
    """
    LeViT Conv Embeddings with Batch Norm, used in the initial patch embedding layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=..., groups=..., bn_weight_init=...) -> None:
        ...
    
    def forward(self, embeddings): # -> Any:
        ...
    


class LevitPatchEmbeddings(nn.Module):
    """
    LeViT patch embeddings, for final embeddings to be passed to transformer blocks. It consists of multiple
    `LevitConvEmbeddings`.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, pixel_values): # -> Any:
        ...
    


class MLPLayerWithBN(nn.Module):
    def __init__(self, input_dim, output_dim, bn_weight_init=...) -> None:
        ...
    
    def forward(self, hidden_state): # -> Any:
        ...
    


class LevitSubsample(nn.Module):
    def __init__(self, stride, resolution) -> None:
        ...
    
    def forward(self, hidden_state):
        ...
    


class LevitAttention(nn.Module):
    def __init__(self, hidden_sizes, key_dim, num_attention_heads, attention_ratio, resolution) -> None:
        ...
    
    @torch.no_grad()
    def train(self, mode=...): # -> None:
        ...
    
    def get_attention_biases(self, device): # -> Tensor:
        ...
    
    def forward(self, hidden_state): # -> Any:
        ...
    


class LevitAttentionSubsample(nn.Module):
    def __init__(self, input_dim, output_dim, key_dim, num_attention_heads, attention_ratio, stride, resolution_in, resolution_out) -> None:
        ...
    
    @torch.no_grad()
    def train(self, mode=...): # -> None:
        ...
    
    def get_attention_biases(self, device): # -> Tensor:
        ...
    
    def forward(self, hidden_state): # -> Any:
        ...
    


class LevitMLPLayer(nn.Module):
    """
    MLP Layer with `2X` expansion in contrast to ViT with `4X`.
    """
    def __init__(self, input_dim, hidden_dim) -> None:
        ...
    
    def forward(self, hidden_state): # -> Any:
        ...
    


class LevitResidualLayer(nn.Module):
    """
    Residual Block for LeViT
    """
    def __init__(self, module, drop_rate) -> None:
        ...
    
    def forward(self, hidden_state):
        ...
    


class LevitStage(nn.Module):
    """
    LeViT Stage consisting of `LevitMLPLayer` and `LevitAttention` layers.
    """
    def __init__(self, config, idx, hidden_sizes, key_dim, depths, num_attention_heads, attention_ratio, mlp_ratio, down_ops, resolution_in) -> None:
        ...
    
    def get_resolution(self): # -> Any:
        ...
    
    def forward(self, hidden_state): # -> Any:
        ...
    


class LevitEncoder(nn.Module):
    """
    LeViT Encoder consisting of multiple `LevitStage` stages.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_state, output_hidden_states=..., return_dict=...): # -> tuple[Any, ...] | BaseModelOutputWithNoAttention:
        ...
    


class LevitClassificationLayer(nn.Module):
    """
    LeViT Classification Layer
    """
    def __init__(self, input_dim, output_dim) -> None:
        ...
    
    def forward(self, hidden_state): # -> Any:
        ...
    


class LevitPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LevitConfig
    base_model_prefix = ...
    main_input_name = ...
    _no_split_modules = ...


LEVIT_START_DOCSTRING = ...
LEVIT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Levit model outputting raw features without any specific head on top.", LEVIT_START_DOCSTRING)
class LevitModel(LevitPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(LEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPoolingAndNoAttention, config_class=_CONFIG_FOR_DOC, modality="vision", expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: torch.FloatTensor = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        ...
    


@add_start_docstrings("""
    Levit Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """, LEVIT_START_DOCSTRING)
class LevitForImageClassification(LevitPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(LEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=ImageClassifierOutputWithNoAttention, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def forward(self, pixel_values: torch.FloatTensor = ..., labels: Optional[torch.LongTensor] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


@add_start_docstrings("""
    LeViT Model transformer with image classification heads on top (a linear layer on top of the final hidden state and
    a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet. .. warning::
           This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet
           supported.
    """, LEVIT_START_DOCSTRING)
class LevitForImageClassificationWithTeacher(LevitPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(LEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=LevitForImageClassificationWithTeacherOutput, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def forward(self, pixel_values: torch.FloatTensor = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, LevitForImageClassificationWithTeacherOutput]:
        ...
    


