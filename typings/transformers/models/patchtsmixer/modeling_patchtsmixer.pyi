"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_patchtsmixer import PatchTSMixerConfig

""" PyTorch PatchTSMixer model."""
logger = ...
_CONFIG_FOR_DOC = ...
PATCHTSMIXER_START_DOCSTRING = ...
PATCHTSMIXER_INPUTS_DOCSTRING = ...
class PatchTSMixerGatedAttention(nn.Module):
    """
    Module that applies gated attention to input data.

    Args:
        in_size (`int`): The input size.
        out_size (`int`): The output size.
    """
    def __init__(self, in_size: int, out_size: int) -> None:
        ...

    def forward(self, inputs):
        ...



class PatchTSMixerBatchNorm(nn.Module):
    """
    Compute batch normalization over the sequence length (time) dimension.
    """
    def __init__(self, config: PatchTSMixerConfig) -> None:
        ...

    def forward(self, inputs: torch.Tensor): # -> Any:
        """
        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length, d_model)`):
                input for Batch norm calculation
        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, d_model)`
        """
        ...



class PatchTSMixerPositionalEncoding(nn.Module):
    """
    Class for positional encoding
    """
    def __init__(self, config: PatchTSMixerConfig) -> None:
        ...

    def forward(self, patch_input: torch.Tensor): # -> Tensor:
        ...



class PatchTSMixerNormLayer(nn.Module):
    """Normalization block

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """
    def __init__(self, config: PatchTSMixerConfig) -> None:
        ...

    def forward(self, inputs: torch.Tensor): # -> Tensor:
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the normalization layer.
        Returns:
            `torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`
        """
        ...



class PatchTSMixerMLP(nn.Module):
    def __init__(self, in_features, out_features, config) -> None:
        ...

    def forward(self, inputs: torch.Tensor): # -> Tensor:
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the MLP layer.
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        ...



class PatchTSMixerChannelFeatureMixerBlock(nn.Module):
    """This module mixes the features in the channel dimension.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """
    def __init__(self, config: PatchTSMixerConfig) -> None:
        ...

    def forward(self, inputs: torch.Tensor): # -> Tensor:
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                input to the MLP layer
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        ...



class PatchTSMixerAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = ..., is_decoder: bool = ..., bias: bool = ..., is_causal: bool = ..., config: Optional[PatchTSMixerConfig] = ...) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor] = ..., past_key_value: Optional[Tuple[torch.Tensor]] = ..., attention_mask: Optional[torch.Tensor] = ..., layer_head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        ...



class PatchMixerBlock(nn.Module):
    """This module mixes the patch dimension.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """
    def __init__(self, config: PatchTSMixerConfig) -> None:
        ...

    def forward(self, hidden_state):
        """
        Args:
            hidden_state (`torch.Tensor`): Input tensor.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        ...



class FeatureMixerBlock(nn.Module):
    """This module mixes the hidden feature dimension.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    """
    def __init__(self, config: PatchTSMixerConfig) -> None:
        ...

    def forward(self, hidden: torch.Tensor): # -> Tensor:
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        ...



class PatchTSMixerLayer(nn.Module):
    """
    The `PatchTSMixer` layer that does all three kinds of mixing.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    """
    def __init__(self, config: PatchTSMixerConfig) -> None:
        ...

    def forward(self, hidden: torch.Tensor): # -> Tensor:
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        ...



class PatchTSMixerBlock(nn.Module):
    """The main computing framework of the `PatchTSMixer` model.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """
    def __init__(self, config: PatchTSMixerConfig) -> None:
        ...

    def forward(self, hidden_state, output_hidden_states: bool = ...): # -> tuple[Any, list[Any]] | tuple[Any, None]:
        """
        Args:
            hidden_state (`torch.Tensor`): The input tensor.
            output_hidden_states (`bool`, *optional*, defaults to False.):
                Whether to output the hidden states as well.

        Returns:
            `torch.Tensor`: The embedding. `list`: List of all hidden states if `output_hidden_states` is set to
            `True`.
        """
        ...



class PatchTSMixerForPredictionHead(nn.Module):
    """Prediction Head for Forecasting

    Args:
        config (`PatchTSMixerConfig`, *required*): Configuration.
    """
    def __init__(self, config: PatchTSMixerConfig, distribution_output=...) -> None:
        ...

    def forward(self, hidden_features): # -> tuple[Any, ...] | Any:
        """

        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size, num_patch, d_model)` in `flatten` mode
                or `(batch_size, n_vars, num_patch, d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

        Returns:
            `torch.Tensor` of shape `(batch_size, prediction_length, nvars)`.

        """
        ...



class PatchTSMixerLinearHead(nn.Module):
    """Linear head for Classification and Regression.

    Args:
        config (`PatchTSMixerConfig`, *required*):

    """
    def __init__(self, config: PatchTSMixerConfig, distribution_output=...) -> None:
        ...

    def forward(self, hidden_features): # -> Any:
        """
        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size x num_patch x d_model)` in `flatten` mode
                or `(batch_size x n_vars x num_patch x d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

        Returns:
            `torch.Tensor` of shape `(batch_size x num_targets)`.
        """
        ...



class PatchTSMixerPreTrainedModel(PreTrainedModel):
    config_class = PatchTSMixerConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...


class PatchTSMixerPretrainHead(nn.Module):
    """Pretraining head.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """
    def __init__(self, config: PatchTSMixerConfig) -> None:
        ...

    def forward(self, hidden_features): # -> Any:
        """
        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size x num_patch x d_model)` in `flatten` mode
                or `(batch_size x n_vars x num_patch x d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

        Returns:
            `torch.Tensor` of shape `(batch_size x n_vars x num_patch x patch_length)`.
        """
        ...



def random_masking(inputs: torch.Tensor, mask_ratio: float, unmasked_channel_indices: list = ..., channel_consistent_masking: bool = ..., mask_value: int = ...): # -> tuple[Tensor, Tensor]:
    """random_masking: Mask the input considering the control variables.

    Args:
        inputs (`torch.Tensor` of shape `(batch_size, num_channels, sequence_length, num_features)`):
            The input tensor to mask.
        mask_ratio (`float`):
            Masking ratio applied to mask the input data during random pretraining. It is the number between 0 and 1.
        unmasked_channel_indices (list, *optional*):
            Indices of channels that will not be masked.
        channel_consistent_masking (bool, *optional*, defaults to `False`):
            When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary
            across channels.
        mask_value (int, *optional*, defaults to 0):
            Define the value of masked patches for pretraining.

    Returns:
        `tuple(torch.Tensor)`: inputs_mask, masked input, same shape as input Tensor and mask tensor of shape [bs x c x
        n]
    """
    ...

def forecast_masking(inputs: torch.Tensor, num_forecast_mask_patches: Union[list, int], unmasked_channel_indices: list = ..., mask_value: int = ...): # -> tuple[Tensor, Tensor]:
    """Forecast masking that masks the last K patches where K is from the num_forecast_mask_patches.
    If num_forecast_mask_patches is a list, samples in the batch will be randomly masked by numbers defined in the list.

    Parameters:
        inputs (`torch.Tensor`):
            Input of shape `(bs, num_channels, num_patch, patch_length)`
        num_forecast_mask_patches (`list`):
            Number of patches to be masked at the end of each batch sample. e.g. 4 or [3, 5].
        unmasked_channel_indices (`list`, *optional*):
            Indices of channels that are not masked.
        mask_value (`int`, *optional*, defaults to 0):
            Values in the masked patches will be filled by `mask_value`.

    Returns:
        `tuple(torch.Tensor)`: inputs_mask, masked input, same shape as inputs Tensor and Mask tensor of shape `(bs,
        num_channels , num_patch)` or `(bs, tsg1, tsg2, num_channels, num_patch)`
    """
    ...

class PatchTSMixerPatchify(nn.Module):
    """
    A class to patchify the time series sequence into different patches

    Returns:
        `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
    """
    def __init__(self, config: PatchTSMixerConfig) -> None:
        ...

    def forward(self, past_values: torch.Tensor): # -> Tensor:
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Input for patchification

        Returns:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
        """
        ...



class PatchTSMixerMasking(nn.Module):
    """
    Class to perform random or forecast masking.

    Parameters:
        config (`PatchTSMixerConfig`): model config
    Returns:
        x_mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`)
            Masked patched input
        mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches)`)
            Bool tensor indicating True on masked points
    """
    def __init__(self, config: PatchTSMixerConfig) -> None:
        ...

    def forward(self, patch_input: torch.Tensor): # -> tuple[Tensor, Tensor]:
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Patch input

        Return:
            masked_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`)
                Masked patched input
            mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches)`)
                Bool tensor indicating True on masked points

        """
        ...



class PatchTSMixerStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along the first dimension, and then normalizes it by
    subtracting from the mean and dividing by the standard deviation.
    """
    def __init__(self, config: PatchTSMixerConfig) -> None:
        ...

    def forward(self, data: torch.Tensor, observed_indicator: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        ...



class PatchTSMixerMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along the first dimension, and scales the data
    accordingly.
    """
    def __init__(self, config: PatchTSMixerConfig) -> None:
        ...

    def forward(self, data: torch.Tensor, observed_indicator: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        ...



class PatchTSMixerNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along the first dimension, and therefore applies no scaling to the input data.
    """
    def __init__(self, config: PatchTSMixerConfig) -> None:
        ...

    def forward(self, data: torch.Tensor, observed_indicator: torch.Tensor = ...) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        ...



@dataclass
class PatchTSMixerEncoderOutput(ModelOutput):
    """
    Base class for `PatchTSMixerEncoderOutput`, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, d_model)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
    """
    last_hidden_state: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...


class PatchTSMixerEncoder(PatchTSMixerPreTrainedModel):
    """
    Encoder for PatchTSMixer which inputs patched time-series and outputs patched embeddings.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """
    def __init__(self, config: PatchTSMixerConfig) -> None:
        ...

    @replace_return_docstrings(output_type=PatchTSMixerEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, past_values: torch.Tensor, output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, PatchTSMixerEncoderOutput]:
        r"""
        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
                Context values of the time series. For a pretraining task, this denotes the input time series to
                predict the masked portion. For a forecasting task, this denotes the history/past time series values.
                Similarly, for classification or regression tasks, it denotes the appropriate context values of the
                time series.

                For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series,
                it is greater than 1.

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.

            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, n_vars, num_patches, d_model)`
        """
        ...



@dataclass
class PatchTSMixerModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor`  of shape `(batch_size, num_channels, num_patches, d_model)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
        patch_input (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`):
            Patched input data to the model.
        mask: (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches)`,*optional*):
            Bool Tensor indicating True in masked patches and False otherwise.
        loc: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the mean of the context window per channel. Used for revin denorm outside the model, if revin
            enabled.
        scale: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the std dev of the context window per channel. Used for revin denorm outside the model, if revin
            enabled.
    """
    last_hidden_state: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    patch_input: torch.FloatTensor = ...
    mask: Optional[torch.FloatTensor] = ...
    loc: Optional[torch.FloatTensor] = ...
    scale: Optional[torch.FloatTensor] = ...


@add_start_docstrings("The PatchTSMixer Model for time-series forecasting.", PATCHTSMIXER_START_DOCSTRING)
class PatchTSMixerModel(PatchTSMixerPreTrainedModel):
    def __init__(self, config: PatchTSMixerConfig, mask_input: bool = ...) -> None:
        ...

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, past_values: torch.Tensor, observed_mask: Optional[torch.Tensor] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> PatchTSMixerModelOutput:
        r"""
        observed_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]`:
                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Returns:

        """
        ...



@dataclass
class PatchTSMixerForPreTrainingOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForPreTrainingOutput`].

    Args:
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, patch_length)`):
            Prediction output from the pretrain head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings before passing through the head.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss
    """
    loss: Optional[torch.FloatTensor] = ...
    prediction_outputs: torch.FloatTensor = ...
    last_hidden_state: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...


class PatchTSMixerForPretraining(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for mask pretraining.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """
    def __init__(self, config: PatchTSMixerConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, past_values: torch.Tensor, observed_mask: Optional[torch.Tensor] = ..., output_hidden_states: Optional[bool] = ..., return_loss: bool = ..., return_dict: Optional[bool] = ...) -> PatchTSMixerForPreTrainingOutput:
        r"""
        observed_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]`:
                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
        return_loss (`bool`,  *optional*):
            Whether to return the loss in the `forward` call.

        Returns:

        """
        ...



@dataclass
class PatchTSMixerForPredictionOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForPredictionOutput`].

    Args:
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_input_channels)`):
            Prediction output from the forecast head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
        loc (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
            Input mean
        scale (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
            Input std dev

    """
    loss: Optional[torch.FloatTensor] = ...
    prediction_outputs: torch.FloatTensor = ...
    last_hidden_state: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    loc: torch.FloatTensor = ...
    scale: torch.FloatTensor = ...


@dataclass
class SamplePatchTSMixerPredictionOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Args:
        sequences (`torch.FloatTensor` of shape `(batch_size, num_samples, prediction_length, number_channels)`):
            Sampled values from the chosen distribution.
    """
    sequences: torch.FloatTensor = ...


@dataclass
class SamplePatchTSMixerRegressionOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Args:
        sequences (`torch.FloatTensor` of shape `(batch_size, num_samples, num_targets)`
                Sampled values from the chosen distribution.
    """
    sequences: torch.FloatTensor = ...


def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    ...

def weighted_average(input_tensor: torch.Tensor, weights: Optional[torch.Tensor] = ..., dim=...) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given `dim`, masking values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Args:
        input_tensor (`torch.FloatTensor`):
            Input tensor, of which the average must be computed.
        weights (`torch.FloatTensor`, *optional*):
            Weights tensor, of the same shape as `input_tensor`.
        dim (`int`, *optional*):
            The dim along which to average `input_tensor`.

    Returns:
        `torch.FloatTensor`: The tensor with values averaged along the specified `dim`.
    """
    ...

class PatchTSMixerForPrediction(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for forecasting application.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """
    def __init__(self, config: PatchTSMixerConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerForPredictionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, past_values: torch.Tensor, observed_mask: Optional[torch.Tensor] = ..., future_values: Optional[torch.Tensor] = ..., output_hidden_states: Optional[bool] = ..., return_loss: bool = ..., return_dict: Optional[bool] = ...) -> PatchTSMixerForPredictionOutput:
        r"""
        observed_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]`:
                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
        future_values (`torch.FloatTensor` of shape `(batch_size, target_len, num_input_channels)` for forecasting,:
            `(batch_size, num_targets)` for regression, or `(batch_size,)` for classification, *optional*): Target
            values of the time series, that serve as labels for the model. The `future_values` is what the
            Transformer needs during training to learn to output, given the `past_values`. Note that, this is NOT
            required for a pretraining task.

            For a forecasting task, the shape is be `(batch_size, target_len, num_input_channels)`. Even if we want
            to forecast only specific channels by setting the indices in `prediction_channel_indices` parameter,
            pass the target data with all channels, as channel Filtering for both prediction and target will be
            manually applied before the loss computation.
        return_loss (`bool`,  *optional*):
            Whether to return the loss in the `forward` call.

        Returns:

        """
        ...

    def generate(self, past_values: torch.Tensor, observed_mask: Optional[torch.Tensor] = ...) -> SamplePatchTSMixerPredictionOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the future.

            observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Return:
            [`SamplePatchTSMixerPredictionOutput`] where the outputs `sequences` tensor will have shape `(batch_size,
            number of samples, prediction_length, num_input_channels)`.
        """
        ...



@dataclass
class PatchTSMixerForTimeSeriesClassificationOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForTimeSeriesClassificationOutput`].

    Args:
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            Prediction output from the classfication head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
    """
    loss: Optional[torch.FloatTensor] = ...
    prediction_outputs: torch.FloatTensor = ...
    last_hidden_state: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...


class PatchTSMixerForTimeSeriesClassification(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for classification application.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """
    def __init__(self, config: PatchTSMixerConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerForTimeSeriesClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, past_values: torch.Tensor, target_values: torch.Tensor = ..., output_hidden_states: Optional[bool] = ..., return_loss: bool = ..., return_dict: Optional[bool] = ...) -> PatchTSMixerForTimeSeriesClassificationOutput:
        r"""
        target_values (`torch.FloatTensor` of shape `(batch_size, target_len, num_input_channels)` for forecasting,
            `(batch_size, num_targets)` for regression, or `(batch_size,)` for classification, *optional*): Target
            values of the time series, that serve as labels for the model. The `target_values` is what the
            Transformer needs during training to learn to output, given the `past_values`. Note that, this is NOT
            required for a pretraining task.

            For a forecasting task, the shape is be `(batch_size, target_len, num_input_channels)`. Even if we want
            to forecast only specific channels by setting the indices in `prediction_channel_indices` parameter,
            pass the target data with all channels, as channel Filtering for both prediction and target will be
            manually applied before the loss computation.

            For a classification task, it has a shape of `(batch_size,)`.

            For a regression task, it has a shape of `(batch_size, num_targets)`.
        return_loss (`bool`, *optional*):
            Whether to return the loss in the `forward` call.

        Returns:

        """
        ...



@dataclass
class PatchTSMixerForRegressionOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForRegressionOutput`].

    Args:
        regression_outputs (`torch.FloatTensor` of shape `(batch_size, num_targets)`):
            Prediction output from the regression head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
    """
    loss: Optional[torch.FloatTensor] = ...
    regression_outputs: torch.FloatTensor = ...
    last_hidden_state: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...


class InjectScalerStatistics4D(nn.Module):
    def __init__(self, d_model: int, num_patches: int, expansion: int = ...) -> None:
        ...

    def forward(self, inputs: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor): # -> Tensor:
        """
        Args:
            inputs (`torch.Tensor` of shape `(batch_size, num_input_channels, num_patch, d_model)`)
            loc (`torch.Tensor` of shape `(batch_size, 1, num_input_channels)`)
            scale (`torch.Tensor` of shape `(batch_size, 1, num_input_channels)`)
        Returns:
            `torch.Tensor` of shape `(batch_size, num_input_channels, num_patch, d_model)`
        """
        ...



class PatchTSMixerForRegression(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for regression application.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """
    def __init__(self, config: PatchTSMixerConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerForRegressionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, past_values: torch.Tensor, target_values: torch.Tensor = ..., output_hidden_states: Optional[bool] = ..., return_loss: bool = ..., return_dict: Optional[bool] = ...) -> PatchTSMixerForRegressionOutput:
        r"""
        target_values (`torch.FloatTensor` of shape `(batch_size, target_len, num_input_channels)` for forecasting,
            `(batch_size, num_targets)` for regression, or `(batch_size,)` for classification, *optional*): Target
            values of the time series, that serve as labels for the model. The `target_values` is what the
            Transformer needs during training to learn to output, given the `past_values`. Note that, this is NOT
            required for a pretraining task.

            For a forecasting task, the shape is be `(batch_size, target_len, num_input_channels)`. Even if we want
            to forecast only specific channels by setting the indices in `prediction_channel_indices` parameter,
            pass the target data with all channels, as channel Filtering for both prediction and target will be
            manually applied before the loss computation.

            For a classification task, it has a shape of `(batch_size,)`.

            For a regression task, it has a shape of `(batch_size, num_targets)`.
        return_loss (`bool`, *optional*):
            Whether to return the loss in the `forward` call.

        Returns:

        """
        ...

    def generate(self, past_values: torch.Tensor) -> SamplePatchTSMixerRegressionOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the target values.

        Return:
            [`SamplePatchTSMixerRegressionOutput`] where the outputs `sequences` tensor will have shape `(batch_size,
            number of samples, num_targets)`.
        """
        ...
