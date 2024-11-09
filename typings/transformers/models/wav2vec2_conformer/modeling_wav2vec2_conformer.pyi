"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn

from ...modeling_outputs import (
    CausalLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from .configuration_wav2vec2_conformer import Wav2Vec2ConformerConfig

""" PyTorch Wav2Vec2-Conformer model."""
logger = ...
_HIDDEN_STATES_START_POSITION = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_CTC_EXPECTED_OUTPUT = ...
_CTC_EXPECTED_LOSS = ...
@dataclass
class Wav2Vec2ConformerForPreTrainingOutput(ModelOutput):
    """
    Output type of [`Wav2Vec2ConformerForPreTraining`], with potential hidden states and attentions.

    Args:
        loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the [official
            paper](https://arxiv.org/pdf/2006.11477.pdf) . (classification) loss.
        projected_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Hidden-states of the model projected to *config.proj_codevector_dim* that can be used to predict the masked
            projected quantized states.
        projected_quantized_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Quantized extracted feature vectors projected to *config.proj_codevector_dim* representing the positive
            target vectors for contrastive loss.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        contrastive_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            The contrastive loss (L_m) as stated in the [official paper](https://arxiv.org/pdf/2006.11477.pdf) .
        diversity_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            The diversity loss (L_d) as stated in the [official paper](https://arxiv.org/pdf/2006.11477.pdf) .
    """
    loss: Optional[torch.FloatTensor] = ...
    projected_states: torch.FloatTensor = ...
    projected_quantized_states: torch.FloatTensor = ...
    codevector_perplexity: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...
    contrastive_loss: Optional[torch.FloatTensor] = ...
    diversity_loss: Optional[torch.FloatTensor] = ...


class Wav2Vec2ConformerNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...

    def forward(self, hidden_states):
        ...



class Wav2Vec2ConformerLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...

    def forward(self, hidden_states):
        ...



class Wav2Vec2ConformerGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...

    def forward(self, hidden_states):
        ...



class Wav2Vec2ConformerPositionalConvEmbedding(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states):
        ...



class Wav2Vec2ConformerRotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding
    Reference : https://blog.eleuther.ai/rotary-embeddings/ Paper: https://arxiv.org/pdf/2104.09864.pdf
    """
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> Tensor:
        ...



class Wav2Vec2ConformerRelPositionalEmbedding(nn.Module):
    """Relative positional encoding module."""
    def __init__(self, config) -> None:
        ...

    def extend_pe(self, x): # -> None:
        ...

    def forward(self, hidden_states: torch.Tensor): # -> Tensor:
        ...



class Wav2Vec2ConformerSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings) -> None:
        ...

    def forward(self, hidden_states):
        ...



class Wav2Vec2ConformerFeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""
    def __init__(self, config) -> None:
        ...

    def forward(self, input_values): # -> Any:
        ...



class Wav2Vec2ConformerFeatureProjection(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> tuple[Any, Any]:
        ...



class Wav2Vec2ConformerFeedForward(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> Any:
        ...



class Wav2Vec2ConformerConvolutionModule(nn.Module):
    """Convolution block used in the conformer block"""
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> Any:
        ...



class Wav2Vec2ConformerSelfAttention(nn.Module):
    """Construct an Wav2Vec2ConformerSelfAttention object.
    Can be enhanced with rotary or relative position embeddings.
    """
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., relative_position_embeddings: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        ...



class Wav2Vec2ConformerEncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = ..., relative_position_embeddings: Optional[torch.Tensor] = ..., output_attentions: bool = ...): # -> tuple[Any, Any]:
        ...



class Wav2Vec2ConformerEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states, attention_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> tuple[Any, ...] | BaseModelOutput:
        ...



class Wav2Vec2ConformerGumbelVectorQuantizer(nn.Module):
    """
    Vector quantization using gumbel softmax. See `[CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
    """
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states, mask_time_indices=...): # -> tuple[Tensor | Any, Tensor]:
        ...



class Wav2Vec2ConformerAdapter(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> Any:
        ...



class Wav2Vec2ConformerAdapterLayer(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> Tensor:
        ...



class Wav2Vec2ConformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Wav2Vec2ConformerConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...


WAV2VEC2_CONFORMER_START_DOCSTRING = ...
WAV2VEC2_CONFORMER_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Wav2Vec2Conformer Model transformer outputting raw hidden-states without any specific head on top.", WAV2VEC2_CONFORMER_START_DOCSTRING)
class Wav2Vec2ConformerModel(Wav2Vec2ConformerPreTrainedModel):
    def __init__(self, config: Wav2Vec2ConformerConfig) -> None:
        ...

    def freeze_feature_encoder(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        ...

    @add_start_docstrings_to_model_forward(WAV2VEC2_CONFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=Wav2Vec2BaseModelOutput, config_class=_CONFIG_FOR_DOC, modality="audio", expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = ..., mask_time_indices: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        ...



@add_start_docstrings("""Wav2Vec2Conformer Model with a quantizer and `VQ` head on top.""", WAV2VEC2_CONFORMER_START_DOCSTRING)
class Wav2Vec2ConformerForPreTraining(Wav2Vec2ConformerPreTrainedModel):
    def __init__(self, config: Wav2Vec2ConformerConfig) -> None:
        ...

    def set_gumbel_temperature(self, temperature: int): # -> None:
        """
        Set the Gumbel softmax temperature to a given value. Only necessary for training
        """
        ...

    def freeze_feature_encoder(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        ...

    @staticmethod
    def compute_contrastive_logits(target_features: torch.FloatTensor, negative_features: torch.FloatTensor, predicted_features: torch.FloatTensor, temperature: int = ...): # -> Tensor:
        """
        Compute logits for contrastive loss based using cosine similarity as the distance measure between
        `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
        """
        ...

    @add_start_docstrings_to_model_forward(WAV2VEC2_CONFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Wav2Vec2ConformerForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = ..., mask_time_indices: Optional[torch.BoolTensor] = ..., sampled_negative_indices: Optional[torch.BoolTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, Wav2Vec2ConformerForPreTrainingOutput]:
        r"""
        mask_time_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
            masked extracted features in *config.proj_codevector_dim* space.
        sampled_negative_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_negatives)`, *optional*):
            Indices indicating which quantized target vectors are used as negative sampled vectors in contrastive loss.
            Required input for pre-training.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoFeatureExtractor, Wav2Vec2ConformerForPreTraining
        >>> from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import _compute_mask_indices, _sample_negative_indices
        >>> from datasets import load_dataset

        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large")
        >>> model = Wav2Vec2ConformerForPreTraining.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> input_values = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt").input_values  # Batch size 1

        >>> # compute masked indices
        >>> batch_size, raw_sequence_length = input_values.shape
        >>> sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length).item()
        >>> mask_time_indices = _compute_mask_indices(
        ...     shape=(batch_size, sequence_length), mask_prob=0.2, mask_length=2
        ... )
        >>> sampled_negative_indices = _sample_negative_indices(
        ...     features_shape=(batch_size, sequence_length),
        ...     num_negatives=model.config.num_negatives,
        ...     mask_time_indices=mask_time_indices,
        ... )
        >>> mask_time_indices = torch.tensor(data=mask_time_indices, device=input_values.device, dtype=torch.long)
        >>> sampled_negative_indices = torch.tensor(
        ...     data=sampled_negative_indices, device=input_values.device, dtype=torch.long
        ... )

        >>> with torch.no_grad():
        ...     outputs = model(input_values, mask_time_indices=mask_time_indices)

        >>> # compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)
        >>> cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)

        >>> # show that cosine similarity is much higher than random
        >>> cosine_sim[mask_time_indices.to(torch.bool)].mean() > 0.5
        tensor(True)

        >>> # for contrastive loss training model should be put into train mode
        >>> model = model.train()
        >>> loss = model(
        ...     input_values, mask_time_indices=mask_time_indices, sampled_negative_indices=sampled_negative_indices
        ... ).loss
        ```"""
        ...



@add_start_docstrings("""Wav2Vec2Conformer Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""", WAV2VEC2_CONFORMER_START_DOCSTRING)
class Wav2Vec2ConformerForCTC(Wav2Vec2ConformerPreTrainedModel):
    def __init__(self, config, target_lang: Optional[str] = ...) -> None:
        ...

    def freeze_feature_encoder(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        ...

    @add_start_docstrings_to_model_forward(WAV2VEC2_CONFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC, expected_output=_CTC_EXPECTED_OUTPUT, expected_loss=_CTC_EXPECTED_LOSS)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[torch.Tensor] = ...) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        ...



@add_start_docstrings("""
    Wav2Vec2Conformer Model with a sequence classification head on top (a linear layer over the pooled output) for
    tasks like SUPERB Keyword Spotting.
    """, WAV2VEC2_CONFORMER_START_DOCSTRING)
class Wav2Vec2ConformerForSequenceClassification(Wav2Vec2ConformerPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    def freeze_feature_encoder(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        ...

    def freeze_base_model(self): # -> None:
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        ...

    @add_start_docstrings_to_model_forward(WAV2VEC2_CONFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC, modality="audio")
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[torch.Tensor] = ...) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...



@add_start_docstrings("""
    Wav2Vec2Conformer Model with a frame classification head on top for tasks like Speaker Diarization.
    """, WAV2VEC2_CONFORMER_START_DOCSTRING)
class Wav2Vec2ConformerForAudioFrameClassification(Wav2Vec2ConformerPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    def freeze_feature_encoder(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        ...

    def freeze_base_model(self): # -> None:
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        ...

    @add_start_docstrings_to_model_forward(WAV2VEC2_CONFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC, modality="audio")
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = ..., labels: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...



class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=..., margin=...) -> None:
        ...

    def forward(self, hidden_states, labels): # -> Any:
        ...



class TDNNLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...



@add_start_docstrings("""
    Wav2Vec2Conformer Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """, WAV2VEC2_CONFORMER_START_DOCSTRING)
class Wav2Vec2ConformerForXVector(Wav2Vec2ConformerPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    def freeze_feature_encoder(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        ...

    def freeze_base_model(self): # -> None:
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        ...

    @add_start_docstrings_to_model_forward(WAV2VEC2_CONFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=XVectorOutput, config_class=_CONFIG_FOR_DOC, modality="audio")
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[torch.Tensor] = ...) -> Union[Tuple, XVectorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
