"""
This type stub file was generated by pyright.
"""

from typing import Optional, Tuple, Union

import torch
from torch import nn

from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_layoutlm import LayoutLMConfig

""" PyTorch LayoutLM model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
LayoutLMLayerNorm = nn.LayerNorm
class LayoutLMEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config) -> None:
        ...

    def forward(self, input_ids=..., bbox=..., token_type_ids=..., position_ids=..., inputs_embeds=...): # -> Any:
        ...



class LayoutLMSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None:
        ...

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...



class LayoutLMSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...



class LayoutLMAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None:
        ...

    def prune_heads(self, heads): # -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...



class LayoutLMIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...



class LayoutLMOutput(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...



class LayoutLMLayer(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...

    def feed_forward_chunk(self, attention_output): # -> Any:
        ...



class LayoutLMEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        ...



class LayoutLMPooler(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...



class LayoutLMPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...



class LayoutLMLMPredictionHead(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> Any:
        ...



class LayoutLMOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        ...



class LayoutLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LayoutLMConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...


LAYOUTLM_START_DOCSTRING = ...
LAYOUTLM_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare LayoutLM Model transformer outputting raw hidden-states without any specific head on top.", LAYOUTLM_START_DOCSTRING)
class LayoutLMModel(LayoutLMPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    def get_input_embeddings(self): # -> Embedding:
        ...

    def set_input_embeddings(self, value): # -> None:
        ...

    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., bbox: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, LayoutLMModel
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        >>> model = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")

        >>> words = ["Hello", "world"]
        >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

        >>> token_boxes = []
        >>> for word, box in zip(words, normalized_word_boxes):
        ...     word_tokens = tokenizer.tokenize(word)
        ...     token_boxes.extend([box] * len(word_tokens))
        >>> # add bounding boxes of cls + sep tokens
        >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        >>> encoding = tokenizer(" ".join(words), return_tensors="pt")
        >>> input_ids = encoding["input_ids"]
        >>> attention_mask = encoding["attention_mask"]
        >>> token_type_ids = encoding["token_type_ids"]
        >>> bbox = torch.tensor([token_boxes])

        >>> outputs = model(
        ...     input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids
        ... )

        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        ...



@add_start_docstrings("""LayoutLM Model with a `language modeling` head on top.""", LAYOUTLM_START_DOCSTRING)
class LayoutLMForMaskedLM(LayoutLMPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None:
        ...

    def get_input_embeddings(self): # -> Embedding:
        ...

    def get_output_embeddings(self): # -> Linear:
        ...

    def set_output_embeddings(self, new_embeddings): # -> None:
        ...

    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., bbox: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, LayoutLMForMaskedLM
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        >>> model = LayoutLMForMaskedLM.from_pretrained("microsoft/layoutlm-base-uncased")

        >>> words = ["Hello", "[MASK]"]
        >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

        >>> token_boxes = []
        >>> for word, box in zip(words, normalized_word_boxes):
        ...     word_tokens = tokenizer.tokenize(word)
        ...     token_boxes.extend([box] * len(word_tokens))
        >>> # add bounding boxes of cls + sep tokens
        >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        >>> encoding = tokenizer(" ".join(words), return_tensors="pt")
        >>> input_ids = encoding["input_ids"]
        >>> attention_mask = encoding["attention_mask"]
        >>> token_type_ids = encoding["token_type_ids"]
        >>> bbox = torch.tensor([token_boxes])

        >>> labels = tokenizer("Hello world", return_tensors="pt")["input_ids"]

        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     bbox=bbox,
        ...     attention_mask=attention_mask,
        ...     token_type_ids=token_type_ids,
        ...     labels=labels,
        ... )

        >>> loss = outputs.loss
        ```"""
        ...



@add_start_docstrings("""
    LayoutLM Model with a sequence classification head on top (a linear layer on top of the pooled output) e.g. for
    document image classification tasks such as the [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.
    """, LAYOUTLM_START_DOCSTRING)
class LayoutLMForSequenceClassification(LayoutLMPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    def get_input_embeddings(self): # -> Embedding:
        ...

    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., bbox: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, LayoutLMForSequenceClassification
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        >>> model = LayoutLMForSequenceClassification.from_pretrained("microsoft/layoutlm-base-uncased")

        >>> words = ["Hello", "world"]
        >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

        >>> token_boxes = []
        >>> for word, box in zip(words, normalized_word_boxes):
        ...     word_tokens = tokenizer.tokenize(word)
        ...     token_boxes.extend([box] * len(word_tokens))
        >>> # add bounding boxes of cls + sep tokens
        >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        >>> encoding = tokenizer(" ".join(words), return_tensors="pt")
        >>> input_ids = encoding["input_ids"]
        >>> attention_mask = encoding["attention_mask"]
        >>> token_type_ids = encoding["token_type_ids"]
        >>> bbox = torch.tensor([token_boxes])
        >>> sequence_label = torch.tensor([1])

        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     bbox=bbox,
        ...     attention_mask=attention_mask,
        ...     token_type_ids=token_type_ids,
        ...     labels=sequence_label,
        ... )

        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        ...



@add_start_docstrings("""
    LayoutLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    sequence labeling (information extraction) tasks such as the [FUNSD](https://guillaumejaume.github.io/FUNSD/)
    dataset and the [SROIE](https://rrc.cvc.uab.es/?ch=13) dataset.
    """, LAYOUTLM_START_DOCSTRING)
class LayoutLMForTokenClassification(LayoutLMPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    def get_input_embeddings(self): # -> Embedding:
        ...

    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., bbox: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, LayoutLMForTokenClassification
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        >>> model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")

        >>> words = ["Hello", "world"]
        >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

        >>> token_boxes = []
        >>> for word, box in zip(words, normalized_word_boxes):
        ...     word_tokens = tokenizer.tokenize(word)
        ...     token_boxes.extend([box] * len(word_tokens))
        >>> # add bounding boxes of cls + sep tokens
        >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        >>> encoding = tokenizer(" ".join(words), return_tensors="pt")
        >>> input_ids = encoding["input_ids"]
        >>> attention_mask = encoding["attention_mask"]
        >>> token_type_ids = encoding["token_type_ids"]
        >>> bbox = torch.tensor([token_boxes])
        >>> token_labels = torch.tensor([1, 1, 0, 0]).unsqueeze(0)  # batch size of 1

        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     bbox=bbox,
        ...     attention_mask=attention_mask,
        ...     token_type_ids=token_type_ids,
        ...     labels=token_labels,
        ... )

        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        ...



@add_start_docstrings("""
    LayoutLM Model with a span classification head on top for extractive question-answering tasks such as
    [DocVQA](https://rrc.cvc.uab.es/?ch=17) (a linear layer on top of the final hidden-states output to compute `span
    start logits` and `span end logits`).
    """, LAYOUTLM_START_DOCSTRING)
class LayoutLMForQuestionAnswering(LayoutLMPreTrainedModel):
    def __init__(self, config, has_visual_segment_embedding=...) -> None:
        ...

    def get_input_embeddings(self): # -> Embedding:
        ...

    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., bbox: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., start_positions: Optional[torch.LongTensor] = ..., end_positions: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Example:

        In the example below, we prepare a question + context pair for the LayoutLM model. It will give us a prediction
        of what it thinks the answer is (the span of the answer within the texts parsed from the image).

        ```python
        >>> from transformers import AutoTokenizer, LayoutLMForQuestionAnswering
        >>> from datasets import load_dataset
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("impira/layoutlm-document-qa", add_prefix_space=True)
        >>> model = LayoutLMForQuestionAnswering.from_pretrained("impira/layoutlm-document-qa", revision="1e3ebac")

        >>> dataset = load_dataset("nielsr/funsd", split="train")
        >>> example = dataset[0]
        >>> question = "what's his name?"
        >>> words = example["words"]
        >>> boxes = example["bboxes"]

        >>> encoding = tokenizer(
        ...     question.split(), words, is_split_into_words=True, return_token_type_ids=True, return_tensors="pt"
        ... )
        >>> bbox = []
        >>> for i, s, w in zip(encoding.input_ids[0], encoding.sequence_ids(0), encoding.word_ids(0)):
        ...     if s == 1:
        ...         bbox.append(boxes[w])
        ...     elif i == tokenizer.sep_token_id:
        ...         bbox.append([1000] * 4)
        ...     else:
        ...         bbox.append([0] * 4)
        >>> encoding["bbox"] = torch.tensor([bbox])

        >>> word_ids = encoding.word_ids(0)
        >>> outputs = model(**encoding)
        >>> loss = outputs.loss
        >>> start_scores = outputs.start_logits
        >>> end_scores = outputs.end_logits
        >>> start, end = word_ids[start_scores.argmax(-1)], word_ids[end_scores.argmax(-1)]
        >>> print(" ".join(words[start : end + 1]))
        M. Hamann P. Harper, P. Martinez
        ```"""
        ...
