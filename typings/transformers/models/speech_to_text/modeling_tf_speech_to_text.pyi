"""
This type stub file was generated by pyright.
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Union
from ...modeling_tf_outputs import TFSeq2SeqLMOutput, TFSeq2SeqModelOutput
from ...modeling_tf_utils import TFCausalLanguageModelingLoss, TFModelInputType, TFPreTrainedModel, keras, keras_serializable, unpack_inputs
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_speech_to_text import Speech2TextConfig

"""TensorFlow Speech2Text model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
LARGE_NEGATIVE = ...
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    ...

class TFConv1dSubsampler(keras.layers.Layer):
    """
    Convolutional subsampler: a stack of 1D convolution (along temporal dimension) followed by non-linear activation
    via gated linear units (https://arxiv.org/abs/1911.08460)
    """
    def __init__(self, config: Speech2TextConfig, **kwargs) -> None:
        ...
    
    def call(self, input_features: tf.Tensor) -> tf.Tensor:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


class TFSpeech2TextSinusoidalPositionalEmbedding(keras.layers.Layer):
    """This module produces sinusoidal positional embeddings of any length."""
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = ..., **kwargs) -> None:
        ...
    
    def call(self, input_ids: tf.Tensor, past_key_values_length: int = ...) -> tf.Tensor:
        ...
    
    @staticmethod
    def create_position_ids_from_input_ids(input_ids: tf.Tensor, padding_idx: int, past_key_values_length: Optional[int] = ...) -> tf.Tensor:
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: tf.Tensor x:
        Returns: tf.Tensor
        """
        ...
    


class TFSpeech2TextAttention(keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = ..., is_decoder: bool = ..., bias: bool = ..., **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, key_value_states: tf.Tensor | None = ..., past_key_value: Tuple[Tuple[tf.Tensor]] | None = ..., attention_mask: tf.Tensor | None = ..., layer_head_mask: tf.Tensor | None = ..., training: Optional[bool] = ...) -> Tuple[tf.Tensor, tf.Tensor | None]:
        """Input shape: Batch x Time x Channel"""
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


class TFSpeech2TextEncoderLayer(keras.layers.Layer):
    def __init__(self, config: Speech2TextConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, layer_head_mask: tf.Tensor, training: bool = ...): # -> tuple[Any, Any]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`tf.Tensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`
        """
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


class TFSpeech2TextDecoderLayer(keras.layers.Layer):
    def __init__(self, config: Speech2TextConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states, attention_mask: tf.Tensor | None = ..., encoder_hidden_states: tf.Tensor | None = ..., encoder_attention_mask: tf.Tensor | None = ..., layer_head_mask: tf.Tensor | None = ..., cross_attn_layer_head_mask: tf.Tensor | None = ..., past_key_value: Tuple[tf.Tensor] | None = ..., training=...) -> Tuple[tf.Tensor, tf.Tensor, Tuple[Tuple[tf.Tensor]]]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`tf.Tensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`tf.Tensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`tf.Tensor`): mask for attention heads in a given layer of size
                `(decoder_attention_heads,)`
            cross_attn_layer_head_mask (`tf.Tensor`): mask for heads of the cross-attention module.
                `(decoder_attention_heads,)`
            past_key_value (`Tuple(tf.Tensor)`): cached past key and value projection states
        """
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


class TFSpeech2TextPreTrainedModel(TFPreTrainedModel):
    config_class = Speech2TextConfig
    base_model_prefix = ...
    main_input_name = ...
    _keys_to_ignore_on_load_unexpected = ...
    @property
    def input_signature(self): # -> dict[str, Any]:
        ...
    


SPEECH_TO_TEXT_START_DOCSTRING = ...
SPEECH_TO_TEXT_INPUTS_DOCSTRING = ...
@keras_serializable
class TFSpeech2TextEncoder(keras.layers.Layer):
    config_class = Speech2TextConfig
    def __init__(self, config: Speech2TextConfig, **kwargs) -> None:
        ...
    
    @unpack_inputs
    def call(self, input_features=..., attention_mask=..., head_mask=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=...): # -> tuple[Any, ...] | TFBaseModelOutput:
        """
        Args:
            input_features (`tf.Tensor` of shape `(batch_size, sequence_length, feature_size)`):
                Float values of fbank features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the fbank features,
                padding and conversion into a tensor of floats. See [`~Speech2TextFeatureExtractor.__call__`]
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`tf.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, `optional):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


@keras_serializable
class TFSpeech2TextDecoder(keras.layers.Layer):
    config_class = Speech2TextConfig
    def __init__(self, config: Speech2TextConfig, **kwargs) -> None:
        ...
    
    def get_embed_tokens(self): # -> TFSharedEmbeddings:
        ...
    
    def set_embed_tokens(self, embed_tokens): # -> None:
        ...
    
    @unpack_inputs
    def call(self, input_ids=..., inputs_embeds=..., attention_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., head_mask=..., cross_attn_head_mask=..., past_key_values=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=...):
        r"""
        Args:
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`Speech2TextTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`tf.Tensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`tf.Tensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`tf.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`tf.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


@keras_serializable
class TFSpeech2TextMainLayer(keras.layers.Layer):
    config_class = Speech2TextConfig
    def __init__(self, config: Speech2TextConfig, **kwargs) -> None:
        ...
    
    def get_input_embeddings(self): # -> TFSharedEmbeddings:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    @unpack_inputs
    def call(self, input_features=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., head_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., encoder_outputs=..., past_key_values=..., decoder_inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs): # -> TFSeq2SeqModelOutput:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


@add_start_docstrings("The bare Speech2Text Model outputting raw hidden-states without any specific head on top.", SPEECH_TO_TEXT_START_DOCSTRING)
class TFSpeech2TextModel(TFSpeech2TextPreTrainedModel):
    def __init__(self, config: Speech2TextConfig, *inputs, **kwargs) -> None:
        ...
    
    def get_encoder(self): # -> TFSpeech2TextEncoder:
        ...
    
    def get_decoder(self): # -> TFSpeech2TextDecoder:
        ...
    
    @unpack_inputs
    @add_start_docstrings_to_model_forward(SPEECH_TO_TEXT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_features: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., decoder_input_ids: np.ndarray | tf.Tensor | None = ..., decoder_attention_mask: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., decoder_head_mask: np.ndarray | tf.Tensor | None = ..., cross_attn_head_mask: np.ndarray | tf.Tensor | None = ..., encoder_outputs: np.ndarray | tf.Tensor | None = ..., past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = ..., decoder_inputs_embeds: np.ndarray | tf.Tensor | None = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: bool = ..., **kwargs) -> Union[Tuple, TFSeq2SeqModelOutput]:
        ...
    
    def serving_output(self, output): # -> TFSeq2SeqModelOutput:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


@add_start_docstrings("The Speech2Text Model with a language modeling head. Can be used for summarization.", SPEECH_TO_TEXT_START_DOCSTRING)
class TFSpeech2TextForConditionalGeneration(TFSpeech2TextPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config: Speech2TextConfig) -> None:
        ...
    
    def get_encoder(self): # -> TFSpeech2TextEncoder:
        ...
    
    def get_decoder(self): # -> TFSpeech2TextDecoder:
        ...
    
    def resize_token_embeddings(self, new_num_tokens: int) -> tf.Variable:
        ...
    
    def get_output_embeddings(self):
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    @unpack_inputs
    @add_start_docstrings_to_model_forward(SPEECH_TO_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_features: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., decoder_input_ids: np.ndarray | tf.Tensor | None = ..., decoder_attention_mask: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., decoder_head_mask: np.ndarray | tf.Tensor | None = ..., cross_attn_head_mask: np.ndarray | tf.Tensor | None = ..., encoder_outputs: np.ndarray | tf.Tensor | None = ..., past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = ..., decoder_inputs_embeds: np.ndarray | tf.Tensor | None = ..., labels: np.ndarray | tf.Tensor | None = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ..., **kwargs) -> Union[Tuple, TFSeq2SeqLMOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import Speech2TextProcessor, TFSpeech2TextForConditionalGeneration
        >>> from datasets import load_dataset
        >>> import soundfile as sf

        >>> model = TFSpeech2TextForConditionalGeneration.from_pretrained(
        ...     "facebook/s2t-small-librispeech-asr", from_pt=True
        ... )
        >>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


        >>> def map_to_array(batch):
        ...     speech, _ = sf.read(batch["file"])
        ...     batch["speech"] = speech
        ...     return batch


        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)
        >>> ds.set_format(type="tf")

        >>> input_features = processor(
        ...     ds["speech"][0], sampling_rate=16000, return_tensors="tf"
        ... ).input_features  # Batch size 1
        >>> generated_ids = model.generate(input_features)

        >>> transcription = processor.batch_decode(generated_ids)
        ```"""
        ...
    
    def serving_output(self, output): # -> TFSeq2SeqLMOutput:
        ...
    
    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=..., attention_mask=..., head_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., use_cache=..., encoder_outputs=..., **kwargs): # -> dict[str, Any]:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    
    def tf_to_pt_weight_rename(self, tf_weight): # -> tuple[Literal['lm_head.weight'], Literal['model.decoder.embed_tokens.weight']] | tuple[Any]:
        ...
    


