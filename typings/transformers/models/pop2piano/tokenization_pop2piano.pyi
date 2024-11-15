"""
This type stub file was generated by pyright.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import pretty_midi

from ...feature_extraction_utils import BatchFeature
from ...tokenization_utils import BatchEncoding, PaddingStrategy, PreTrainedTokenizer, TruncationStrategy
from ...utils import TensorType, is_pretty_midi_available

"""Tokenization class for Pop2Piano."""
if is_pretty_midi_available():
    ...
logger = ...
VOCAB_FILES_NAMES = ...
def token_time_to_note(number, cutoff_time_idx, current_idx):
    ...

def token_note_to_note(number, current_velocity, default_velocity, note_onsets_ready, current_idx, notes):
    ...

class Pop2PianoTokenizer(PreTrainedTokenizer):
    """
    Constructs a Pop2Piano tokenizer. This tokenizer does not require training.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab (`str`):
            Path to the vocab file which contains the vocabulary.
        default_velocity (`int`, *optional*, defaults to 77):
            Determines the default velocity to be used while creating midi Notes.
        num_bars (`int`, *optional*, defaults to 2):
            Determines cutoff_time_idx in for each token.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"-1"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to 1):
            The end of sequence token.
        pad_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to 0):
             A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to 2):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
    """
    model_input_names = ...
    vocab_files_names = ...
    def __init__(self, vocab, default_velocity=..., num_bars=..., unk_token=..., eos_token=..., pad_token=..., bos_token=..., **kwargs) -> None:
        ...

    @property
    def vocab_size(self): # -> int:
        """Returns the vocabulary size of the tokenizer."""
        ...

    def get_vocab(self): # -> dict[Any, Any]:
        """Returns the vocabulary of the tokenizer."""
        ...

    def relative_batch_tokens_ids_to_notes(self, tokens: np.ndarray, beat_offset_idx: int, bars_per_batch: int, cutoff_time_idx: int): # -> list[Any] | ndarray[Any, dtype[Any]]:
        """
        Converts relative tokens to notes which are then used to generate pretty midi object.

        Args:
            tokens (`numpy.ndarray`):
                Tokens to be converted to notes.
            beat_offset_idx (`int`):
                Denotes beat offset index for each note in generated Midi.
            bars_per_batch (`int`):
                A parameter to control the Midi output generation.
            cutoff_time_idx (`int`):
                Denotes the cutoff time index for each note in generated Midi.
        """
        ...

    def relative_batch_tokens_ids_to_midi(self, tokens: np.ndarray, beatstep: np.ndarray, beat_offset_idx: int = ..., bars_per_batch: int = ..., cutoff_time_idx: int = ...):
        """
        Converts tokens to Midi. This method calls `relative_batch_tokens_ids_to_notes` method to convert batch tokens
        to notes then uses `notes_to_midi` method to convert them to Midi.

        Args:
            tokens (`numpy.ndarray`):
                Denotes tokens which alongside beatstep will be converted to Midi.
            beatstep (`np.ndarray`):
                We get beatstep from feature extractor which is also used to get Midi.
            beat_offset_idx (`int`, *optional*, defaults to 0):
                Denotes beat offset index for each note in generated Midi.
            bars_per_batch (`int`, *optional*, defaults to 2):
                A parameter to control the Midi output generation.
            cutoff_time_idx (`int`, *optional*, defaults to 12):
                Denotes the cutoff time index for each note in generated Midi.
        """
        ...

    def relative_tokens_ids_to_notes(self, tokens: np.ndarray, start_idx: float, cutoff_time_idx: float = ...): # -> list[Any] | ndarray[Any, dtype[Any]]:
        """
        Converts relative tokens to notes which will then be used to create Pretty Midi objects.

        Args:
            tokens (`numpy.ndarray`):
                Relative Tokens which will be converted to notes.
            start_idx (`float`):
                A parameter which denotes the starting index.
            cutoff_time_idx (`float`, *optional*):
                A parameter used while converting tokens to notes.
        """
        ...

    def notes_to_midi(self, notes: np.ndarray, beatstep: np.ndarray, offset_sec: int = ...):
        """
        Converts notes to Midi.

        Args:
            notes (`numpy.ndarray`):
                This is used to create Pretty Midi objects.
            beatstep (`numpy.ndarray`):
                This is the extrapolated beatstep that we get from feature extractor.
            offset_sec (`int`, *optional*, defaults to 0.0):
                This represents the offset seconds which is used while creating each Pretty Midi Note.
        """
        ...

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        """
        Saves the tokenizer's vocabulary dictionary to the provided save_directory.

        Args:
            save_directory (`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.
            filename_prefix (`Optional[str]`, *optional*):
                A prefix to add to the names of the files saved by the tokenizer.
        """
        ...

    def encode_plus(self, notes: Union[np.ndarray, List[pretty_midi.Note]], truncation_strategy: Optional[TruncationStrategy] = ..., max_length: Optional[int] = ..., **kwargs) -> BatchEncoding:
        r"""
        This is the `encode_plus` method for `Pop2PianoTokenizer`. It converts the midi notes to the transformer
        generated token ids. It only works on a single batch, to process multiple batches please use
        `batch_encode_plus` or `__call__` method.

        Args:
            notes (`numpy.ndarray` of shape `[sequence_length, 4]` or `list` of `pretty_midi.Note` objects):
                This represents the midi notes. If `notes` is a `numpy.ndarray`:
                    - Each sequence must have 4 values, they are `onset idx`, `offset idx`, `pitch` and `velocity`.
                If `notes` is a `list` containing `pretty_midi.Note` objects:
                    - Each sequence must have 4 attributes, they are `start`, `end`, `pitch` and `velocity`.
            truncation_strategy ([`~tokenization_utils_base.TruncationStrategy`], *optional*):
                Indicates the truncation strategy that is going to be used during truncation.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).

        Returns:
            `BatchEncoding` containing the tokens ids.
        """
        ...

    def batch_encode_plus(self, notes: Union[np.ndarray, List[pretty_midi.Note]], truncation_strategy: Optional[TruncationStrategy] = ..., max_length: Optional[int] = ..., **kwargs) -> BatchEncoding:
        r"""
        This is the `batch_encode_plus` method for `Pop2PianoTokenizer`. It converts the midi notes to the transformer
        generated token ids. It works on multiple batches by calling `encode_plus` multiple times in a loop.

        Args:
            notes (`numpy.ndarray` of shape `[batch_size, sequence_length, 4]` or `list` of `pretty_midi.Note` objects):
                This represents the midi notes. If `notes` is a `numpy.ndarray`:
                    - Each sequence must have 4 values, they are `onset idx`, `offset idx`, `pitch` and `velocity`.
                If `notes` is a `list` containing `pretty_midi.Note` objects:
                    - Each sequence must have 4 attributes, they are `start`, `end`, `pitch` and `velocity`.
            truncation_strategy ([`~tokenization_utils_base.TruncationStrategy`], *optional*):
                Indicates the truncation strategy that is going to be used during truncation.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).

        Returns:
            `BatchEncoding` containing the tokens ids.
        """
        ...

    def __call__(self, notes: Union[np.ndarray, List[pretty_midi.Note], List[List[pretty_midi.Note]],], padding: Union[bool, str, PaddingStrategy] = ..., truncation: Union[bool, str, TruncationStrategy] = ..., max_length: Optional[int] = ..., pad_to_multiple_of: Optional[int] = ..., return_attention_mask: Optional[bool] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., verbose: bool = ..., **kwargs) -> BatchEncoding:
        r"""
        This is the `__call__` method for `Pop2PianoTokenizer`. It converts the midi notes to the transformer generated
        token ids.

        Args:
            notes (`numpy.ndarray` of shape `[batch_size, max_sequence_length, 4]` or `list` of `pretty_midi.Note` objects):
                This represents the midi notes.

                If `notes` is a `numpy.ndarray`:
                    - Each sequence must have 4 values, they are `onset idx`, `offset idx`, `pitch` and `velocity`.
                If `notes` is a `list` containing `pretty_midi.Note` objects:
                    - Each sequence must have 4 attributes, they are `start`, `end`, `pitch` and `velocity`.
            padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `False`):
                Activates and controls padding. Accepts the following values:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
                Activates and controls truncation. Accepts the following values:

                - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will
                  truncate token by token, removing a token from the longest sequence in the pair if a pair of
                  sequences (or a batch of pairs) is provided.
                - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
                  greater than the model maximum admissible input size).
            max_length (`int`, *optional*):
                Controls the maximum length to use by one of the truncation/padding parameters. If left unset or set to
                `None`, this will use the predefined model maximum length if a maximum length is required by one of the
                truncation/padding parameters. If the model has no specific maximum input length (like XLNet)
                truncation/padding to a maximum length will be deactivated.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the `return_outputs` attribute.

                [What are attention masks?](../glossary#attention-mask)
            return_tensors (`str` or [`~file_utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.

        Returns:
            `BatchEncoding` containing the token_ids.
        """
        ...

    def batch_decode(self, token_ids, feature_extractor_output: BatchFeature, return_midi: bool = ...): # -> BatchEncoding:
        r"""
        This is the `batch_decode` method for `Pop2PianoTokenizer`. It converts the token_ids generated by the
        transformer to midi_notes and returns them.

        Args:
            token_ids (`Union[np.ndarray, torch.Tensor, tf.Tensor]`):
                Output token_ids of `Pop2PianoConditionalGeneration` model.
            feature_extractor_output (`BatchFeature`):
                Denotes the output of `Pop2PianoFeatureExtractor.__call__`. It must contain `"beatstep"` and
                `"extrapolated_beatstep"`. Also `"attention_mask_beatsteps"` and
                `"attention_mask_extrapolated_beatstep"`
                 should be present if they were returned by the feature extractor.
            return_midi (`bool`, *optional*, defaults to `True`):
                Whether to return midi object or not.
        Returns:
            If `return_midi` is True:
                - `BatchEncoding` containing both `notes` and `pretty_midi.pretty_midi.PrettyMIDI` objects.
            If `return_midi` is False:
                - `BatchEncoding` containing `notes`.
        """
        ...
