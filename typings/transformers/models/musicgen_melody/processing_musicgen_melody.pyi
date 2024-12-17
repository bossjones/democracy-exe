"""
This type stub file was generated by pyright.
"""

from ...processing_utils import ProcessorMixin

"""
Text/audio processor class for MusicGen Melody
"""
class MusicgenMelodyProcessor(ProcessorMixin):
    r"""
    Constructs a MusicGen Melody processor which wraps a Wav2Vec2 feature extractor - for raw audio waveform processing - and a T5 tokenizer into a single processor
    class.

    [`MusicgenProcessor`] offers all the functionalities of [`MusicgenMelodyFeatureExtractor`] and [`T5Tokenizer`]. See
    [`~MusicgenProcessor.__call__`] and [`~MusicgenProcessor.decode`] for more information.

    Args:
        feature_extractor (`MusicgenMelodyFeatureExtractor`):
            An instance of [`MusicgenMelodyFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`T5Tokenizer`):
            An instance of [`T5Tokenizer`]. The tokenizer is a required input.
    """
    feature_extractor_class = ...
    tokenizer_class = ...
    def __init__(self, feature_extractor, tokenizer) -> None:
        ...
    
    def get_decoder_prompt_ids(self, task=..., language=..., no_timestamps=...):
        ...
    
    def __call__(self, audio=..., text=..., **kwargs):
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `audio`
        and `kwargs` arguments to MusicgenMelodyFeatureExtractor's [`~MusicgenMelodyFeatureExtractor.__call__`] if `audio` is not
        `None` to pre-process the audio. It also forwards the `text` and `kwargs` arguments to
        PreTrainedTokenizer's [`~PreTrainedTokenizer.__call__`] if `text` is not `None`. Please refer to the doctsring of the above two methods for more information.

        Args:
            audio (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The audio or batch of audios to be prepared. Each audio can be NumPy array or PyTorch tensor. In case
                of a NumPy array/PyTorch tensor, each audio should be a mono-stereo signal of shape (T), where T is the sample length of the audio.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to the feature extractor and/or the
                tokenizer.
        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:
            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **input_features** -- Audio input features to be fed to a model. Returned when `audio` is not `None`.
            - **attention_mask** -- List of token indices specifying which tokens should be attended to by the model when `text` is not `None`.
            When only `audio` is specified, returns the timestamps attention mask.
        """
        ...
    
    def batch_decode(self, *args, **kwargs): # -> List[ndarray[Any, Any]]:
        """
        This method is used to decode either batches of audio outputs from the MusicGen model, or batches of token ids
        from the tokenizer. In the case of decoding token ids, this method forwards all its arguments to T5Tokenizer's
        [`~PreTrainedTokenizer.batch_decode`]. Please refer to the docstring of this method for more information.
        """
        ...
    
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to T5Tokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the
        docstring of this method for more information.
        """
        ...
    
    def get_unconditional_inputs(self, num_samples=..., return_tensors=...):
        """
        Helper function to get null inputs for unconditional generation, enabling the model to be used without the
        feature extractor or tokenizer.

        Args:
            num_samples (int, *optional*):
                Number of audio samples to unconditionally generate.

        Example:
        ```python
        >>> from transformers import MusicgenMelodyForConditionalGeneration, MusicgenMelodyProcessor

        >>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

        >>> # get the unconditional (or 'null') inputs for the model
        >>> processor = MusicgenMelodyProcessor.from_pretrained("facebook/musicgen-melody")
        >>> unconditional_inputs = processor.get_unconditional_inputs(num_samples=1)

        >>> audio_samples = model.generate(**unconditional_inputs, max_new_tokens=256)
        ```"""
        ...
    


