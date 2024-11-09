"""
This type stub file was generated by pyright.
"""

from ...processing_utils import ProcessorMixin

"""
Speech processor class for Whisper
"""
class WhisperProcessor(ProcessorMixin):
    r"""
    Constructs a Whisper processor which wraps a Whisper feature extractor and a Whisper tokenizer into a single
    processor.

    [`WhisperProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`] and [`WhisperTokenizer`]. See
    the [`~WhisperProcessor.__call__`] and [`~WhisperProcessor.decode`] for more information.

    Args:
        feature_extractor (`WhisperFeatureExtractor`):
            An instance of [`WhisperFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`WhisperTokenizer`):
            An instance of [`WhisperTokenizer`]. The tokenizer is a required input.
    """
    feature_extractor_class = ...
    tokenizer_class = ...
    def __init__(self, feature_extractor, tokenizer) -> None:
        ...

    def get_decoder_prompt_ids(self, task=..., language=..., no_timestamps=...):
        ...

    def __call__(self, *args, **kwargs):
        """
        Forwards the `audio` argument to WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] and the `text`
        argument to [`~WhisperTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more
        information.
        """
        ...

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to WhisperTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        ...

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to WhisperTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        ...

    def get_prompt_ids(self, text: str, return_tensors=...):
        ...
