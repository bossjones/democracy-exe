"""
This type stub file was generated by pyright.
"""

from contextlib import contextmanager
from ...processing_utils import ProcessorMixin

"""
Processor class for Donut.
"""
class DonutProcessor(ProcessorMixin):
    r"""
    Constructs a Donut processor which wraps a Donut image processor and an XLMRoBERTa tokenizer into a single
    processor.

    [`DonutProcessor`] offers all the functionalities of [`DonutImageProcessor`] and
    [`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`]. See the [`~DonutProcessor.__call__`] and
    [`~DonutProcessor.decode`] for more information.

    Args:
        image_processor ([`DonutImageProcessor`], *optional*):
            An instance of [`DonutImageProcessor`]. The image processor is a required input.
        tokenizer ([`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`], *optional*):
            An instance of [`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`]. The tokenizer is a required input.
    """
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor=..., tokenizer=..., **kwargs) -> None:
        ...
    
    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to AutoImageProcessor's
        [`~AutoImageProcessor.__call__`] and returns its output. If used in the context
        [`~DonutProcessor.as_target_processor`] this method forwards all its arguments to DonutTokenizer's
        [`~DonutTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more information.
        """
        ...
    
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to DonutTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please refer
        to the docstring of this method for more information.
        """
        ...
    
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to DonutTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the
        docstring of this method for more information.
        """
        ...
    
    @contextmanager
    def as_target_processor(self): # -> Generator[None, Any, None]:
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning TrOCR.
        """
        ...
    
    def token2json(self, tokens, is_inner_value=..., added_vocab=...): # -> list[dict[Any, Any]] | dict[Any, Any] | dict[str, Any] | list[Any]:
        """
        Convert a (generated) token sequence into an ordered JSON format.
        """
        ...
    
    @property
    def feature_extractor_class(self): # -> str:
        ...
    
    @property
    def feature_extractor(self):
        ...
    


