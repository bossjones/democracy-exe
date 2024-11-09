"""
This type stub file was generated by pyright.
"""

from typing import List, Union

from ..utils import add_end_docstrings
from .base import ArgumentHandler, ChunkPipeline, build_pipeline_init_args

logger = ...
class ZeroShotClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for zero-shot for text classification by turning each possible label into an NLI
    premise/hypothesis pair.
    """
    def __call__(self, sequences, labels, hypothesis_template): # -> tuple[list[Any], list[str] | Any]:
        ...



@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))
class ZeroShotClassificationPipeline(ChunkPipeline):
    """
    NLI-based zero-shot classification pipeline using a `ModelForSequenceClassification` trained on NLI (natural
    language inference) tasks. Equivalent of `text-classification` pipelines, but these models don't require a
    hardcoded number of potential classes, they can be chosen at runtime. It usually means it's slower but it is
    **much** more flexible.

    Any combination of sequences and labels can be passed and each combination will be posed as a premise/hypothesis
    pair and passed to the pretrained model. Then, the logit for *entailment* is taken as the logit for the candidate
    label being valid. Any NLI model can be used, but the id of the *entailment* label must be included in the model
    config's :attr:*~transformers.PretrainedConfig.label2id*.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> oracle = pipeline(model="facebook/bart-large-mnli")
    >>> oracle(
    ...     "I have a problem with my iphone that needs to be resolved asap!!",
    ...     candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
    ... )
    {'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'], 'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}

    >>> oracle(
    ...     "I have a problem with my iphone that needs to be resolved asap!!",
    ...     candidate_labels=["english", "german"],
    ... )
    {'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['english', 'german'], 'scores': [0.814, 0.186]}
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This NLI pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"zero-shot-classification"`.

    The models that this pipeline can use are models that have been fine-tuned on an NLI task. See the up-to-date list
    of available models on [huggingface.co/models](https://huggingface.co/models?search=nli).
    """
    def __init__(self, args_parser=..., *args, **kwargs) -> None:
        ...

    @property
    def entailment_id(self): # -> int:
        ...

    def __call__(self, sequences: Union[str, List[str]], *args, **kwargs): # -> list[Any] | PipelineIterator | Generator[Any, Any, None] | Tensor | Any | None:
        """
        Classify the sequence(s) given as inputs. See the [`ZeroShotClassificationPipeline`] documentation for more
        information.

        Args:
            sequences (`str` or `List[str]`):
                The sequence(s) to classify, will be truncated if the model input is too large.
            candidate_labels (`str` or `List[str]`):
                The set of possible class labels to classify each sequence into. Can be a single label, a string of
                comma-separated labels, or a list of labels.
            hypothesis_template (`str`, *optional*, defaults to `"This example is {}."`):
                The template used to turn each label into an NLI-style hypothesis. This template must include a {} or
                similar syntax for the candidate label to be inserted into the template. For example, the default
                template is `"This example is {}."` With the candidate label `"sports"`, this would be fed into the
                model like `"<cls> sequence to classify <sep> This example is sports . <sep>"`. The default template
                works well in many cases, but it may be worthwhile to experiment with different templates depending on
                the task setting.
            multi_label (`bool`, *optional*, defaults to `False`):
                Whether or not multiple candidate labels can be true. If `False`, the scores are normalized such that
                the sum of the label likelihoods for each sequence is 1. If `True`, the labels are considered
                independent and probabilities are normalized for each candidate by doing a softmax of the entailment
                score vs. the contradiction score.

        Return:
            A `dict` or a list of `dict`: Each result comes as a dictionary with the following keys:

            - **sequence** (`str`) -- The sequence for which this is the output.
            - **labels** (`List[str]`) -- The labels sorted by order of likelihood.
            - **scores** (`List[float]`) -- The probabilities for each of the labels.
        """
        ...

    def preprocess(self, inputs, candidate_labels=..., hypothesis_template=...): # -> Generator[dict[str | Any, Any], Any, None]:
        ...

    def postprocess(self, model_outputs, multi_label=...): # -> dict[str, Any]:
        ...
