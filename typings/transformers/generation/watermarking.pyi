"""
This type stub file was generated by pyright.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Optional, Union
from ..configuration_utils import PretrainedConfig
from ..utils import is_torch_available
from .configuration_utils import WatermarkingConfig

if is_torch_available():
    ...
logger = ...
@dataclass
class WatermarkDetectorOutput:
    """
    Outputs of a watermark detector.

    Args:
        num_tokens_scored (np.array of shape (batch_size)):
            Array containing the number of tokens scored for each element in the batch.
        num_green_tokens (np.array of shape (batch_size)):
            Array containing the number of green tokens for each element in the batch.
        green_fraction (np.array of shape (batch_size)):
            Array containing the fraction of green tokens for each element in the batch.
        z_score (np.array of shape (batch_size)):
            Array containing the z-score for each element in the batch. Z-score here shows
            how many standard deviations away is the green token count in the input text
            from the expected green token count for machine-generated text.
        p_value (np.array of shape (batch_size)):
            Array containing the p-value for each batch obtained from z-scores.
        prediction (np.array of shape (batch_size)), *optional*:
            Array containing boolean predictions whether a text is machine-generated for each element in the batch.
        confidence (np.array of shape (batch_size)), *optional*:
            Array containing confidence scores of a text being machine-generated for each element in the batch.
    """
    num_tokens_scored: np.array = ...
    num_green_tokens: np.array = ...
    green_fraction: np.array = ...
    z_score: np.array = ...
    p_value: np.array = ...
    prediction: Optional[np.array] = ...
    confidence: Optional[np.array] = ...


class WatermarkDetector:
    r"""
    Detector for detection of watermark generated text. The detector needs to be given the exact same settings that were
    given during text generation to replicate the watermark greenlist generation and so detect the watermark. This includes
    the correct device that was used during text generation, the correct watermarking arguments and the correct tokenizer vocab size.
    The code was based on the [original repo](https://github.com/jwkirchenbauer/lm-watermarking/tree/main).

    See [the paper](https://arxiv.org/abs/2306.04634) for more information.

    Args:
        model_config (`PretrainedConfig`):
            The model config that will be used to get model specific arguments used when generating.
        device (`str`):
            The device which was used during watermarked text generation.
        watermarking_config (Union[`WatermarkingConfig`, `Dict`]):
            The exact same watermarking config and arguments used when generating text.
        ignore_repeated_ngrams (`bool`, *optional*, defaults to `False`):
            Whether to count every unique ngram only once or not.
        max_cache_size (`int`, *optional*, defaults to 128):
            The max size to be used for LRU caching of seeding/sampling algorithms called for every token.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, WatermarkDetector, WatermarkingConfig

    >>> model_id = "openai-community/gpt2"
    >>> model = AutoModelForCausalLM.from_pretrained(model_id)
    >>> tok = AutoTokenizer.from_pretrained(model_id)
    >>> tok.pad_token_id = tok.eos_token_id
    >>> tok.padding_side = "left"

    >>> inputs = tok(["This is the beginning of a long story", "Alice and Bob are"], padding=True, return_tensors="pt")
    >>> input_len = inputs["input_ids"].shape[-1]

    >>> # first generate text with watermark and without
    >>> watermarking_config = WatermarkingConfig(bias=2.5, seeding_scheme="selfhash")
    >>> out_watermarked = model.generate(**inputs, watermarking_config=watermarking_config, do_sample=False, max_length=20)
    >>> out = model.generate(**inputs, do_sample=False, max_length=20)

    >>> # now we can instantiate the detector and check the generated text
    >>> detector = WatermarkDetector(model_config=model.config, device="cpu", watermarking_config=watermarking_config)
    >>> detection_out_watermarked = detector(out_watermarked, return_dict=True)
    >>> detection_out = detector(out, return_dict=True)
    >>> detection_out_watermarked.prediction
    array([ True,  True])

    >>> detection_out.prediction
    array([False,  False])
    ```
    """
    def __init__(self, model_config: PretrainedConfig, device: str, watermarking_config: Union[WatermarkingConfig, Dict], ignore_repeated_ngrams: bool = ..., max_cache_size: int = ...) -> None:
        ...
    
    def __call__(self, input_ids: torch.LongTensor, z_threshold: float = ..., return_dict: bool = ...) -> Union[WatermarkDetectorOutput, np.array]:
        """
                Args:
                input_ids (`torch.LongTensor`):
                    The watermark generated text. It is advised to remove the prompt, which can affect the detection.
                z_threshold (`Dict`, *optional*, defaults to `3.0`):
                    Changing this threshold will change the sensitivity of the detector. Higher z threshold gives less
                    sensitivity and vice versa for lower z threshold.
                return_dict (`bool`,  *optional*, defaults to `False`):
                    Whether to return `~generation.WatermarkDetectorOutput` or not. If not it will return boolean predictions,
        ma
                Return:
                    [`~generation.WatermarkDetectorOutput`] or `np.array`: A [`~generation.WatermarkDetectorOutput`]
                    if `return_dict=True` otherwise a `np.array`.

        """
        ...
    


