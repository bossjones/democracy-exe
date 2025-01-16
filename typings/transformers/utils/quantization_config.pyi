"""
This type stub file was generated by pyright.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from ..utils import is_torch_available

if is_torch_available():
    ...
logger = ...
class QuantizationMethod(str, Enum):
    BITS_AND_BYTES = ...
    GPTQ = ...
    AWQ = ...
    AQLM = ...
    QUANTO = ...
    EETQ = ...
    HQQ = ...
    FBGEMM_FP8 = ...


class AWQLinearVersion(str, Enum):
    GEMM = ...
    GEMV = ...
    EXLLAMA = ...
    @staticmethod
    def from_str(version: str): # -> Literal[AWQLinearVersion.GEMM, AWQLinearVersion.GEMV, AWQLinearVersion.EXLLAMA]:
        ...
    


class AwqBackendPackingMethod(str, Enum):
    AUTOAWQ = ...
    LLMAWQ = ...


@dataclass
class QuantizationConfigMixin:
    """
    Mixin class for quantization config
    """
    quant_method: QuantizationMethod
    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=..., **kwargs): # -> tuple[Self, dict[str, Any]] | Self:
        """
        Instantiates a [`QuantizationConfigMixin`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            return_unused_kwargs (`bool`,*optional*, defaults to `False`):
                Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
                `PreTrainedModel`.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`QuantizationConfigMixin`]: The configuration object instantiated from those parameters.
        """
        ...
    
    def to_json_file(self, json_file_path: Union[str, os.PathLike]): # -> None:
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default
                `QuantizationConfig()` is serialized to JSON file.
        """
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        ...
    
    def __iter__(self): # -> Generator[tuple[str, Any], Any, None]:
        """allows `dict(obj)` for situations where obj may be a dict or QuantizationConfigMixin"""
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def to_json_string(self, use_diff: bool = ...) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        ...
    
    def update(self, **kwargs): # -> dict[str, Any]:
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
        returning all the unused kwargs.

        Args:
            kwargs (`Dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        ...
    


@dataclass
class HqqConfig(QuantizationConfigMixin):
    """
    This is wrapper around hqq's BaseQuantizeConfig.

    Args:
        nbits (`int`, *optional*, defaults to 4):
            Number of bits. Supported values are (8, 4, 3, 2, 1).
        group_size (`int`, *optional*, defaults to 64):
            Group-size value. Supported values are any value that is divisble by weight.shape[axis]).
        quant_zero (`bool`, *optional*, defaults to `True`):
            Quantize the zero-point if set to `True`.
        quant_scale (`bool`, *optional*, defaults to `False`):
            Quantize the scaling if set to `True`.
        offload_meta (`bool`, *optional*, defaults to `False`):
            Offload the meta-data to the CPU if set to `True`.
        view_as_float (`bool`, *optional*, defaults to `False`):
            View the quantized weight as float (used in distributed training) if set to `True`.
        axis (`int`, *optional*, defaults to 0):
            Axis along which grouping is performed. Supported values are 0 or 1.
        dynamic_config (dict, *optional*):
            Parameters for dynamic configuration. The key is the name tag of the layer and the value is a quantization config.
            If set, each layer specified by its id will use its dedicated quantization configuration.
        skip_modules (`List[str]`, *optional*, defaults to `['lm_head']`):
            List of `nn.Linear` layers to skip.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    """
    def __init__(self, nbits: int = ..., group_size: int = ..., quant_zero: bool = ..., quant_scale: bool = ..., offload_meta: bool = ..., view_as_float: bool = ..., axis: int = ..., dynamic_config: Optional[dict] = ..., skip_modules: List[str] = ..., **kwargs) -> None:
        ...
    
    def post_init(self): # -> None:
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.
        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        ...
    


@dataclass
class BitsAndBytesConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `bitsandbytes`.

    This replaces `load_in_8bit` or `load_in_4bit`therefore both options are mutually exclusive.

    Currently only supports `LLM.int8()`, `FP4`, and `NF4` quantization. If more methods are added to `bitsandbytes`,
    then more arguments will be added to this class.

    Args:
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 8-bit quantization with LLM.int8().
        load_in_4bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from
            `bitsandbytes`.
        llm_int8_threshold (`float`, *optional*, defaults to 6.0):
            This corresponds to the outlier threshold for outlier detection as described in `LLM.int8() : 8-bit Matrix
            Multiplication for Transformers at Scale` paper: https://arxiv.org/abs/2208.07339 Any hidden states value
            that is above this threshold will be considered an outlier and the operation on those values will be done
            in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but
            there are some exceptional systematic outliers that are very differently distributed for large models.
            These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of
            magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6,
            but a lower threshold might be needed for more unstable models (small models, fine-tuning).
        llm_int8_skip_modules (`List[str]`, *optional*):
            An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as
            Jukebox that has several heads in different places and not necessarily at the last position. For example
            for `CausalLM` models, the last `lm_head` is kept in its original `dtype`.
        llm_int8_enable_fp32_cpu_offload (`bool`, *optional*, defaults to `False`):
            This flag is used for advanced use cases and users that are aware of this feature. If you want to split
            your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use
            this flag. This is useful for offloading large models such as `google/flan-t5-xxl`. Note that the int8
            operations will not be run on CPU.
        llm_int8_has_fp16_weight (`bool`, *optional*, defaults to `False`):
            This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not
            have to be converted back and forth for the backward pass.
        bnb_4bit_compute_dtype (`torch.dtype` or str, *optional*, defaults to `torch.float32`):
            This sets the computational type which might be different than the input type. For example, inputs might be
            fp32, but computation can be set to bf16 for speedups.
        bnb_4bit_quant_type (`str`,  *optional*, defaults to `"fp4"`):
            This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types
            which are specified by `fp4` or `nf4`.
        bnb_4bit_use_double_quant (`bool`, *optional*, defaults to `False`):
            This flag is used for nested quantization where the quantization constants from the first quantization are
            quantized again.
        bnb_4bit_quant_storage (`torch.dtype` or str, *optional*, defaults to `torch.uint8`):
            This sets the storage type to pack the quanitzed 4-bit prarams.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    """
    def __init__(self, load_in_8bit=..., load_in_4bit=..., llm_int8_threshold=..., llm_int8_skip_modules=..., llm_int8_enable_fp32_cpu_offload=..., llm_int8_has_fp16_weight=..., bnb_4bit_compute_dtype=..., bnb_4bit_quant_type=..., bnb_4bit_use_double_quant=..., bnb_4bit_quant_storage=..., **kwargs) -> None:
        ...
    
    @property
    def load_in_4bit(self): # -> bool:
        ...
    
    @load_in_4bit.setter
    def load_in_4bit(self, value: bool): # -> None:
        ...
    
    @property
    def load_in_8bit(self): # -> bool:
        ...
    
    @load_in_8bit.setter
    def load_in_8bit(self, value: bool): # -> None:
        ...
    
    def post_init(self): # -> None:
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        ...
    
    def is_quantizable(self): # -> bool:
        r"""
        Returns `True` if the model is quantizable, `False` otherwise.
        """
        ...
    
    def quantization_method(self): # -> Literal['llm_int8', 'fp4', 'nf4'] | None:
        r"""
        This method returns the quantization method used for the model. If the model is not quantizable, it returns
        `None`.
        """
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        ...
    


class ExllamaVersion(int, Enum):
    ONE = ...
    TWO = ...


@dataclass
class GPTQConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `optimum` api for gptq quantization relying on auto_gptq backend.

    Args:
        bits (`int`):
            The number of bits to quantize to, supported numbers are (2, 3, 4, 8).
        tokenizer (`str` or `PreTrainedTokenizerBase`, *optional*):
            The tokenizer used to process the dataset. You can pass either:
                - A custom tokenizer object.
                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                    using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
        dataset (`Union[List[str]]`, *optional*):
            The dataset used for quantization. You can provide your own dataset in a list of string or just use the
            original datasets used in GPTQ paper ['wikitext2','c4','c4-new']
        group_size (`int`, *optional*, defaults to 128):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        damp_percent (`float`, *optional*, defaults to 0.1):
            The percent of the average Hessian diagonal to use for dampening. Recommended value is 0.1.
        desc_act (`bool`, *optional*, defaults to `False`):
            Whether to quantize columns in order of decreasing activation size. Setting it to False can significantly
            speed up inference but the perplexity may become slightly worse. Also known as act-order.
        sym (`bool`, *optional*, defaults to `True`):
            Whether to use symetric quantization.
        true_sequential (`bool`, *optional*, defaults to `True`):
            Whether to perform sequential quantization even within a single Transformer block. Instead of quantizing
            the entire block at once, we perform layer-wise quantization. As a result, each layer undergoes
            quantization using inputs that have passed through the previously quantized layers.
        use_cuda_fp16 (`bool`, *optional*, defaults to `False`):
            Whether or not to use optimized cuda kernel for fp16 model. Need to have model in fp16.
        model_seqlen (`int`, *optional*):
            The maximum sequence length that the model can take.
        block_name_to_quantize (`str`, *optional*):
            The transformers block name to quantize. If None, we will infer the block name using common patterns (e.g. model.layers)
        module_name_preceding_first_block (`List[str]`, *optional*):
            The layers that are preceding the first Transformer block.
        batch_size (`int`, *optional*, defaults to 1):
            The batch size used when processing the dataset
        pad_token_id (`int`, *optional*):
            The pad token id. Needed to prepare the dataset when `batch_size` > 1.
        use_exllama (`bool`, *optional*):
            Whether to use exllama backend. Defaults to `True` if unset. Only works with `bits` = 4.
        max_input_length (`int`, *optional*):
            The maximum input length. This is needed to initialize a buffer that depends on the maximum expected input
            length. It is specific to the exllama backend with act-order.
        exllama_config (`Dict[str, Any]`, *optional*):
            The exllama config. You can specify the version of the exllama kernel through the `version` key. Defaults
            to `{"version": 1}` if unset.
        cache_block_outputs (`bool`, *optional*, defaults to `True`):
            Whether to cache block outputs to reuse as inputs for the succeeding block.
        modules_in_block_to_quantize (`List[List[str]]`, *optional*):
            List of list of module names to quantize in the specified block. This argument is useful to exclude certain linear modules from being quantized.
            The block to quantize can be specified by setting `block_name_to_quantize`. We will quantize each list sequentially. If not set, we will quantize all linear layers.
            Example: `modules_in_block_to_quantize =[["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"], ["self_attn.o_proj"]]`.
            In this example, we will first quantize the q,k,v layers simultaneously since they are independent.
            Then, we will quantize `self_attn.o_proj` layer with the q,k,v layers quantized. This way, we will get
            better results since it reflects the real input `self_attn.o_proj` will get when the model is quantized.
    """
    def __init__(self, bits: int, tokenizer: Any = ..., dataset: Optional[Union[List[str], str]] = ..., group_size: int = ..., damp_percent: float = ..., desc_act: bool = ..., sym: bool = ..., true_sequential: bool = ..., use_cuda_fp16: bool = ..., model_seqlen: Optional[int] = ..., block_name_to_quantize: Optional[str] = ..., module_name_preceding_first_block: Optional[List[str]] = ..., batch_size: int = ..., pad_token_id: Optional[int] = ..., use_exllama: Optional[bool] = ..., max_input_length: Optional[int] = ..., exllama_config: Optional[Dict[str, Any]] = ..., cache_block_outputs: bool = ..., modules_in_block_to_quantize: Optional[List[List[str]]] = ..., **kwargs) -> None:
        ...
    
    def get_loading_attributes(self): # -> dict[str, Any]:
        ...
    
    def post_init(self): # -> None:
        r"""
        Safety checker that arguments are correct
        """
        ...
    
    def to_dict(self): # -> Dict[str, Any]:
        ...
    
    def to_dict_optimum(self): # -> Dict[str, Any]:
        """
        Get compatible dict for optimum gptq config
        """
        ...
    
    @classmethod
    def from_dict_optimum(cls, config_dict): # -> Self:
        """
        Get compatible class with optimum gptq config dict
        """
        ...
    


@dataclass
class AwqConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `auto-awq` library awq quantization relying on auto_awq backend.

    Args:
        bits (`int`, *optional*, defaults to 4):
            The number of bits to quantize to.
        group_size (`int`, *optional*, defaults to 128):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        zero_point (`bool`, *optional*, defaults to `True`):
            Whether to use zero point quantization.
        version (`AWQLinearVersion`, *optional*, defaults to `AWQLinearVersion.GEMM`):
            The version of the quantization algorithm to use. GEMM is better for big batch_size (e.g. >= 8) otherwise,
            GEMV is better (e.g. < 8 ). GEMM models are compatible with Exllama kernels.
        backend (`AwqBackendPackingMethod`, *optional*, defaults to `AwqBackendPackingMethod.AUTOAWQ`):
            The quantization backend. Some models might be quantized using `llm-awq` backend. This is useful for users
            that quantize their own models using `llm-awq` library.
        do_fuse (`bool`, *optional*, defaults to `False`):
            Whether to fuse attention and mlp layers together for faster inference
        fuse_max_seq_len (`int`, *optional*):
            The Maximum sequence length to generate when using fusing.
        modules_to_fuse (`dict`, *optional*, default to `None`):
            Overwrite the natively supported fusing scheme with the one specified by the users.
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
            Note you cannot quantize directly with transformers, please refer to `AutoAWQ` documentation for quantizing HF models.
        exllama_config (`Dict[str, Any]`, *optional*):
            You can specify the version of the exllama kernel through the `version` key, the maximum sequence
            length through the `max_input_len` key, and the maximum batch size through the `max_batch_size` key.
            Defaults to `{"version": 2, "max_input_len": 2048, "max_batch_size": 8}` if unset.
    """
    def __init__(self, bits: int = ..., group_size: int = ..., zero_point: bool = ..., version: AWQLinearVersion = ..., backend: AwqBackendPackingMethod = ..., do_fuse: Optional[bool] = ..., fuse_max_seq_len: Optional[int] = ..., modules_to_fuse: Optional[dict] = ..., modules_to_not_convert: Optional[List] = ..., exllama_config: Optional[Dict[str, int]] = ..., **kwargs) -> None:
        ...
    
    def post_init(self): # -> None:
        r"""
        Safety checker that arguments are correct
        """
        ...
    
    def get_loading_attributes(self): # -> dict[str, Any]:
        ...
    


@dataclass
class AqlmConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about `aqlm` parameters.

    Args:
        in_group_size (`int`, *optional*, defaults to 8):
            The group size along the input dimension.
        out_group_size (`int`, *optional*, defaults to 1):
            The group size along the output dimension. It's recommended to always use 1.
        num_codebooks (`int`, *optional*, defaults to 1):
            Number of codebooks for the Additive Quantization procedure.
        nbits_per_codebook (`int`, *optional*, defaults to 16):
            Number of bits encoding a single codebook vector. Codebooks size is 2**nbits_per_codebook.
        linear_weights_not_to_quantize (`Optional[List[str]]`, *optional*):
            List of full paths of `nn.Linear` weight parameters that shall not be quantized.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    """
    def __init__(self, in_group_size: int = ..., out_group_size: int = ..., num_codebooks: int = ..., nbits_per_codebook: int = ..., linear_weights_not_to_quantize: Optional[List[str]] = ..., **kwargs) -> None:
        ...
    
    def post_init(self): # -> None:
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        ...
    


@dataclass
class QuantoConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `quanto`.

    Args:
        weights (`str`, *optional*, defaults to `"int8"`):
            The target dtype for the weights after quantization. Supported values are ("float8","int8","int4","int2")
        activations (`str`, *optional*):
            The target dtype for the activations after quantization. Supported values are (None,"int8","float8")
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
    """
    def __init__(self, weights=..., activations=..., modules_to_not_convert: Optional[List] = ..., **kwargs) -> None:
        ...
    
    def post_init(self): # -> None:
        r"""
        Safety checker that arguments are correct
        """
        ...
    


@dataclass
class EetqConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `eetq`.

    Args:
        weights (`str`, *optional*, defaults to `"int8"`):
            The target dtype for the weights. Supported value is only "int8"
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision.
    """
    def __init__(self, weights: str = ..., modules_to_not_convert: Optional[List] = ..., **kwargs) -> None:
        ...
    
    def post_init(self): # -> None:
        r"""
        Safety checker that arguments are correct
        """
        ...
    


@dataclass
class FbgemmFp8Config(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using fbgemm fp8 quantization.

    Args:
        activation_scale_ub (`float`, *optional*, defaults to 1200.0):
            The activation scale upper bound. This is used when quantizing the input activation.
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision.
    """
    def __init__(self, activation_scale_ub: float = ..., modules_to_not_convert: Optional[List] = ..., **kwargs) -> None:
        ...
    
    def get_loading_attributes(self): # -> dict[str, Any]:
        ...
    


