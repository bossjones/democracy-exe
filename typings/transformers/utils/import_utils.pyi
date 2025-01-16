"""
This type stub file was generated by pyright.
"""

from functools import lru_cache
from types import ModuleType
from typing import Any

"""
Import utilities: Utilities related to imports and our lazy inits.
"""
logger = ...
ENV_VARS_TRUE_VALUES = ...
ENV_VARS_TRUE_AND_AUTO_VALUES = ...
USE_TF = ...
USE_TORCH = ...
USE_JAX = ...
USE_TORCH_XLA = ...
FORCE_TF_AVAILABLE = ...
TORCH_FX_REQUIRED_VERSION = ...
ACCELERATE_MIN_VERSION = ...
FSDP_MIN_VERSION = ...
XLA_FSDPV2_MIN_VERSION = ...
_apex_available = ...
_aqlm_available = ...
_av_available = ...
_bitsandbytes_available = ...
_eetq_available = ...
_fbgemm_gpu_available = ...
_galore_torch_available = ...
_lomo_available = ...
_bs4_available = ...
_coloredlogs_available = ...
_cv2_available = ...
_datasets_available = ...
_decord_available = ...
_detectron2_available = ...
_faiss_available = ...
_faiss_version = ...
_ftfy_available = ...
_g2p_en_available = ...
_jieba_available = ...
_jinja_available = ...
_kenlm_available = ...
_keras_nlp_available = ...
_levenshtein_available = ...
_librosa_available = ...
_natten_available = ...
_nltk_available = ...
_onnx_available = ...
_openai_available = ...
_optimum_available = ...
_auto_gptq_available = ...
_auto_awq_available = ...
_quanto_available = ...
_pandas_available = ...
_peft_available = ...
_phonemizer_available = ...
_psutil_available = ...
_py3nvml_available = ...
_pyctcdecode_available = ...
_pygments_available = ...
_pytesseract_available = ...
_pytest_available = ...
_pytorch_quantization_available = ...
_rjieba_available = ...
_sacremoses_available = ...
_safetensors_available = ...
_scipy_available = ...
_sentencepiece_available = ...
_is_seqio_available = ...
_is_gguf_available = ...
_sklearn_available = ...
if _sklearn_available:
    ...
_smdistributed_available = ...
_soundfile_available = ...
_spacy_available = ...
_tensorflow_probability_available = ...
_tensorflow_text_available = ...
_tf2onnx_available = ...
_timm_available = ...
_tokenizers_available = ...
_torchaudio_available = ...
_torchdistx_available = ...
_torchvision_available = ...
_mlx_available = ...
_hqq_available = ...
_torch_version = ...
_torch_available = ...
if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    ...
else:
    _torch_available = ...
_tf_version = ...
_tf_available = ...
if FORCE_TF_AVAILABLE in ENV_VARS_TRUE_VALUES:
    _tf_available = ...
else:
    ...
_essentia_available = ...
_essentia_version = ...
_pretty_midi_available = ...
_pretty_midi_version = ...
ccl_version = ...
_is_ccl_available = ...
ccl_version = ...
_flax_available = ...
if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
    ...
_torch_fx_available = ...
if _torch_available:
    torch_version = ...
    _torch_fx_available = ...
_torch_xla_available = ...
if USE_TORCH_XLA in ENV_VARS_TRUE_VALUES:
    ...
def is_kenlm_available(): # -> Tuple[bool, str] | bool:
    ...

def is_cv2_available(): # -> bool:
    ...

def is_torch_available(): # -> bool:
    ...

def is_torch_deterministic(): # -> bool:
    """
    Check whether pytorch uses deterministic algorithms by looking if torch.set_deterministic_debug_mode() is set to 1 or 2"
    """
    ...

def is_hqq_available(): # -> Tuple[bool, str] | bool:
    ...

def is_pygments_available(): # -> Tuple[bool, str] | bool:
    ...

def get_torch_version(): # -> str:
    ...

def is_torch_sdpa_available(): # -> bool:
    ...

def is_torchvision_available(): # -> Tuple[bool, str] | bool:
    ...

def is_galore_torch_available(): # -> Tuple[bool, str] | bool:
    ...

def is_lomo_available(): # -> Tuple[bool, str] | bool:
    ...

def is_pyctcdecode_available(): # -> Tuple[bool, str] | bool:
    ...

def is_librosa_available(): # -> Tuple[bool, str] | bool:
    ...

def is_essentia_available(): # -> bool:
    ...

def is_pretty_midi_available(): # -> bool:
    ...

def is_torch_cuda_available(): # -> bool:
    ...

def is_mamba_ssm_available(): # -> Tuple[bool, str] | bool:
    ...

def is_mamba_2_ssm_available(): # -> bool:
    ...

def is_causal_conv1d_available(): # -> Tuple[bool, str] | bool:
    ...

def is_mambapy_available(): # -> Tuple[bool, str] | bool:
    ...

def is_torch_mps_available(): # -> bool:
    ...

def is_torch_bf16_gpu_available(): # -> bool:
    ...

def is_torch_bf16_cpu_available(): # -> bool:
    ...

def is_torch_bf16_available(): # -> bool:
    ...

@lru_cache()
def is_torch_fp16_available_on_device(device): # -> bool:
    ...

@lru_cache()
def is_torch_bf16_available_on_device(device): # -> bool:
    ...

def is_torch_tf32_available(): # -> bool:
    ...

def is_torch_fx_available(): # -> bool:
    ...

def is_peft_available(): # -> Tuple[bool, str] | bool:
    ...

def is_bs4_available(): # -> bool:
    ...

def is_tf_available(): # -> bool:
    ...

def is_coloredlogs_available(): # -> Tuple[bool, str] | bool:
    ...

def is_tf2onnx_available(): # -> Tuple[bool, str] | bool:
    ...

def is_onnx_available(): # -> Tuple[bool, str] | bool:
    ...

def is_openai_available(): # -> Tuple[bool, str] | bool:
    ...

def is_flax_available(): # -> bool:
    ...

def is_ftfy_available(): # -> Tuple[bool, str] | bool:
    ...

def is_g2p_en_available(): # -> Tuple[bool, str] | bool:
    ...

@lru_cache()
def is_torch_tpu_available(check_device=...): # -> bool:
    "Checks if `torch_xla` is installed and potentially if a TPU is in the environment"
    ...

@lru_cache
def is_torch_xla_available(check_is_tpu=..., check_is_gpu=...): # -> bool:
    """
    Check if `torch_xla` is available. To train a native pytorch job in an environment with torch xla installed, set
    the USE_TORCH_XLA to false.
    """
    ...

@lru_cache()
def is_torch_neuroncore_available(check_device=...): # -> bool:
    ...

@lru_cache()
def is_torch_npu_available(check_device=...): # -> Literal[False]:
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    ...

@lru_cache()
def is_torch_mlu_available(check_device=...): # -> Literal[False]:
    "Checks if `torch_mlu` is installed and potentially if a MLU is in the environment"
    ...

def is_torchdynamo_available(): # -> bool:
    ...

def is_torch_compile_available(): # -> bool:
    ...

def is_torchdynamo_compiling(): # -> bool:
    ...

def is_torch_tensorrt_fx_available(): # -> bool:
    ...

def is_datasets_available(): # -> Tuple[bool, str] | bool:
    ...

def is_detectron2_available(): # -> Tuple[bool, str] | bool:
    ...

def is_rjieba_available(): # -> Tuple[bool, str] | bool:
    ...

def is_psutil_available(): # -> Tuple[bool, str] | bool:
    ...

def is_py3nvml_available(): # -> Tuple[bool, str] | bool:
    ...

def is_sacremoses_available(): # -> Tuple[bool, str] | bool:
    ...

def is_apex_available(): # -> Tuple[bool, str] | bool:
    ...

def is_aqlm_available(): # -> Tuple[bool, str] | bool:
    ...

def is_av_available(): # -> bool:
    ...

def is_ninja_available(): # -> bool:
    r"""
    Code comes from *torch.utils.cpp_extension.is_ninja_available()*. Returns `True` if the
    [ninja](https://ninja-build.org/) build system is available on the system, `False` otherwise.
    """
    ...

def is_ipex_available(): # -> bool:
    ...

@lru_cache
def is_torch_xpu_available(check_device=...): # -> bool:
    """
    Checks if XPU acceleration is available either via `intel_extension_for_pytorch` or
    via stock PyTorch (>=2.4) and potentially if a XPU is in the environment
    """
    ...

def is_bitsandbytes_available(): # -> bool:
    ...

def is_flash_attn_2_available(): # -> bool:
    ...

def is_flash_attn_greater_or_equal_2_10(): # -> bool:
    ...

@lru_cache()
def is_flash_attn_greater_or_equal(library_version: str): # -> bool:
    ...

def is_torchdistx_available(): # -> Tuple[bool, str] | bool:
    ...

def is_faiss_available(): # -> bool:
    ...

def is_scipy_available(): # -> Tuple[bool, str] | bool:
    ...

def is_sklearn_available(): # -> bool:
    ...

def is_sentencepiece_available(): # -> Tuple[bool, str] | bool:
    ...

def is_seqio_available(): # -> Tuple[bool, str] | bool:
    ...

def is_gguf_available(): # -> Tuple[bool, str] | bool:
    ...

def is_protobuf_available(): # -> bool:
    ...

def is_accelerate_available(min_version: str = ...): # -> bool:
    ...

def is_fsdp_available(min_version: str = ...): # -> bool:
    ...

def is_optimum_available(): # -> Tuple[bool, str] | bool:
    ...

def is_auto_awq_available(): # -> bool:
    ...

def is_quanto_available(): # -> Tuple[bool, str] | bool:
    ...

def is_auto_gptq_available(): # -> Tuple[bool, str] | bool:
    ...

def is_eetq_available(): # -> Tuple[bool, str] | bool:
    ...

def is_fbgemm_gpu_available(): # -> Tuple[bool, str] | bool:
    ...

def is_levenshtein_available(): # -> Tuple[bool, str] | bool:
    ...

def is_optimum_neuron_available(): # -> Tuple[bool, str] | bool:
    ...

def is_safetensors_available(): # -> Tuple[bool, str] | bool:
    ...

def is_tokenizers_available(): # -> Tuple[bool, str] | bool:
    ...

@lru_cache
def is_vision_available(): # -> bool:
    ...

def is_pytesseract_available(): # -> Tuple[bool, str] | bool:
    ...

def is_pytest_available(): # -> Tuple[bool, str] | bool:
    ...

def is_spacy_available(): # -> Tuple[bool, str] | bool:
    ...

def is_tensorflow_text_available(): # -> Tuple[bool, str] | bool:
    ...

def is_keras_nlp_available(): # -> Tuple[bool, str] | bool:
    ...

def is_in_notebook(): # -> bool:
    ...

def is_pytorch_quantization_available(): # -> Tuple[bool, str] | bool:
    ...

def is_tensorflow_probability_available(): # -> Tuple[bool, str] | bool:
    ...

def is_pandas_available(): # -> Tuple[bool, str] | bool:
    ...

def is_sagemaker_dp_enabled(): # -> bool:
    ...

def is_sagemaker_mp_enabled(): # -> bool:
    ...

def is_training_run_on_sagemaker(): # -> bool:
    ...

def is_soundfile_availble(): # -> Tuple[bool, str] | bool:
    ...

def is_timm_available(): # -> Tuple[bool, str] | bool:
    ...

def is_natten_available(): # -> Tuple[bool, str] | bool:
    ...

def is_nltk_available(): # -> Tuple[bool, str] | bool:
    ...

def is_torchaudio_available(): # -> Tuple[bool, str] | bool:
    ...

def is_speech_available(): # -> Tuple[bool, str] | bool:
    ...

def is_phonemizer_available(): # -> Tuple[bool, str] | bool:
    ...

def torch_only_method(fn): # -> Callable[..., Any]:
    ...

def is_ccl_available(): # -> bool:
    ...

def is_decord_available(): # -> bool:
    ...

def is_sudachi_available(): # -> bool:
    ...

def get_sudachi_version(): # -> str:
    ...

def is_sudachi_projection_available(): # -> bool:
    ...

def is_jumanpp_available(): # -> bool:
    ...

def is_cython_available(): # -> bool:
    ...

def is_jieba_available(): # -> Tuple[bool, str] | bool:
    ...

def is_jinja_available(): # -> Tuple[bool, str] | bool:
    ...

def is_mlx_available(): # -> Tuple[bool, str] | bool:
    ...

AV_IMPORT_ERROR = ...
CV2_IMPORT_ERROR = ...
DATASETS_IMPORT_ERROR = ...
TOKENIZERS_IMPORT_ERROR = ...
SENTENCEPIECE_IMPORT_ERROR = ...
PROTOBUF_IMPORT_ERROR = ...
FAISS_IMPORT_ERROR = ...
PYTORCH_IMPORT_ERROR = ...
TORCHVISION_IMPORT_ERROR = ...
PYTORCH_IMPORT_ERROR_WITH_TF = ...
TF_IMPORT_ERROR_WITH_PYTORCH = ...
BS4_IMPORT_ERROR = ...
SKLEARN_IMPORT_ERROR = ...
TENSORFLOW_IMPORT_ERROR = ...
DETECTRON2_IMPORT_ERROR = ...
FLAX_IMPORT_ERROR = ...
FTFY_IMPORT_ERROR = ...
LEVENSHTEIN_IMPORT_ERROR = ...
G2P_EN_IMPORT_ERROR = ...
PYTORCH_QUANTIZATION_IMPORT_ERROR = ...
TENSORFLOW_PROBABILITY_IMPORT_ERROR = ...
TENSORFLOW_TEXT_IMPORT_ERROR = ...
PANDAS_IMPORT_ERROR = ...
PHONEMIZER_IMPORT_ERROR = ...
SACREMOSES_IMPORT_ERROR = ...
SCIPY_IMPORT_ERROR = ...
SPEECH_IMPORT_ERROR = ...
TIMM_IMPORT_ERROR = ...
NATTEN_IMPORT_ERROR = ...
NUMEXPR_IMPORT_ERROR = ...
NLTK_IMPORT_ERROR = ...
VISION_IMPORT_ERROR = ...
PYTESSERACT_IMPORT_ERROR = ...
PYCTCDECODE_IMPORT_ERROR = ...
ACCELERATE_IMPORT_ERROR = ...
CCL_IMPORT_ERROR = ...
ESSENTIA_IMPORT_ERROR = ...
LIBROSA_IMPORT_ERROR = ...
PRETTY_MIDI_IMPORT_ERROR = ...
DECORD_IMPORT_ERROR = ...
CYTHON_IMPORT_ERROR = ...
JIEBA_IMPORT_ERROR = ...
PEFT_IMPORT_ERROR = ...
JINJA_IMPORT_ERROR = ...
BACKENDS_MAPPING = ...
def requires_backends(obj, backends): # -> None:
    ...

class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """
    def __getattribute__(cls, key): # -> Any | None:
        ...
    


def is_torch_fx_proxy(x): # -> bool:
    ...

class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """
    def __init__(self, name, module_file, import_structure, module_spec=..., extra_objects=...) -> None:
        ...
    
    def __dir__(self): # -> Iterable[str]:
        ...
    
    def __getattr__(self, name: str) -> Any:
        ...
    
    def __reduce__(self): # -> tuple[type[Self], tuple[Any, str | None, Any]]:
        ...
    


class OptionalDependencyNotAvailable(BaseException):
    """Internally used error class for signalling an optional dependency was not found."""
    ...


def direct_transformers_import(path: str, file=...) -> ModuleType:
    """Imports transformers directly

    Args:
        path (`str`): The path to the source file
        file (`str`, *optional*): The file to join with the path. Defaults to "__init__.py".

    Returns:
        `ModuleType`: The resulting imported module
    """
    ...

