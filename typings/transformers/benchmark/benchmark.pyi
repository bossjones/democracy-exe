"""
This type stub file was generated by pyright.
"""

from ..configuration_utils import PretrainedConfig
from ..utils import is_py3nvml_available, is_torch_available
from .benchmark_utils import Benchmark
from .benchmark_args import PyTorchBenchmarkArguments

"""
Benchmarking the library on inference and training in PyTorch.
"""
if is_torch_available():
    ...
if is_py3nvml_available():
    ...
logger = ...
class PyTorchBenchmark(Benchmark):
    args: PyTorchBenchmarkArguments
    configs: PretrainedConfig
    framework: str = ...
    @property
    def framework_version(self): # -> TorchVersion:
        ...
    


