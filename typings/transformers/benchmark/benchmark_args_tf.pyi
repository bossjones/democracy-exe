"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass

import tensorflow as tf

from ..utils import is_tf_available
from .benchmark_args_utils import BenchmarkArguments

if is_tf_available():
    ...
logger = ...
@dataclass
class TensorFlowBenchmarkArguments(BenchmarkArguments):
    deprecated_args = ...
    def __init__(self, **kwargs) -> None:
        """
        This __init__ is there for legacy code. When removing deprecated args completely, the class can simply be
        deleted
        """
        ...

    tpu_name: str = ...
    device_idx: int = ...
    eager_mode: bool = ...
    use_xla: bool = ...
    @property
    def is_tpu(self) -> bool:
        ...

    @property
    def strategy(self) -> tf.distribute.Strategy:
        ...

    @property
    def gpu_list(self):
        ...

    @property
    def n_gpu(self) -> int:
        ...

    @property
    def is_gpu(self) -> bool:
        ...
