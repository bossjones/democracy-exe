"""
This type stub file was generated by pyright.
"""

import numpy as np
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from .utils import ExplicitEnum, is_torch_available

"""
PyTorch-independent utilities for the Trainer class.
"""
if is_torch_available():
    ...
def seed_worker(_): # -> None:
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    ...

def enable_full_determinism(seed: int, warn_only: bool = ...): # -> None:
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    - https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism for tensorflow
    """
    ...

def set_seed(seed: int, deterministic: bool = ...): # -> None:
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    ...

def neftune_post_forward_hook(module, input, output):
    """
    Implements the NEFTune forward pass for the model using forward hooks. Note this works only for torch.nn.Embedding
    layers. This method is slightly adapted from the original source code that can be found here:
    https://github.com/neelsjain/NEFTune Simply add it to your model as follows:
    ```python
    model = ...
    model.embed_tokens.neftune_noise_alpha = 0.1
    model.embed_tokens.register_forward_hook(neftune_post_forward_hook)
    ```
    Args:
        module (`torch.nn.Module`):
            The embedding module where the hook is attached. Note that you need to set `module.neftune_noise_alpha` to
            the desired noise alpha value.
        input (`torch.Tensor`):
            The input tensor to the model.
        output (`torch.Tensor`):
            The output tensor of the model (i.e. the embeddings).
    """
    ...

class EvalPrediction:
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*): Input data passed to the model.
        losses (`np.ndarray`, *optional*): Loss values computed during evaluation.
    """
    def __init__(self, predictions: Union[np.ndarray, Tuple[np.ndarray]], label_ids: Union[np.ndarray, Tuple[np.ndarray]], inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = ..., losses: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = ...) -> None:
        ...
    
    def __iter__(self): # -> Iterator[ndarray[Any, Any] | Tuple[ndarray[Any, Any]]]:
        ...
    
    def __getitem__(self, idx):
        ...
    


class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]
    ...


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    ...


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float
    metrics: Dict[str, float]
    ...


PREFIX_CHECKPOINT_DIR = ...
_re_checkpoint = ...
def get_last_checkpoint(folder): # -> None:
    ...

class IntervalStrategy(ExplicitEnum):
    NO = ...
    STEPS = ...
    EPOCH = ...


class EvaluationStrategy(ExplicitEnum):
    NO = ...
    STEPS = ...
    EPOCH = ...


class HubStrategy(ExplicitEnum):
    END = ...
    EVERY_SAVE = ...
    CHECKPOINT = ...
    ALL_CHECKPOINTS = ...


class BestRun(NamedTuple):
    """
    The best run found by a hyperparameter search (see [`~Trainer.hyperparameter_search`]).

    Parameters:
        run_id (`str`):
            The id of the best run (if models were saved, the corresponding checkpoint will be in the folder ending
            with run-{run_id}).
        objective (`float`):
            The objective that was obtained for this run.
        hyperparameters (`Dict[str, Any]`):
            The hyperparameters picked to get this run.
        run_summary (`Optional[Any]`):
            A summary of tuning experiments. `ray.tune.ExperimentAnalysis` object for Ray backend.
    """
    run_id: str
    objective: Union[float, List[float]]
    hyperparameters: Dict[str, Any]
    run_summary: Optional[Any] = ...


def default_compute_objective(metrics: Dict[str, float]) -> float:
    """
    The default objective to maximize/minimize when doing an hyperparameter search. It is the evaluation loss if no
    metrics are provided to the [`Trainer`], the sum of all metrics otherwise.

    Args:
        metrics (`Dict[str, float]`): The metrics returned by the evaluate method.

    Return:
        `float`: The objective to minimize or maximize
    """
    ...

def default_hp_space_optuna(trial) -> Dict[str, float]:
    ...

def default_hp_space_ray(trial) -> Dict[str, float]:
    ...

def default_hp_space_sigopt(trial): # -> list[dict[str, Any]]:
    ...

def default_hp_space_wandb(trial) -> Dict[str, float]:
    ...

class HPSearchBackend(ExplicitEnum):
    OPTUNA = ...
    RAY = ...
    SIGOPT = ...
    WANDB = ...


def is_main_process(local_rank): # -> bool:
    """
    Whether or not the current process is the local process, based on `xm.get_ordinal()` (for TPUs) first, then on
    `local_rank`.
    """
    ...

def total_processes_number(local_rank): # -> int:
    """
    Return the number of processes launched in parallel. Works with `torch.distributed` and TPUs.
    """
    ...

def speed_metrics(split, start_time, num_samples=..., num_steps=..., num_tokens=...): # -> dict[str, Any]:
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:
    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    - num_steps: number of steps processed
    - num_tokens: number of tokens processed
    """
    ...

class SchedulerType(ExplicitEnum):
    """
    Scheduler names for the parameter `lr_scheduler_type` in [`TrainingArguments`].
    By default, it uses "linear". Internally, this retrieves `get_linear_schedule_with_warmup` scheduler from [`Trainer`].
    Scheduler types:
       - "linear" = get_linear_schedule_with_warmup
       - "cosine" = get_cosine_schedule_with_warmup
       - "cosine_with_restarts" = get_cosine_with_hard_restarts_schedule_with_warmup
       - "polynomial" = get_polynomial_decay_schedule_with_warmup
       - "constant" =  get_constant_schedule
       - "constant_with_warmup" = get_constant_schedule_with_warmup
       - "inverse_sqrt" = get_inverse_sqrt_schedule
       - "reduce_lr_on_plateau" = get_reduce_on_plateau_schedule
       - "cosine_with_min_lr" = get_cosine_with_min_lr_schedule_with_warmup
       - "warmup_stable_decay" = get_wsd_schedule
    """
    LINEAR = ...
    COSINE = ...
    COSINE_WITH_RESTARTS = ...
    POLYNOMIAL = ...
    CONSTANT = ...
    CONSTANT_WITH_WARMUP = ...
    INVERSE_SQRT = ...
    REDUCE_ON_PLATEAU = ...
    COSINE_WITH_MIN_LR = ...
    WARMUP_STABLE_DECAY = ...


class TrainerMemoryTracker:
    """
    A helper class that tracks cpu and gpu memory.

    This class will silently skip unless `psutil` is available. Install with `pip install psutil`.

    When a stage completes, it can pass metrics dict to update with the memory metrics gathered during this stage.

    Example :

    ```python
    self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
    self._memory_tracker.start()
    # code ...
    metrics = {"train_runtime": 10.5}
    self._memory_tracker.stop_and_update_metrics(metrics)
    ```

    At the moment GPU tracking is only for `pytorch`, but can be extended to support `tensorflow`.

    To understand this class' intricacies please read the documentation of [`~Trainer.log_metrics`].
    """
    stages = ...
    def __init__(self, skip_memory_metrics=...) -> None:
        ...
    
    def derive_stage(self): # -> str:
        """derives the stage/caller name automatically"""
        ...
    
    def cpu_mem_used(self): # -> Any:
        """get resident set size memory for the current process"""
        ...
    
    def peak_monitor_func(self): # -> None:
        ...
    
    def start(self): # -> None:
        """start tracking for the caller's stage"""
        ...
    
    def stop(self, stage): # -> None:
        """stop tracking for the passed stage"""
        ...
    
    def update_metrics(self, stage, metrics): # -> None:
        """updates the metrics"""
        ...
    
    def stop_and_update_metrics(self, metrics=...): # -> None:
        """combine stop and metrics update in one call for simpler code"""
        ...
    


def has_length(dataset): # -> bool:
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    ...

def denumpify_detensorize(metrics): # -> Number | list[Any] | tuple[Any, ...] | dict[Any, Any] | Any | Tensor:
    """
    Recursively calls `.item()` on the element of the dictionary passed
    """
    ...

def number_of_arguments(func): # -> int:
    """
    Return the number of arguments of the passed function, even if it's a partial function.
    """
    ...

def find_executable_batch_size(function: callable = ..., starting_batch_size: int = ..., auto_find_batch_size: bool = ...): # -> partial[Any]:
    """
    Args:
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is cut in half and passed to `function`. `function` must take in a `batch_size` parameter as
    its first argument.
        function (`callable`, *optional*)
            A function to wrap
        starting_batch_size (`int`, *optional*)
            The batch size to try and fit into memory
        auto_find_batch_size (`bool`, *optional*)
            If False, will just execute `function`
    """
    ...

class FSDPOption(ExplicitEnum):
    FULL_SHARD = ...
    SHARD_GRAD_OP = ...
    NO_SHARD = ...
    HYBRID_SHARD = ...
    HYBRID_SHARD_ZERO2 = ...
    OFFLOAD = ...
    AUTO_WRAP = ...


class RemoveColumnsCollator:
    """Wrap the data collator to remove unused columns before they are passed to the collator."""
    def __init__(self, data_collator, signature_columns, logger=..., model_name: Optional[str] = ..., description: Optional[str] = ...) -> None:
        ...
    
    def __call__(self, features: List[dict]):
        ...
    


def check_target_module_exists(optim_target_modules, key: str, return_is_regex: bool = ...): # -> tuple[bool, bool] | bool:
    """A helper method to check if the passed module's key name matches any of the target modules in the optim_target_modules.

    Args:
        optim_target_modules (`Union[str, List[str]]`):
            A list of strings to try to match. Can be also a full string.
        key (`str`):
            A key to search any matches in optim_target_modules
        return_is_regex (`bool`):
            If set to `True`, the method will return whether the passed `optim_target_modules`
            is a regex or not.

    Returns:
        `bool` : True of match object if key matches any target modules from config, False or
        None if no match found
        `bool` : If the matched target module is a regex to silence out the warnings in Trainer
        for extra modules being found (only if `target_module_found=True` for an array of regex).
    """
    ...

