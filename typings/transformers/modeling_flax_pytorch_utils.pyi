"""
This type stub file was generated by pyright.
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple
from . import is_safetensors_available, is_torch_available

"""PyTorch - Flax general utilities."""
if is_torch_available():
    ...
if is_safetensors_available():
    ...
logger = ...
def load_pytorch_checkpoint_in_flax_state_dict(flax_model, pytorch_checkpoint_path, is_sharded, allow_missing_keys=...):
    """Load pytorch checkpoints in a flax model"""
    ...

def rename_key_and_reshape_tensor(pt_tuple_key: Tuple[str], pt_tensor: np.ndarray, random_flax_state_dict: Dict[str, jnp.ndarray], model_prefix: str) -> (Tuple[str], np.ndarray):
    """Rename PT weight names to corresponding Flax weight names and reshape tensor if necessary"""
    ...

def convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model):
    ...

def convert_pytorch_sharded_state_dict_to_flax(shard_filenames, flax_model):
    ...

def load_flax_checkpoint_in_pytorch_model(model, flax_checkpoint_path):
    """Load flax checkpoints in a PyTorch model"""
    ...

def load_flax_weights_in_pytorch_model(pt_model, flax_state):
    """Load flax checkpoints in a PyTorch model"""
    ...

