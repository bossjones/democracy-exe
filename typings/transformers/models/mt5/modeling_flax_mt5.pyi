"""
This type stub file was generated by pyright.
"""

import jax.numpy as jnp
from ..t5.modeling_flax_t5 import FlaxT5EncoderModel, FlaxT5ForConditionalGeneration, FlaxT5Model
from .configuration_mt5 import MT5Config

"""Flax mT5 model."""
logger = ...
_CONFIG_FOR_DOC = ...
def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    Shift input ids one token to the right.
    """
    ...

class FlaxMT5Model(FlaxT5Model):
    r"""
    This class overrides [`FlaxT5Model`]. Please check the superclass for the appropriate documentation alongside usage
    examples.

    Examples:

    ```python
    >>> from transformers import FlaxMT5Model, AutoTokenizer

    >>> model = FlaxMT5Model.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, return_tensors="np")

    >>> decoder_input_ids = tokenizer(text_target=summary, return_tensors="np").input_ids

    >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=decoder_input_ids)
    >>> hidden_states = outputs.last_hidden_state
    ```"""
    model_type = ...
    config_class = MT5Config


class FlaxMT5EncoderModel(FlaxT5EncoderModel):
    r"""
    This class overrides [`FlaxT5EncoderModel`]. Please check the superclass for the appropriate documentation
    alongside usage examples.

    Examples:

    ```python
    >>> from transformers import FlaxT5EncoderModel, AutoTokenizer

    >>> model = FlaxT5EncoderModel.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, return_tensors="np")

    >>> decoder_input_ids = tokenizer(text_target=summary, return_tensors="np").input_ids

    >>> outputs = model(input_ids=inputs["input_ids"])
    >>> hidden_states = outputs.last_hidden_state
    ```"""
    model_type = ...
    config_class = MT5Config


class FlaxMT5ForConditionalGeneration(FlaxT5ForConditionalGeneration):
    r"""
    This class overrides [`FlaxT5ForConditionalGeneration`]. Please check the superclass for the appropriate
    documentation alongside usage examples.

    Examples:

    ```python
    >>> from transformers import FlaxMT5ForConditionalGeneration, AutoTokenizer

    >>> model = FlaxMT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, return_tensors="np")

    >>> decoder_input_ids = tokenizer(text_target=summary, return_tensors="np").input_ids

    >>> outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
    >>> logits = outputs.logits
    ```"""
    model_type = ...
    config_class = MT5Config


