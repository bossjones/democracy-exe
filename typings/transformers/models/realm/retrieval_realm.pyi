"""
This type stub file was generated by pyright.
"""

import os
from typing import Optional, Union

import numpy as np

"""REALM Retriever model implementation."""
_REALM_BLOCK_RECORDS_FILENAME = ...
logger = ...
def convert_tfrecord_to_np(block_records_path: str, num_block_records: int) -> np.ndarray:
    ...

class ScaNNSearcher:
    """Note that ScaNNSearcher cannot currently be used within the model. In future versions, it might however be included."""
    def __init__(self, db, num_neighbors, dimensions_per_block=..., num_leaves=..., num_leaves_to_search=..., training_sample_size=...) -> None:
        """Build scann searcher."""
        ...

    def search_batched(self, question_projection):
        ...



class RealmRetriever:
    """The retriever of REALM outputting the retrieved evidence block and whether the block has answers as well as answer
    positions."

        Parameters:
            block_records (`np.ndarray`):
                A numpy array which cantains evidence texts.
            tokenizer ([`RealmTokenizer`]):
                The tokenizer to encode retrieved texts.
    """
    def __init__(self, block_records, tokenizer) -> None:
        ...

    def __call__(self, retrieved_block_ids, question_input_ids, answer_ids, max_length=..., return_tensors=...): # -> tuple[list[Any], list[Any], list[Any], Any] | tuple[None, None, None, Any]:
        ...

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *init_inputs, **kwargs): # -> Self:
        ...

    def save_pretrained(self, save_directory): # -> None:
        ...

    def block_has_answer(self, concat_inputs, answer_ids): # -> tuple[list[Any], list[Any], list[Any]]:
        """check if retrieved_blocks has answers."""
        ...
