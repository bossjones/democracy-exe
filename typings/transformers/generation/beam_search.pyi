"""
This type stub file was generated by pyright.
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from ..utils import add_start_docstrings
from .beam_constraints import Constraint

PROCESS_INPUTS_DOCSTRING = ...
FINALIZE_INPUTS_DOCSTRING = ...
class BeamScorer(ABC):
    """
    Abstract base class for all beam scorers that are used for [`~PreTrainedModel.beam_search`] and
    [`~PreTrainedModel.beam_sample`].
    """
    @abstractmethod
    @add_start_docstrings(PROCESS_INPUTS_DOCSTRING)
    def process(self, input_ids: torch.LongTensor, next_scores: torch.FloatTensor, next_tokens: torch.LongTensor, next_indices: torch.LongTensor, **kwargs) -> Tuple[torch.Tensor]:
        ...
    
    @abstractmethod
    @add_start_docstrings(FINALIZE_INPUTS_DOCSTRING)
    def finalize(self, input_ids: torch.LongTensor, next_scores: torch.FloatTensor, next_tokens: torch.LongTensor, next_indices: torch.LongTensor, max_length: int, **kwargs) -> torch.LongTensor:
        ...
    


class BeamSearchScorer(BeamScorer):
    r"""
    [`BeamScorer`] implementing standard beam search decoding.

    Adapted in part from [Facebook's XLM beam search
    code](https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529).

    Reference for the diverse beam search algorithm and implementation [Ashwin Kalyan's DBS
    implementation](https://github.com/ashwinkalyan/dbs/blob/master/dbs/beam_utils.lua)

    Args:
        batch_size (`int`):
            Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
        num_beams (`int`):
            Number of beams for beam search.
        device (`torch.device`):
            Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
            allocated.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences.
        do_early_stopping (`bool` or `str`, *optional*, defaults to `False`):
            Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
            `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
            beam search algorithm).
        num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            [`~transformers.BeamSearchScorer.finalize`].
        num_beam_groups (`int`, *optional*, defaults to 1):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        max_length (`int`, *optional*):
            The maximum length of the sequence to be generated.
    """
    def __init__(self, batch_size: int, num_beams: int, device: torch.device, length_penalty: Optional[float] = ..., do_early_stopping: Optional[Union[bool, str]] = ..., num_beam_hyps_to_keep: Optional[int] = ..., num_beam_groups: Optional[int] = ..., max_length: Optional[int] = ...) -> None:
        ...
    
    @property
    def is_done(self) -> bool:
        ...
    
    def process(self, input_ids: torch.LongTensor, next_scores: torch.FloatTensor, next_tokens: torch.LongTensor, next_indices: torch.LongTensor, pad_token_id: Optional[Union[int, torch.Tensor]] = ..., eos_token_id: Optional[Union[int, List[int], torch.Tensor]] = ..., beam_indices: Optional[torch.LongTensor] = ..., group_index: Optional[int] = ..., decoder_prompt_len: Optional[int] = ...) -> Dict[str, torch.Tensor]:
        ...
    
    def finalize(self, input_ids: torch.LongTensor, final_beam_scores: torch.FloatTensor, final_beam_tokens: torch.LongTensor, final_beam_indices: torch.LongTensor, max_length: int, pad_token_id: Optional[Union[int, torch.Tensor]] = ..., eos_token_id: Optional[Union[int, List[int], torch.Tensor]] = ..., beam_indices: Optional[torch.LongTensor] = ..., decoder_prompt_len: Optional[int] = ...) -> Tuple[torch.LongTensor]:
        ...
    


class ConstrainedBeamSearchScorer(BeamScorer):
    r"""
    [`BeamScorer`] implementing constrained beam search decoding.


    Args:
        batch_size (`int`):
            Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
        num_beams (`int`):
            Number of beams for beam search.
        constraints (`List[Constraint]`):
            A list of positive constraints represented as `Constraint` objects that must be fulfilled in the generation
            output. For more information, the documentation of [`Constraint`] should be read.
        device (`torch.device`):
            Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
            allocated.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences.
        do_early_stopping (`bool` or `str`, *optional*, defaults to `False`):
            Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
            `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
            beam search algorithm).
        num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            [`~transformers.BeamSearchScorer.finalize`].
        num_beam_groups (`int`, *optional*, defaults to 1):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        max_length (`int`, *optional*):
            The maximum length of the sequence to be generated.
    """
    def __init__(self, batch_size: int, num_beams: int, constraints: List[Constraint], device: torch.device, length_penalty: Optional[float] = ..., do_early_stopping: Optional[Union[bool, str]] = ..., num_beam_hyps_to_keep: Optional[int] = ..., num_beam_groups: Optional[int] = ..., max_length: Optional[int] = ...) -> None:
        ...
    
    @property
    def is_done(self) -> bool:
        ...
    
    def make_constraint_states(self, n): # -> list[ConstraintListState]:
        ...
    
    def check_completes_constraints(self, sequence): # -> bool:
        ...
    
    def process(self, input_ids: torch.LongTensor, next_scores: torch.FloatTensor, next_tokens: torch.LongTensor, next_indices: torch.LongTensor, scores_for_all_vocab: torch.FloatTensor, pad_token_id: Optional[Union[int, torch.Tensor]] = ..., eos_token_id: Optional[Union[int, List[int], torch.Tensor]] = ..., beam_indices: Optional[torch.LongTensor] = ..., decoder_prompt_len: Optional[int] = ...) -> Tuple[torch.Tensor]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size * num_beams, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using any class inheriting from [`PreTrainedTokenizer`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            next_scores (`torch.FloatTensor` of shape `(batch_size, 2 * num_beams)`):
                Current scores of the top `2 * num_beams` non-finished beam hypotheses.
            next_tokens (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
                `input_ids` of the tokens corresponding to the top `2 * num_beams` non-finished beam hypotheses.
            next_indices (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
                Beam indices indicating to which beam hypothesis the `next_tokens` correspond.
            scores_for_all_vocab (`torch.FloatTensor` of shape `(batch_size * num_beams, sequence_length)`):
                The scores of all tokens in the vocabulary for each of the beam hypotheses.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            beam_indices (`torch.LongTensor`, *optional*):
                Beam indices indicating to which beam hypothesis each token correspond.
            decoder_prompt_len (`int`, *optional*):
                The length of prompt that is included in the input to decoder.
        Return:
            `UserDict`: A dictionary composed of the fields as defined above:

                - **next_beam_scores** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Updated scores of
                  all
                non-finished beams.

                - **next_beam_tokens** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Next tokens to be
                  added
                to the non-finished beam_hypotheses.
                - **next_beam_indices** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Beam indices
                indicating to which beam the next tokens shall be added.
        """
        ...
    
    def step_sentence_constraint(self, batch_idx: int, input_ids: torch.LongTensor, vocab_scores: torch.FloatTensor, sent_beam_scores: torch.FloatTensor, sent_beam_tokens: torch.LongTensor, sent_beam_indices: torch.LongTensor, push_progress: bool = ...): # -> tuple[FloatTensor, LongTensor, LongTensor]:
        ...
    
    def finalize(self, input_ids: torch.LongTensor, final_beam_scores: torch.FloatTensor, final_beam_tokens: torch.LongTensor, final_beam_indices: torch.LongTensor, max_length: int, pad_token_id: Optional[Union[int, torch.Tensor]] = ..., eos_token_id: Optional[Union[int, List[int], torch.Tensor]] = ..., beam_indices: Optional[torch.LongTensor] = ..., decoder_prompt_len: Optional[int] = ...) -> Tuple[torch.LongTensor]:
        ...
    


class BeamHypotheses:
    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool, max_length: Optional[int] = ...) -> None:
        """
        Initialize n-best list of hypotheses.
        """
        ...
    
    def __len__(self): # -> int:
        """
        Number of hypotheses in the list.
        """
        ...
    
    def add(self, hyp: torch.LongTensor, sum_logprobs: float, beam_indices: Optional[torch.LongTensor] = ..., generated_len: Optional[int] = ...): # -> None:
        """
        Add a new hypothesis to the list.
        """
        ...
    
    def is_done(self, best_sum_logprobs: float, cur_len: int, decoder_prompt_len: Optional[int] = ...) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """
        ...
    


