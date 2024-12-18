"""
This type stub file was generated by pyright.
"""

from .data_collator import DataCollatorForLanguageModeling, DataCollatorForPermutationLanguageModeling, DataCollatorForSOP, DataCollatorForSeq2Seq, DataCollatorForTokenClassification, DataCollatorForWholeWordMask, DataCollatorWithFlattening, DataCollatorWithPadding, DefaultDataCollator, default_data_collator
from .metrics import glue_compute_metrics, xnli_compute_metrics
from .processors import DataProcessor, InputExample, InputFeatures, SingleSentenceClassificationProcessor, SquadExample, SquadFeatures, SquadV1Processor, SquadV2Processor, glue_convert_examples_to_features, glue_output_modes, glue_processors, glue_tasks_num_labels, squad_convert_examples_to_features, xnli_output_modes, xnli_processors, xnli_tasks_num_labels

