"""
This type stub file was generated by pyright.
"""

from .base_coder import Coder

no_match_error = ...
not_unique_error = ...
other_hunks_applied = ...
class UnifiedDiffCoder(Coder):
    """A coder that uses unified diff format for code modifications."""
    edit_format = ...
    gpt_prompts = ...
    def get_edits(self): # -> list[Any]:
        ...
    
    def apply_edits(self, edits): # -> None:
        ...
    


def do_replace(fname, content, hunk): # -> LiteralString | str | None:
    ...

def collapse_repeats(s): # -> LiteralString:
    ...

def apply_hunk(content, hunk): # -> LiteralString | str | None:
    ...

def flexi_just_search_and_replace(texts): # -> LiteralString | str | None:
    ...

def make_new_lines_explicit(content, hunk): # -> list[str]:
    ...

def cleanup_pure_whitespace_lines(lines): # -> list[Any]:
    ...

def normalize_hunk(hunk): # -> list[str]:
    ...

def directly_apply_hunk(content, hunk): # -> LiteralString | str | None:
    ...

def apply_partial_hunk(content, preceding_context, changes, following_context): # -> LiteralString | str | None:
    ...

def find_diffs(content): # -> list[Any]:
    ...

def process_fenced_block(lines, start_line_num): # -> tuple[int, list[Any]]:
    ...

def hunk_to_before_after(hunk, lines=...): # -> tuple[list[Any], list[Any]] | tuple[LiteralString, LiteralString]:
    ...
