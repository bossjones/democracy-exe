"""
This type stub file was generated by pyright.
"""

from .editblock_coder import EditBlockCoder

class EditBlockFencedCoder(EditBlockCoder):
    """A coder that uses fenced search/replace blocks for code modifications."""
    edit_format = ...
    gpt_prompts = ...


