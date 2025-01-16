"""
This type stub file was generated by pyright.
"""

class RelativeIndenter:
    """Rewrites text files to have relative indentation, which involves
    reformatting the leading white space on lines.  This format makes
    it easier to search and apply edits to pairs of code blocks which
    may differ significantly in their overall level of indentation.

    It removes leading white space which is shared with the preceding
    line.

    Original:
    ```
            Foo # indented 8
                Bar # indented 4 more than the previous line
                Baz # same indent as the previous line
                Fob # same indent as the previous line
    ```

    Becomes:
    ```
            Foo # indented 8
        Bar # indented 4 more than the previous line
    Baz # same indent as the previous line
    Fob # same indent as the previous line
    ```

    If the current line is *less* indented then the previous line,
    uses a unicode character to indicate outdenting.

    Original
    ```
            Foo
                Bar
                Baz
            Fob # indented 4 less than the previous line
    ```

    Becomes:
    ```
            Foo
        Bar
    Baz
    ←←←←Fob # indented 4 less than the previous line
    ```

    This is a similar original to the last one, but every line has
    been uniformly outdented:
    ```
    Foo
        Bar
        Baz
    Fob # indented 4 less than the previous line
    ```

    It becomes this result, which is very similar to the previous
    result.  Only the white space on the first line differs.  From the
    word Foo onwards, it is identical to the previous result.
    ```
    Foo
        Bar
    Baz
    ←←←←Fob # indented 4 less than the previous line
    ```

    """
    def __init__(self, texts) -> None:
        """
        Based on the texts, choose a unicode character that isn't in any of them.
        """
        ...
    
    def select_unique_marker(self, chars): # -> str:
        ...
    
    def make_relative(self, text): # -> LiteralString:
        """
        Transform text to use relative indents.
        """
        ...
    
    def make_absolute(self, text): # -> LiteralString:
        """
        Transform text from relative back to absolute indents.
        """
        ...
    


def map_patches(texts, patches, debug):
    ...

example = ...
def relative_indent(texts): # -> tuple[RelativeIndenter, list[str]]:
    ...

line_padding = ...
def line_pad(text):
    ...

def line_unpad(text): # -> None:
    ...

def dmp_apply(texts, remap=...): # -> None:
    ...

def lines_to_chars(lines, mapping): # -> LiteralString:
    ...

def dmp_lines_apply(texts, remap=...): # -> LiteralString | None:
    ...

def diff_lines(search_text, replace_text): # -> list[Any]:
    ...

def search_and_replace(texts): # -> None:
    ...

def git_cherry_pick_osr_onto_o(texts): # -> str | None:
    ...

def git_cherry_pick_sr_onto_so(texts): # -> str | None:
    ...

class SearchTextNotUnique(ValueError):
    ...


all_preprocs = ...
always_relative_indent = ...
editblock_strategies = ...
never_relative = ...
udiff_strategies = ...
def flexible_search_and_replace(texts, strategies): # -> LiteralString | str | None:
    """Try a series of search/replace methods, starting from the most
    literal interpretation of search_text. If needed, progress to more
    flexible methods, which can accommodate divergence between
    search_text and original_text and yet still achieve the desired
    edits.
    """
    ...

def reverse_lines(text): # -> str:
    ...

def try_strategy(texts, strategy, preproc): # -> LiteralString | str | None:
    ...

def strip_blank_lines(texts): # -> list[Any]:
    ...

def read_text(fname): # -> str:
    ...

def proc(dname): # -> list[Any] | None:
    ...

def colorize_result(result): # -> str:
    ...

def main(dnames): # -> None:
    ...

if __name__ == "__main__":
    status = ...