"""
This type stub file was generated by pyright.
"""

def bracket_parse(s):
    """
    Use Tree.read(s, remove_empty_top_bracketing=True) instead.
    """
    ...

def sinica_parse(s):
    """
    Parse a Sinica Treebank string and return a tree.  Trees are represented as nested brackettings,
    as shown in the following example (X represents a Chinese character):
    S(goal:NP(Head:Nep:XX)|theme:NP(Head:Nhaa:X)|quantity:Dab:X|Head:VL2:X)#0(PERIODCATEGORY)

    :return: A tree corresponding to the string representation.
    :rtype: Tree
    :param s: The string to be converted
    :type s: str
    """
    ...

__all__ = ["bracket_parse", "sinica_parse"]
