"""
This type stub file was generated by pyright.
"""

"""
ALINE
https://webdocs.cs.ualberta.ca/~kondrak/
Copyright 2002 by Grzegorz Kondrak.

ALINE is an algorithm for aligning phonetic sequences, described in [1].
This module is a port of Kondrak's (2002) ALINE. It provides functions for
phonetic sequence alignment and similarity analysis. These are useful in
historical linguistics, sociolinguistics and synchronic phonology.

ALINE has parameters that can be tuned for desired output. These parameters are:
- C_skip, C_sub, C_exp, C_vwl
- Salience weights
- Segmental features

In this implementation, some parameters have been changed from their default
values as described in [1], in order to replicate published results. All changes
are noted in comments.

Example usage
-------------

# Get optimal alignment of two phonetic sequences

>>> align('θin', 'tenwis') # doctest: +SKIP
[[('θ', 't'), ('i', 'e'), ('n', 'n'), ('-', 'w'), ('-', 'i'), ('-', 's')]]

[1] G. Kondrak. Algorithms for Language Reconstruction. PhD dissertation,
University of Toronto.
"""
inf = ...
C_skip = ...
C_sub = ...
C_exp = ...
C_vwl = ...
consonants = ...
R_c = ...
R_v = ...
similarity_matrix = ...
salience = ...
feature_matrix = ...
def align(str1, str2, epsilon=...): # -> list[Any]:
    """
    Compute the alignment of two phonetic strings.

    :param str str1: First string to be aligned
    :param str str2: Second string to be aligned

    :type epsilon: float (0.0 to 1.0)
    :param epsilon: Adjusts threshold similarity score for near-optimal alignments

    :rtype: list(list(tuple(str, str)))
    :return: Alignment(s) of str1 and str2

    (Kondrak 2002: 51)
    """
    ...

def sigma_skip(p): # -> int:
    """
    Returns score of an indel of P.

    (Kondrak 2002: 54)
    """
    ...

def sigma_sub(p, q): # -> Any | float | int:
    """
    Returns score of a substitution of P with Q.

    (Kondrak 2002: 54)
    """
    ...

def sigma_exp(p, q): # -> Any | float | int:
    """
    Returns score of an expansion/compression.

    (Kondrak 2002: 54)
    """
    ...

def delta(p, q): # -> Any | float | Literal[0]:
    """
    Return weighted sum of difference between P and Q.

    (Kondrak 2002: 54)
    """
    ...

def diff(p, q, f): # -> float:
    """
    Returns difference between phonetic segments P and Q for feature F.

    (Kondrak 2002: 52, 54)
    """
    ...

def R(p, q): # -> list[str]:
    """
    Return relevant features for segment comparison.

    (Kondrak 2002: 54)
    """
    ...

def V(p): # -> int:
    """
    Return vowel weight if P is vowel.

    (Kondrak 2002: 54)
    """
    ...

def demo(): # -> None:
    """
    A demonstration of the result of aligning phonetic sequences
    used in Kondrak's (2002) dissertation.
    """
    ...

cognate_data = ...
if __name__ == "__main__":
    ...
