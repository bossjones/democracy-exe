"""
This type stub file was generated by pyright.
"""

from nltk.inference.api import BaseModelBuilderCommand, ModelBuilder
from nltk.inference.prover9 import Prover9CommandParent, Prover9Parent

"""
A model builder that makes use of the external 'Mace4' package.
"""
class MaceCommand(Prover9CommandParent, BaseModelBuilderCommand):
    """
    A ``MaceCommand`` specific to the ``Mace`` model builder.  It contains
    a print_assumptions() method that is used to print the list
    of assumptions in multiple formats.
    """
    _interpformat_bin = ...
    def __init__(self, goal=..., assumptions=..., max_models=..., model_builder=...) -> None:
        """
        :param goal: Input expression to prove
        :type goal: sem.Expression
        :param assumptions: Input expressions to use as assumptions in
            the proof.
        :type assumptions: list(sem.Expression)
        :param max_models: The maximum number of models that Mace will try before
            simply returning false. (Use 0 for no maximum.)
        :type max_models: int
        """
        ...
    
    @property
    def valuation(mbc): # -> None:
        ...
    


class Mace(Prover9Parent, ModelBuilder):
    _mace4_bin = ...
    def __init__(self, end_size=...) -> None:
        ...
    


def spacer(num=...): # -> None:
    ...

def decode_result(found): # -> str:
    """
    Decode the result of model_found()

    :param found: The output of model_found()
    :type found: bool
    """
    ...

def test_model_found(arguments): # -> None:
    """
    Try some proofs and exhibit the results.
    """
    ...

def test_build_model(arguments): # -> None:
    """
    Try to build a ``nltk.sem.Valuation``.
    """
    ...

def test_transform_output(argument_pair): # -> None:
    """
    Transform the model into various Mace4 ``interpformat`` formats.
    """
    ...

def test_make_relation_set(): # -> None:
    ...

arguments = ...
def demo(): # -> None:
    ...

if __name__ == "__main__":
    ...
