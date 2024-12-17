"""
This type stub file was generated by pyright.
"""

from nltk.inference.api import BaseProverCommand, Prover

"""
A theorem prover that makes use of the external 'Prover9' package.
"""
p9_return_codes = ...
class Prover9CommandParent:
    """
    A common base class used by both ``Prover9Command`` and ``MaceCommand``,
    which is responsible for maintaining a goal and a set of assumptions,
    and generating prover9-style input files from them.
    """
    def print_assumptions(self, output_format=...): # -> None:
        """
        Print the list of the current assumptions.
        """
        ...
    


class Prover9Command(Prover9CommandParent, BaseProverCommand):
    """
    A ``ProverCommand`` specific to the ``Prover9`` prover.  It contains
    the a print_assumptions() method that is used to print the list
    of assumptions in multiple formats.
    """
    def __init__(self, goal=..., assumptions=..., timeout=..., prover=...) -> None:
        """
        :param goal: Input expression to prove
        :type goal: sem.Expression
        :param assumptions: Input expressions to use as assumptions in
            the proof.
        :type assumptions: list(sem.Expression)
        :param timeout: number of seconds before timeout; set to 0 for
            no timeout.
        :type timeout: int
        :param prover: a prover.  If not set, one will be created.
        :type prover: Prover9
        """
        ...
    
    def decorate_proof(self, proof_string, simplify=...):
        """
        :see BaseProverCommand.decorate_proof()
        """
        ...
    


class Prover9Parent:
    """
    A common class extended by both ``Prover9`` and ``Mace <mace.Mace>``.
    It contains the functionality required to convert NLTK-style
    expressions into Prover9-style expressions.
    """
    _binary_location = ...
    def config_prover9(self, binary_location, verbose=...): # -> None:
        ...
    
    def prover9_input(self, goal, assumptions): # -> str:
        """
        :return: The input string that should be provided to the
            prover9 binary.  This string is formed based on the goal,
            assumptions, and timeout value of this object.
        """
        ...
    
    def binary_locations(self): # -> list[str]:
        """
        A list of directories that should be searched for the prover9
        executables.  This list is used by ``config_prover9`` when searching
        for the prover9 executables.
        """
        ...
    


def convert_to_prover9(input): # -> list[Any] | str:
    """
    Convert a ``logic.Expression`` to Prover9 format.
    """
    ...

class Prover9(Prover9Parent, Prover):
    _prover9_bin = ...
    _prooftrans_bin = ...
    def __init__(self, timeout=...) -> None:
        ...
    
    def prover9_input(self, goal, assumptions): # -> str:
        """
        :see: Prover9Parent.prover9_input
        """
        ...
    


class Prover9Exception(Exception):
    def __init__(self, returncode, message) -> None:
        ...
    


class Prover9FatalException(Prover9Exception):
    ...


class Prover9LimitExceededException(Prover9Exception):
    ...


def test_config(): # -> None:
    ...

def test_convert_to_prover9(expr): # -> None:
    """
    Test that parsing works OK.
    """
    ...

def test_prove(arguments): # -> None:
    """
    Try some proofs and exhibit the results.
    """
    ...

arguments = ...
expressions = ...
def spacer(num=...): # -> None:
    ...

def demo(): # -> None:
    ...

if __name__ == "__main__":
    ...
