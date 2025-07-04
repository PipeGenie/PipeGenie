#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

"""
Grammar parser for the Genetic Programming algorithm.

This module is used to parse the grammar file and create the primitive set
used by the Genetic Programming algorithm.
"""

import importlib
import math
import random
import warnings
from typing import TYPE_CHECKING

from .parser import ParserFactory

from pipegenie.syntax_tree._encoding import NonTerminalNode, TerminalNode
from pipegenie.syntax_tree._primitive_set import PrimitiveSet

if TYPE_CHECKING:
    from collections.abc import Callable


class TerminalCode:
    """
    Class used to generate the code of the terminal nodes.

    Parameters
    ----------
    terminal_code : str
        The code of the terminal node

    hp_list : list of str
        The list of hyper-parameters of the terminal node

    seed : int
        Seed used to initialize the random number generator of the terminal node
    """

    def __init__(self, terminal_code: str, hp_list: list[str], seed: int):
        self.terminal_code = terminal_code
        self.hp_list = hp_list
        self.seed = seed

    def generate_term_code(self) -> 'Callable[[], object]':
        """
        Generate the code of the terminal node.

        Returns
        -------
        callable : callable
            The code of the terminal
        """
        return self.__call__

    def __call__(self, *args: object) -> object:
        package_name, class_name = self.terminal_code.rsplit(".", 1)

        try:
            module = importlib.import_module(package_name)
            estimator_class = getattr(module, class_name)
            estimator = estimator_class()
        except ImportError as e:
            raise ImportError(f"Error importing module {package_name}") from e
        except AttributeError as e:
            raise AttributeError(f"Error importing class {package_name}.{class_name}") from e
        except TypeError as e:
            raise TypeError(f"Error instantiating class {package_name}.{class_name}") from e
        except Exception as e:
            raise Exception(f"Error creating instance of {self.__class__.__qualname__}") from e

        # Assign the hyper-parameters
        for index, hp in enumerate(self.hp_list):
            hp_name = hp.split("::")[1]
            setattr(estimator, hp_name, args[index])

        estimator.random_state = self.seed  # We assume that the seed is the random state
        return estimator

def get_ephemeral_function(hparam: dict[str, str]) -> 'Callable[[], object]':
    """
    Get the ephemeral function of a hyper-parameter.

    Parameters
    ----------
    hparam : dict of str: str
        The hyper-parameter to get the ephemeral function

    Returns
    -------
    ephemeral : callable
        The ephemeral function
    """
    # Differentiate between the types of hyper-parameters
    hp_type = hparam.get('type')

    # A predefined set of categorical values
    if hp_type == "categorical":
        hp_values = hparam.get('values').split(";")
        is_none = hparam.get('default')

        if is_none == "None":
            hp_values.append(None)

        return lambda values=hp_values: random.choice(values)

    # Boolean value
    if "bool" in hp_type:
        if hp_type == "fix_bool":
            hp_val = hparam.get('value')
            hp_val = hp_val == 'True'
            return lambda val=hp_val: val

        return lambda: bool(random.getrandbits(1))

    # Integer value
    if "int" in hp_type:
        if hp_type == "fix_int":
            hp_val = int(hparam.get('value'))
            return lambda val=hp_val: val

        lb = hparam.get('lower')
        ub = hparam.get('upper')
        is_log = hparam.get('log')
        step = hparam.get('step')

        if is_log == "True":
            lb_log = math.log1p(int(lb))
            ub_log = math.log1p(int(ub))

            if step:
                step = float(step)
                num_steps = int(round((ub_log - lb_log) / step)) + 1
                values = [math.expm1(lb_log + i * step) for i in range(num_steps)]
                return lambda values=values: random.choice(values)
            else:
                return lambda: int(math.expm1(random.uniform(lb_log, ub_log)))

        if step:
            step = float(step)
            num_steps = int(round((int(ub) - int(lb)) / step)) + 1
            values = [int(lb) + i * step for i in range(num_steps)]
            return lambda values=values: random.choice(values)

        return lambda lower=lb, upper=ub: random.randint(int(lower), int(upper))

    # Float value
    if "float" in hp_type:
        if hp_type == "fix_float":
            hp_val = float(hparam.get('value'))
            return lambda val=hp_val: val

        lb = hparam.get('lower')
        ub = hparam.get('upper')
        is_log = hparam.get('log')
        step = hparam.get('step')

        if is_log == "True":
            lb_log = math.log1p(float(lb))
            ub_log = math.log1p(float(ub))

            if step:
                step = float(step)
                num_steps = int(round((ub_log - lb_log) / step)) + 1
                values = [math.expm1(lb_log + i * step) for i in range(num_steps)]
                return lambda values=values: random.choice(values)
            else:
                return lambda: math.expm1(random.uniform(lb_log, ub_log))

        if step:
            step = float(step)
            num_steps = int(round((float(ub) - float(lb)) / step)) + 1
            values = [float(lb) + i * step for i in range(num_steps)]
            return lambda values=values: random.choice(values)

        return lambda lower=lb, upper=ub: random.uniform(float(lower), float(upper))

    # Tuple of values
    if "tuple" in hp_type:
        tuple_func = list(map(get_ephemeral_function, hparam.get('children')))
        return lambda funcs=tuple_func: tuple(func() for func in funcs)

    # Union of types
    if "union" in hp_type:
        union_func = list(map(get_ephemeral_function, hparam.get('children')))
        return lambda funcs=union_func: random.choice(funcs)()

    # Unknown type
    warnings.warn(f"Unknown hyper-parameter type {hp_type}", stacklevel=2)
    return lambda: None

def parse_pipe_grammar(
    filename: str,
    grammar_type: str,
    seed: int,
) -> tuple[str, list[TerminalNode], list[NonTerminalNode], PrimitiveSet, dict[str, list[str]]]:
    """
    Parse the grammar file and create the primitive set.

    This function is specially designed to work with
    parametrized functions, e.g. sklearn estimators.

    Parameters
    ----------
    filename : str
        The name of the file containing the grammar

    grammar_type : str
        Format of the grammar file

    seed : int
        The global random seed

    Returns
    -------
    root : str
        The root symbol of the grammar

    terms : list of TerminalNode
        The list of terminal nodes of the grammar

    non_terms : list of NonTerminalNode
        The list of non terminal nodes of the grammar

    pset : PrimitiveSet
        The primitive set of the grammar

    terms_families : dict of str: list of str
        A dictionary containing the terminal nodes grouped by family
    """
    # Load the grammar file
    parser = ParserFactory.get_parser(grammar_type)
    root, p_algorithm_terms, p_hyperparameter_terms, p_non_terms = parser.parse(filename)

    # Create the primitive set
    pset = PrimitiveSet("grammar_pset")

    # Load non terminals
    non_terms = load_productions(p_non_terms)

    # Check how many hyper-parameters each algorithm has
    alg_hp_dict = {}
    non_terms_copy = list(non_terms)

    for non_term in non_terms_copy:
        for prod_list in non_term.production.split(";"):
            # Algorithm with hyper-parameters should be written as "algorithm_symbol(<args>)"
            # Else, it could be either "algorithm_symbol" or "algorithm_symbol()"
            # Get the algorithm symbol
            alg_prod_list = prod_list.split("(")
            alg_symbol = alg_prod_list[0]

            # Check if the algorithm has hyper-parameters
            if len(alg_prod_list) > 1:
                if alg_prod_list[1] == ")":  # algorithm_symbol() case
                    alg_hp_dict[alg_symbol] = []
                else:
                    # Check if <args> is a symbol or the hyper-parameters
                    # If it is a symbol, it should be in the non terminal list
                    # Else, it should be the hyper-parameters list
                    hp_symbol = alg_prod_list[1].strip(")")
                    hp_prod_list = next((prod_list for prod_list in non_terms
                                         if prod_list.symbol == hp_symbol), None)

                    if hp_prod_list is not None:
                        alg_hp_dict[alg_symbol] = hp_prod_list.prod_list
                    else:
                        alg_hp_dict[alg_symbol] = hp_symbol.split(";")
                        # Create an auxiliary non terminal node to ensure consistent structure
                        aux_non_term = NonTerminalNode(alg_symbol + '_hp', hp_symbol)
                        non_terms.append(aux_non_term)
                        # Remove the original non terminal node
                        non_terms.remove(non_term)
                        # Create a new non terminal node that uses the auxiliary one
                        production = non_term.production.replace(hp_symbol, aux_non_term.symbol)
                        non_terms.append(NonTerminalNode(non_term.symbol, production))
            else:  # algorithm_symbol case
                alg_hp_dict[alg_symbol] = []

    # Create the terminals
    terms = []
    terms_families: dict[str, list[str]] = {}

    # Load algorithms
    for algorithm_term in p_algorithm_terms:
        # Get the node properties
        term_name = algorithm_term.get('name')
        term_family = algorithm_term.get('type')
        term_code = algorithm_term.get('code')
        # Get the hyper-parameters of the algorithm
        hp_list = alg_hp_dict.get(term_name, [])

        # Add an unexistent parameter to ensure arity > 0
        term_args = len(hp_list) + 1

        # Add the primitive and terminal
        code = TerminalCode(term_code, hp_list, seed).generate_term_code()
        pset.add_primitive(code, term_args, name=term_name)
        term = TerminalNode(term_name, pset.primitives[-1], term_family)
        terms.append(term)

        # Add the symbol and family to the dictionary
        if term_family in terms_families:
            terms_families[term_family].append(term_name)
        else:
            terms_families[term_family] = [term_name]

    # Load hyper-parameters
    for hparam in p_hyperparameter_terms:
        hp_name = hparam.get('name')
        hp_type = hparam.get('type')

        if "estimator" not in hp_type:
            # Get the ephemeral function
            ephemeral = get_ephemeral_function(hparam)

            if ephemeral is not None:
                pset.add_ephemeral_constant(hp_name, ephemeral)
                # Add the terminal to the primitive set
                term = TerminalNode(hp_name, pset.terminals[-1])
                terms.append(term)
        else:
            # Special case in which the hyper-parameter is an estimator (e.g. AdaBoost)
            # Get the non terminal node holding the valid estimators
            valid_estimators = hparam.get('estimators')
            non_terms_filtered = [non_term for non_term in non_terms
                                  if non_term.symbol == valid_estimators]

            if len(non_terms_filtered) > 0:
                # Add the hyper-parameter as a non terminal node so it can be expanded
                for non_term in non_terms_filtered:
                    non_terms.append(NonTerminalNode(hp_name, non_term.production))
            else:
                warnings.warn(f"Non terminal node {valid_estimators} not found", stacklevel=2)

    # Return the elements composing the grammar
    return root, terms, non_terms, pset, terms_families

def load_productions(p_non_terms: list[dict[str, str]]) -> list[NonTerminalNode]:
    """
    Load the production rules of the non terminal nodes.

    Parameters
    ----------
    p_non_terms : list of dict of str: str
        The list of non terminal nodes

    Returns
    -------
    non_terms : list of NonTerminalNode
        The list of non terminal nodes with their production rules
    """
    return [
        NonTerminalNode(non_term.get('name'), non_term.get('production'))
        for non_term in p_non_terms
    ]
