#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

import random
from ast import literal_eval
from copy import deepcopy
from inspect import isclass
from math import log1p
from typing import TYPE_CHECKING, SupportsIndex, overload

import numpy as np

from pipegenie.pipeline import Pipeline
from pipegenie.utils import is_bool, is_number, is_tuple

from ._primitive_set import Primitive

if TYPE_CHECKING:
    from typing import Optional, Union

    from ._primitive_set import PrimitiveSet


class TerminalNode:
    """
    Terminal node of a SyntaxTree.

    Corresponds to both primitives and arguments in GP.
    Terminal nodes have 0 arity and include the code being
    executed when the SyntaxTree is evaluated.

    Parameters
    ----------
    symbol : str
        Symbol to represent the terminal node.

    code : Primitive type or Primitive object
        Code to be executed when the SyntaxTree is evaluated.

    family : str, default=None
        Family to which the terminal belongs.
    """

    __slots__ = ('symbol', 'code', 'family')

    def __init__(
        self,
        symbol: str,
        code: 'Union[type[Primitive], Primitive]',
        family: 'Optional[str]' = None,
    ):
        self.symbol = symbol
        self.code = code
        self.family = family

    def __str__(self) -> str:
        return self.symbol

    @property
    def arity(self) -> int:
        return 0

class NonTerminalNode:
    """
    Non-terminal node of a SyntaxTree.

    Each non-terminal node corresponds to a production rule of a grammar.
    Thus, each node has a symbol (production rule left-sided) and the production itself
    (production rule right-sided)

    Parameters
    ----------
    symbol : str
        Symbol of the production rule.

    production : str
        Production rule.

    Examples
    --------
    <example> :: <a> <b> <c>
    >>> node = NonTerminalNode('example', 'a;b;c')
    >>> node.symbol
    'example'
    >>> node.production
    'a;b;c'
    >>> node.prod_list
    ['a', 'b', 'c']
    """

    __slots__ = ('symbol', 'production', 'prod_list')

    def __init__(self, symbol: str, production: str):
        self.symbol = symbol
        # Python does not allow to use a list as the key of a dictionary
        self.production = production
        prod_list = production.split(';')
        new_prod_list = []

        for prod in prod_list:
            new_prod = prod.split('(')

            if len(new_prod) > 1:
                if new_prod[1] == ')':
                    new_prod_list.append(new_prod)
                else:
                    new_prod_list.extend([new_prod[0], new_prod[1].strip(')')])
            else:
                new_prod_list.append(prod)

        self.prod_list = new_prod_list

    def __str__(self) -> str:
        return self.symbol

    @property
    def arity(self) -> int:
        return len(self.prod_list)


class SyntaxTree(list['Union[TerminalNode, NonTerminalNode]']):
    """
    Gramatically based genetic programming tree.

    Tree specifically formatted for optimization of G3P operations. The tree
    is represented with a list where the nodes (terminals and non-terminals)
    are appended in a depth-first order. The nodes appended to the tree are
    required to have an 'arity' property which defines the arity of the
    primitive. An arity of 0 is expected from terminals nodes.

    Parameters
    ----------
    content : list
        List of nodes to initialize the tree.
    """

    def __init__(self, content: list['Union[TerminalNode, NonTerminalNode]']):
        super().__init__(content)

    def __deepcopy__(self, memo: dict[int, 'SyntaxTree']) -> 'SyntaxTree':
        new = self.__class__(self)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

    def __str__(self) -> str:
        """
        Return a string representation of the tree in a human-readable format.
        """
        return " ".join([str(node) for node in self])

    def search_subtree(self, begin: int) -> slice:
        """
        Search for a subtree starting at the given index.

        Parameters
        ----------
        begin : int
            Index to start the search.

        Returns
        -------
        subtree : slice
            Slice object representing the subtree.
        """
        end = begin + 1
        total = self[begin].arity

        while total > 0:
            total += self[end].arity - 1
            end += 1

        return slice(begin, end)

    def to_graph(self) -> tuple[dict[str, str], list[tuple[str, str]]]:
        """
        Create a directed graph representation of the tree.

        Returns
        -------
        nodes : dict of str: str
            Dictionary containing the nodes of the graph.
            The keys are the node ids used in the edges.
            The values are the node labels.

        edges : list of tuples of str, str
            List of tuples containing the edges of the graph.
            The first element of the tuple is the parent node id
            and the second element is the child node id.
        """

        def add_nodes_edges(tree: SyntaxTree, index: int, parent: 'Optional[str]' = None) -> None:
            node = tree[index]
            node_id = f"{index}_{node.symbol}".replace("::", "_")  # graphviz issue with "::"

            if isinstance(node, TerminalNode) and "::" in node.symbol: # Hyper-parameter
                nodes[node_id] = str(node.symbol.split("::")[1]) + "=" + str(node.code.name)
            else:
                nodes[node_id] = str(node)

            if parent:
                edges.append((parent, node_id))

            if isinstance(node, NonTerminalNode):
                current_index = index + 1

                for _ in node.prod_list:
                    subtree = tree.search_subtree(current_index)
                    add_nodes_edges(tree, current_index, node_id)
                    current_index = subtree.stop

        nodes: dict[str, str] = {}
        edges: list[tuple[str, str]] = []
        add_nodes_edges(self, 0)
        return nodes, edges

    @overload
    def __getitem__(self, key: SupportsIndex) -> 'Union[TerminalNode, NonTerminalNode]': ...
    @overload
    def __getitem__(self, key: slice) -> 'SyntaxTree': ...

    def __getitem__(
        self,
        key: 'Union[SupportsIndex, slice]',
    ) -> 'Union[TerminalNode, NonTerminalNode, SyntaxTree]':
        if isinstance(key, SupportsIndex):
            result: 'Union[TerminalNode, NonTerminalNode]' = super().__getitem__(key)
            return result

        if isinstance(key, slice):
            return self.__class__(super().__getitem__(key))

        raise TypeError(f"Invalid key type: {type(key)}")


class SyntaxTreeSchema:
    """
    Schema used to guarantee valid individuals are created.

    Constains the set of terminals and non-terminals that can be used to construct a
    SyntaxTree, among others. It also ensures the construction of trees with a
    bounded size.

    Parameters
    ----------
    root : str
        Symbol of the root production rule.

    max_deriv_size : int
        Maximum number of derivations allowed in the tree.

    terminals : list of TerminalNode
        List of terminal nodes.

    non_terminals : list of NonTerminalNode
        List of non-terminal nodes.
    """

    __slots__ = ('primitive_set', 'terminals', 'non_terminals',
                'max_deriv_size', 'min_deriv_size', 'root',
                'terminals_map', 'non_terminals_map', 'cardinality_map',
                '_min_derivs_cache', '_max_derivs_cache', '_is_recursive_cache',
                '_partitions_cache', '_parent_cache')

    def __init__(
        self,
        root: str,
        max_deriv_size: int,
        terminals: list[TerminalNode],
        non_terminals: list[NonTerminalNode],
    ):
        self.root = root
        self.max_deriv_size = max_deriv_size
        self.terminals = terminals
        self.non_terminals = non_terminals
        self.min_deriv_size = -1
        self.terminals_map: dict[str, TerminalNode] = {}
        self.non_terminals_map: dict[str, list[NonTerminalNode]] = {}
        self.cardinality_map: dict[str, list[int]] = {}
        self._min_derivs_cache: dict[str, int] = {}
        self._max_derivs_cache: dict[str, int] = {}
        self._is_recursive_cache: dict[str, bool] = {}
        self._partitions_cache: \
            dict[tuple[int, int, tuple[int, ...], tuple[int, ...]], list[list[int]]] = {}
        self._parent_cache: dict[str, list[str]] = {}
        self._configure()

    def _build_terminals_dict(self) -> None:
        """
        Build and set the terminals dictionary.
        """
        self.terminals_map = {terminal.symbol: terminal for terminal in self.terminals}

    def _build_non_terminals_dict(self) -> None:
        """
        Build and set the non terminals dictionary.
        """
        # Used to classify symbols
        self.non_terminals_map = {}

        # Classify non-term symbols
        for node in self.non_terminals:
            self.non_terminals_map.setdefault(node.symbol, []).append(node)

    def _build_cardinality_dict(self) -> None:
        """
        Build and set the cardinality dictionary.

        This dictionary contains cardinality of all production rules
        (from cero to max number of derivations)
        """
        self.cardinality_map = {non_term.production: [-1] * (1 + self.max_deriv_size)
                                for non_term in self.non_terminals}

    def _calc_min_deriv_size(self) -> None:
        """
        Calculate and set the minimum number of derivations.
        """
        self.min_deriv_size = self.min_derivations(self.root)

    def create_syntax_tree(self) -> SyntaxTree:
        """
        Create a new syntax tree of a random size in range (minDerivSize, maxDerivSize).

        Returns
        -------
        stree : SyntaxTree
            SyntaxTree conformant with the grammar defined.
        """
        # Create resulting tree
        stree = SyntaxTree([])
        # Randomly determine the number of derivarion
        number_of_derivs = random.randint(self.min_deriv_size, self.max_deriv_size)
        # Fill result branch
        self.fill_tree_branch(stree, self.root, number_of_derivs)
        # Return resulting tree
        return stree

    def fill_tree_branch(self, tree: SyntaxTree, symbol: str, number_of_derivs: int) -> None:
        """
        Fill a SyntaxTree using the symbol and the allowed number of derivations.

        Parameters
        ----------
        tree : SyntaxTree
            Tree to fill.

        symbol : str
            The new symbol (terminal or non-terminal) to add

        number_of_derivs : int
            The number of derivations.
        """
        if symbol in self.terminals_map:
            term = self.terminals_map[symbol]

            if isclass(term.code):
                tree.append(TerminalNode(symbol, term.code()))
            else:
                tree.append(TerminalNode(symbol, term.code))
        else:
            # Select a production rule
            selected_prod = self._select_production(symbol, number_of_derivs)
            # Expand the non terminal node
            self._expand_non_terminal(tree, symbol, number_of_derivs, selected_prod)

    def _expand_non_terminal(
        self,
        tree: SyntaxTree,
        symbol: str,
        number_of_derivs: int,
        selected_prod: 'Optional[NonTerminalNode]' = None,
    ) -> None:
        if selected_prod is not None:
            # Add this node
            tree.append(selected_prod)
            # Select a partition for this production rule
            selected_part = self._select_partition(selected_prod.prod_list, number_of_derivs - 1)
            # Apply partition, expanding production symbols
            sel_prod_size = len(selected_part)

            for i in range(sel_prod_size):
                self.fill_tree_branch(tree, selected_prod.prod_list[i], selected_part[i])
        else:
            self.fill_tree_branch(tree, symbol, number_of_derivs - 1)

    def _select_production(
        self,
        symbol: str,
        number_of_derivs: int,
        selected_prod: 'Optional[NonTerminalNode]' = None,
    ) -> 'Optional[NonTerminalNode]':
        """
        Select a production rule for a symbol of the grammar.

        Parameters
        ----------
        symbol : str
            Symbol to expand.

        number_of_derivs : int
            Number of derivations available.

        selected_prod : NonTerminalNode, default=None
            Production rule that might be selected beforehand. Then the method
            is called to compute cardinalities.

        Returns
        -------
        prod_rule : NonTerminalNode or None
            A production rule for the given symbol or None if this symbol
            cannot be expanded using exactly such number of derivations.
        """
        # Get the productions
        prod_rules = self.non_terminals_map[symbol] if selected_prod is None else [selected_prod]

        # Number of productions
        number_of_prod_pules = len(prod_rules)
        # Create productions roulette
        roulette = np.zeros(number_of_prod_pules)

        # Fill roulette
        for i in range(number_of_prod_pules):
            cardinalities = self.cardinality_map[prod_rules[i].production]

            # If this cardinality has not been calculated, calculate it
            if cardinalities[number_of_derivs - 1] == -1:
                cardinality = self._prod_rule_deriv_cardinality(
                    prod_rules[i].prod_list,
                    number_of_derivs - 1,
                )
                cardinalities[number_of_derivs - 1] = cardinality
                self.cardinality_map[prod_rules[i].production] = cardinalities

            # Apply logarithm to balance the roulette
            # (low values remain low and high values are lowered)
            # Useful when there is more than one branch to expand
            # and the number of available terminals is unbalanced
            # (e.g. one branch has 10 terminals and the other only has 2,
            # the first one will still have a higher probability of being selected,
            # but will have a lower chance of monopolizing the selection)
            roulette[i] = log1p(cardinalities[number_of_derivs - 1])

            if i != 0:
                roulette[i] += roulette[i - 1]

        # Choose a production at random
        rand_val = roulette[-1] * random.uniform(0, 1)

        return next((prod_rule for index, prod_rule in enumerate(prod_rules)
                     if rand_val < roulette[index]), None)

    def _select_partition(self, prod_rule: list[str], number_of_derivs: int) -> list[int]:
        """
        Select a partition to expand a symbol using a production rule.

        Parameters
        ----------
        prod_rule : list of str
            Production rule to expand.

        number_of_derivs : int
            Number of derivations available.

        Returns
        -------
        partition : list of int
            The partition selected.
        """
        min_derivs = tuple([self.min_derivations(symbol) for symbol in prod_rule])
        max_derivs = tuple([self.max_derivations(symbol) for symbol in prod_rule])
        partitions = self._partitions(number_of_derivs, len(prod_rule), min_derivs, max_derivs)
        # Number of partitions
        number_of_parts = len(partitions)
        # Create partitions roulette
        roulette = np.zeros(number_of_parts)

        # Set roulette values
        for i in range(number_of_parts):
            # Apply logarithm to balance the roulette
            # (low values remain low and high values are lowered)
            roulette[i] = log1p(self._prod_rule_part_cardinality(prod_rule, partitions[i]))

            if i != 0:
                roulette[i] = roulette[i] + roulette[i-1]

        # Choose a part at random
        rand_val = roulette[-1] * random.uniform(0, 1)

        return next(part for index, part in enumerate(partitions) if rand_val < roulette[index])

    def _symbol_cardinality(self, symbol: str, number_of_derivs: int) -> int:
        """
        Cardinality of a grammar symbol for the given number of derivations.

        Parameters
        ----------
        symbol : str
            The grammar symbol (terminal or non-terminal).

        number_of_derivs : int
            Number of derivations allowed.

        Returns
        -------
        cardinality : int
            Cardinality of the symbol.
        """
        if symbol in self.terminals_map:
            return 1 if number_of_derivs == 0 else 0

        if (symbol in self.cardinality_map and
                self.cardinality_map[symbol][number_of_derivs] != -1):
            return self.cardinality_map[symbol][number_of_derivs]

        result = sum(self._prod_rule_deriv_cardinality(prod_rule.prod_list, number_of_derivs - 1)
                     for prod_rule in self.non_terminals_map[symbol])

        if number_of_derivs >= 0 and symbol in self.cardinality_map:
            self.cardinality_map[symbol][number_of_derivs] = result

        return result

    def _prod_rule_deriv_cardinality(self, prod_rule: list[str], number_of_derivs: int) -> int:
        """
        Cardinality of a production rule for the given number of derivations.

        Parameters
        ----------
        prod_rule : list of str
            Production rule.

        number_of_derivs : int
            Number of derivations allowed.

        Returns
        -------
        cardinality : int
            Cardinality of the production rule.
        """
        min_derivs = tuple([self.min_derivations(symbol) for symbol in prod_rule])
        max_derivs = tuple([self.max_derivations(symbol) for symbol in prod_rule])
        partitions = self._partitions(number_of_derivs, len(prod_rule), min_derivs, max_derivs)
        return sum(self._prod_rule_part_cardinality(prod_rule, partition)
                   for partition in partitions)

    def _prod_rule_part_cardinality(self, prod_rule: list[str], partition: list[int]) -> int:
        """
        Cardinality of a production rule for the given partition.

        Parameters
        ----------
        prod_rule : list of str
            Production rule.

        partition : list of int
            The given partition.

        Returns
        -------
        cardinality : int
            Cardinality of the production rule for the partition.
        """
        result = 1

        for prod, part in zip(prod_rule, partition, strict=True):
            factor = self._symbol_cardinality(prod, part)

            if factor == 0:
                return 0

            result *= factor

        return result

    #@lru_cache(maxsize=256)
    def _partitions(
        self,
        total: int,
        dimension: int,
        min_derivs: tuple[int, ...],
        max_derivs: tuple[int, ...],
    ) -> list[list[int]]:
        """
        Compute the partitions for the production rule.

        Parameters
        ----------
        total : int
            Total number of derivations remaining.

        dimension : int
            Number of symbols in the production rule.

        min_derivs : tuple of int
            Minimum number of derivations for each symbol.

        max_derivs : tuple of int
            Maximum number of derivations for each symbol.

        Returns
        -------
        partitions : list of list of int
            All possible partitions for the production rule.
        """
        if (total, dimension, min_derivs, max_derivs) in self._partitions_cache:
            return self._partitions_cache[(total, dimension, min_derivs, max_derivs)]

        if dimension == 1:
            if min_derivs[0] <= total <= max_derivs[0]:
                return [[total]]

            return []

        result = []

        for i in range(min_derivs[0], min(total, max_derivs[0]) + 1):
            sub_partitions = self._partitions(
                total - i,
                dimension - 1,
                min_derivs[1:],
                max_derivs[1:],
            )

            result.extend([[i] + sub_partition for sub_partition in sub_partitions])

        self._partitions_cache[(total, dimension, min_derivs, max_derivs)] = result
        return result

    def _configure(self) -> None:
        """
        Configure the syntax tree schema.

        Configures the different dictionaries and sets the minimum
        derivation size given the set of terminals and non terminals.
        """
        self._build_terminals_dict()
        self._build_non_terminals_dict()
        self._build_cardinality_dict()
        self._calc_min_deriv_size()

    #@lru_cache(maxsize=100)
    def min_derivations(self, symbol: str) -> int:
        """
        Compute the minimum number of derivations of a given node.

        Parameters
        ----------
        symbol : str
            Symbol of the node.

        Returns
        -------
        min_deriv : int
            Minimum number of derivations.
        """
        if symbol in self._min_derivs_cache:
            return self._min_derivs_cache[symbol]

        def _min_derivations(symbol: str, visited: set[str]) -> int:
            if symbol in self.terminals_map:
                return 0 # Terminal nodes have no further derivations

            if symbol in visited:
                return self.max_deriv_size

            visited.add(symbol)
            min_deriv = self.max_deriv_size

            for prod_rule in self.non_terminals_map.get(symbol, []):
                deriv_count = sum(_min_derivations(sub_symbol, visited)
                                  for sub_symbol in prod_rule.prod_list)
                min_deriv = min(min_deriv, deriv_count + 1) # +1 for the current derivation step

            visited.remove(symbol)
            return min_deriv

        visited: set[str] = set()
        result = _min_derivations(symbol, visited)
        self._min_derivs_cache[symbol] = result
        return result

    #@lru_cache(maxsize=100)
    def max_derivations(self, symbol: str) -> int:
        """
        Compute the maximum number of derivations allowed for a given node, considering recursion.

        Parameters
        ----------
        symbol : str
            Symbol of the node.

        Returns
        -------
        max_deriv : int
            Maximum number of derivations allowed.
        """
        if symbol in self._max_derivs_cache:
            return self._max_derivs_cache[symbol]

        if symbol in self.terminals_map:
            return 0 # Terminal nodes have no further derivations

        if self.is_recursive(symbol):
            return self.max_deriv_size # If recursive, return the max derivation size limit

        def _max_derivations(symbol: str, visited: set[str]) -> int:
            visited.add(symbol)
            max_deriv = 0

            for prod_rule in self.non_terminals_map.get(symbol, []):
                deriv_count = sum(_max_derivations(sub_symbol, visited)
                                  for sub_symbol in prod_rule.prod_list)
                max_deriv = max(max_deriv, deriv_count + 1) # +1 for the current derivation step

            visited.remove(symbol)
            return max_deriv

        visited: set[str] = set()
        result = _max_derivations(symbol, visited)
        self._max_derivs_cache[symbol] = result
        return result

    #@lru_cache(maxsize=100)
    def is_recursive(self, symbol: str) -> bool:
        """
        Check if a symbol reaches a recursive production rule.

        Parameters
        ----------
        symbol : str
            Symbol to check for recursion.

        Returns
        -------
        recursive : bool
            True if the symbol reaches a recursive production rule, False otherwise.
        """
        if symbol in self._is_recursive_cache:
            return self._is_recursive_cache[symbol]

        def _is_recursive(symbol: str, visited: set[str]) -> bool:
            if symbol in visited:
                return True

            visited.add(symbol)

            for prod_rule in self.non_terminals_map.get(symbol, []):
                for sub_symbol in prod_rule.prod_list:
                    if _is_recursive(sub_symbol, visited):
                        return True

            visited.remove(symbol)
            return False

        visited: set[str] = set()
        result = _is_recursive(symbol, visited)
        self._is_recursive_cache[symbol] = result
        return result

    #@lru_cache(maxsize=100)
    def get_parent_symbols(self, symbol: str) -> list[str]:
        """
        Get the parent symbols of a given symbol.

        Parameters
        ----------
        symbol : str
            Symbol to get the parent symbols.

        Returns
        -------
        parents : list of str
            List of parent symbols.
        """
        if symbol in self._parent_cache:
            return self._parent_cache[symbol]

        parents = []

        for parent_symbol, prod_rules in self.non_terminals_map.items():
            for prod_rule in prod_rules:
                if symbol in prod_rule.prod_list:
                    parents.append(parent_symbol)
                    break

        self._parent_cache[symbol] = parents
        return parents

class SyntaxTreePipeline(SyntaxTree):
    """
    SyntaxTree that includes a machine learning pipeline to be executed.

    Parameters
    ----------
    content : list of TerminalNode and NonTerminalNode objects
        List of nodes to initialize the tree.
    """

    def __init__(self, content: list['Union[TerminalNode, NonTerminalNode]']):
        super().__init__(content)
        self.pipeline: 'Optional[Pipeline]' = None
        self._cached_str: str = ""

    def _get_term_str(
        self,
        terms: list['TerminalNode'],
        term_index: int,
        index_offset: int = 0,
    ) -> tuple[str, int, int]:
        """
        Get the string representation of a terminal node.

        Parameters
        ----------
        terms : list of TerminalNode
            List of terminal nodes.

        term_index : int
            Index of the terminal node.

        index_offser : int, default=0
            Term index offset in case of non terminal hyper-parameter

        Returns
        -------
        term_str : str
            String representation of the terminal node.

        term_index : int
            Updated index.

        index_offset : int
            Updated term index offset in case of non terminal hyper-parameter
        """
        start_index = term_index
        term = terms[start_index]

        if isinstance(term.code, Primitive):  # Algorithm
            term_str = term.symbol + "("
            term_index += 1

            while (term_index < len(terms) and
                   term_index - start_index < term.code.arity + index_offset):
                arg_str, term_index, index_offset = self._get_term_str(
                    terms,
                    term_index,
                    index_offset,
                )
                term_str += arg_str + ","

            return term_str.rstrip(',') + ")", term_index, term.code.arity

        # Hyper-parameter
        # Get the hyper-parameter value
        arg_val = term.code.name

        # Check if value is a string
        if not is_bool(arg_val) and not is_number(arg_val) and not is_tuple(arg_val):
            arg_val = "'" + arg_val + "'"

        return str(arg_val), term_index + 1, index_offset if index_offset != 0 else 0

    def __str__(self) -> str:
        """
        Return a string representation of the tree in a human-readable format.
        """
        if self._cached_str:
            return self._cached_str

        # Get the terminals (i.e. functions and its hyper-parameters)
        terms = [elem for elem in self if isinstance(elem, TerminalNode)]
        # Build the string
        pipe_str = ""
        term_index = 0

        while term_index < len(terms):
            term_str, term_index, _ = self._get_term_str(terms, term_index)
            pipe_str += term_str + ";"

        # Remove the last ";"
        self._cached_str = pipe_str.rstrip(";")
        return self._cached_str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SyntaxTreePipeline):
            raise ValueError(f"Cannot compare SyntaxTreePipeline with {type(other)}")

        return str(self) == str(other)

    def reset(self) -> None:
        """
        Reset the syntax tree pipeline.
        """
        self.pipeline = None
        self._cached_str = ""

    @overload
    def __setitem__(
        self,
        key: SupportsIndex,
        value: 'Union[TerminalNode, NonTerminalNode]',
    ) -> None: ...
    @overload
    def __setitem__(self, key: slice, value: 'SyntaxTree') -> None: ...

    def __setitem__(
        self,
        key: 'Union[SupportsIndex, slice]',
        value: 'Union[TerminalNode, NonTerminalNode, SyntaxTree]',
    ) -> None:
        """
        Set an item in the syntax tree pipeline and reset the cached string representation.
        """
        super().__setitem__(key, value)
        self._cached_str = ""

    def __delitem__(self, key: 'Union[SupportsIndex, slice]') -> None:
        """
        Delete an item from the syntax tree pipeline and reset the cached string representation.
        """
        super().__delitem__(key)
        self._cached_str = ""

    def append(self, object: 'Union[TerminalNode, NonTerminalNode]') -> None:
        """
        Append a node to the syntax tree pipeline and reset the cached string representation.
        """
        super().append(object)
        self._cached_str = ""

    def extend(self, iterable: 'list[Union[TerminalNode, NonTerminalNode]]') -> None:
        """
        Extend the syntax tree pipeline with new nodes and reset the cached string representation.
        """
        super().extend(iterable)
        self._cached_str = ""

    def create_pipeline(
        self,
        pset: 'PrimitiveSet',
        memory_limit: 'Optional[int]' = None,
    ) -> None:
        """
        Create a pipeline from the syntax tree content.

        Parameters
        ----------
        pset : PrimitiveSet
            The primitive set used to create the pipeline.

        memory_limit : int, default=None
            Maximum memory usage in MB. If `None`, memory usage is not limited.
        """
        def parse_step(step_str: str, context: dict[str, object]) -> object:
            step_name, step_args = step_str.split("(", 1)
            step_args = step_args.rstrip(")")
            step_class = context.get(step_name)

            if not step_class:
                raise ValueError(f"Unknown step: {step_name}")

            return step_class(*parse_arguments(step_args, context))

        def parse_arguments(args_str: str, context: dict[str, object]) -> tuple:
            args = []
            nested_level = 0
            current_arg = ""

            for char in args_str + ",":  # Add a trailing comma to simplify parsing
                if char == "(":
                    nested_level += 1
                    current_arg += char
                elif char == ")":
                    nested_level -= 1
                    current_arg += char
                elif char == "," and nested_level == 0:
                    current_arg = current_arg.strip()

                    if current_arg:  # Avoid empty strings
                        if "(" in current_arg and current_arg.split("(")[0] in context:
                            # Nested estimator
                            args.append(parse_step(current_arg, context))
                        else:
                            # Literal value
                            args.append(literal_eval(current_arg))

                    current_arg = ""
                else:
                    current_arg += char

            return tuple(args)

        # Transform the strings into the steps of the pipeline
        try:
            steps = []
            steps_str = str(self).replace("'None'", "None").split(";")

            for index, step_str in enumerate(steps_str):
                steps.append((str(index), parse_step(step_str, pset.context)))

            self.pipeline = Pipeline(steps, memory_limit=memory_limit)
        except Exception as e:
            self.pipeline = None
            raise ValueError(f"Error creating pipeline: {e}")
