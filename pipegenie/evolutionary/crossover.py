#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

"""
Crossover operators for Genetic Programming.
"""

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pipegenie.syntax_tree._encoding import NonTerminalNode

if TYPE_CHECKING:
    from typing import Optional, Union

    from pipegenie.syntax_tree._encoding import SyntaxTreeSchema
    from ._individual import Individual


class CrossoverBase(ABC):
    """
    Base class for crossover operators.

    Parameters
    ----------
    p_crossover : float, default=0.8
        The probability of applying the crossover operator.
    """

    def __init__(self, p_crossover: float = 0.8):
        self.p_crossover = p_crossover

    @abstractmethod
    def _apply_crossover(
        self,
        parent1: 'Individual',
        parent2: 'Individual',
        schema: 'SyntaxTreeSchema',
    ) -> 'tuple[Individual, Individual]':
        """
        Apply the crossover operator to the individuals.
        """
        raise NotImplementedError("Method '_apply_crossover' must be implemented in subclass")

    def cross(
        self,
        parent1: 'Individual',
        parent2: 'Individual',
        schema: 'SyntaxTreeSchema',
    ) -> 'tuple[Individual, Individual, bool]':
        """
        Perform crossover between two parents.

        Parameters
        ----------
        parent1 : Individual
            The first parent.

        parent2 : Individual
            The second parent.

        schema : SyntaxTreeSchema
            The schema used to generate the individuals.

        Returns
        -------
        son1 : Individual
            The first son.

        son2 : Individual
            The second son.

        changed : bool
            True if at least one of the sons is different from the parents, False otherwise.
        """
        if random.random() < self.p_crossover:
            par1_str = str(parent1)
            par2_str = str(parent2)

            parent1, parent2 = self._apply_crossover(parent1, parent2, schema)

            # Ensure sons are different
            son1_str = str(parent1)
            son2_str = str(parent2)

            return parent1, parent2, son1_str != par1_str or son2_str != par2_str

        return parent1, parent2, False

class BranchSwapperMixin:
    """
    Mixin class for swapping branches in syntax trees.
    """

    def _swap_branches(
        self,
        schema: 'SyntaxTreeSchema',
        ind1: 'Individual',
        ind2: 'Individual',
        start1: int,
        start2: int,
    ) -> 'tuple[Individual, Individual]':
        """
        Perform the swap of branches in the individuals.
        """
        # Get the slice to swap
        slice1 = ind1.search_subtree(start1)
        slice2 = ind2.search_subtree(start2)

        # Get branch depth (to check maximum size)
        p0_non_terms = sum(1 for node in ind1 if isinstance(node, NonTerminalNode))
        p0_swap_branch = sum([1 for node in ind1[slice1] if isinstance(node, NonTerminalNode)])
        p1_non_terms = sum(1 for node in ind2 if isinstance(node, NonTerminalNode))
        p1_swap_branch = sum([1 for node in ind2[slice2] if isinstance(node, NonTerminalNode)])

        # Check maximum number of derivation conditions
        max_derivations = schema.max_deriv_size
        cond0 = (p0_non_terms - p0_swap_branch + p1_swap_branch > max_derivations)
        cond1 = (p1_non_terms - p1_swap_branch + p0_swap_branch > max_derivations)

        if cond0 or cond1:
            return ind1, ind2

        # Swap and return the individuals
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
        return ind1, ind2

class MultiCrossover(CrossoverBase, BranchSwapperMixin):
    """
    Multi-crossover operator for syntax trees.

    The operator checks the common hyper-parameters between the individuals.
    If the individuals have more than one common hyper-parameter,
    the crossover is performed on the hyper-parameters.
    Otherwise, the crossover is performed on the structure of the individuals.
    """

    def _select_symbol_incl(
        self,
        ind1: 'Individual',
        ind2: 'Individual',
        incl: 'Optional[list[str]]' = None,
    ) -> 'Union[tuple[int, int], tuple[None, None]]':
        """
        Select a common non-terminal node in the individuals.
        """
        # Individual length
        ind1_len = len(ind1)

        # Generate a tree position at random
        start_pos = random.randint(0, ind1_len)
        act_pos = start_pos

        for index1 in range(ind1_len):
            # Update the current position
            act_pos = (start_pos + index1) % ind1_len
            # Get the node
            node1 = ind1[act_pos]
            # Check symbol is an included one
            if incl is not None and node1.symbol in incl:
                # Find the ocurrences of the symbol in the other individual
                indexes2 = [index2 for index2, node2 in enumerate(ind2)
                            if node1.symbol == node2.symbol]
                # Return a coincidence at random
                if len(indexes2) != 0:
                    return act_pos, random.choice(indexes2)

        # There are no occurrences
        return None, None

    def _cx_struct(
        self,
        ind1: 'Individual',
        ind2: 'Individual',
        schema: 'SyntaxTreeSchema',
    ) -> 'tuple[Individual, Individual]':
        """
        Crossover the structure of the individuals.
        """
        # get the common nodes (symbols)
        symbols_ind1 = [node.symbol for node in ind1 if isinstance(node, NonTerminalNode)]
        symbols_ind2 = [node.symbol for node in ind2 if isinstance(node, NonTerminalNode)]
        incl = sorted(set(symbols_ind1) & set(symbols_ind2))
        # Select a common non terminal node
        start1, start2 = self._select_symbol_incl(ind1, ind2, incl)

        if start1 is None or start2 is None:
            return ind1, ind2

        return self._swap_branches(schema, ind1, ind2, start1, start2)

    def _swap_hp(
        self,
        ind1: 'Individual',
        ind2: 'Individual',
        hp_name: str,
    ) -> 'tuple[Individual, Individual]':
        """
        Perform the swap of the hyper-parameter in the individuals.
        """
        idx1 = next(idx for idx, node in enumerate(ind1) if node.symbol == hp_name)
        idx2 = next(idx for idx, node in enumerate(ind2) if node.symbol == hp_name)
        ind1[idx1], ind2[idx2] = ind2[idx2], ind1[idx1]
        return ind1, ind2

    def _cx_hps(
        self,
        ind1: 'Individual',
        ind2: 'Individual',
        common_hps: list[str],
    ) -> 'tuple[Individual, Individual]':
        """
        Crossover the hyper-parameters in the individuals.
        """
        # randomly select a crossover point
        # TODO: maybe uniform crossover instead of 1-px crossover? because of alphabetical sorting
        cxpoint = random.randint(1, len(common_hps) - 1)

        # swap the hyper-parameters
        for idx, hp_name in enumerate(common_hps):
            if idx < cxpoint:
                ind1, ind2 = self._swap_hp(ind1, ind2, hp_name)

        return ind1, ind2

    def _apply_crossover(
        self,
        parent1: 'Individual',
        parent2: 'Individual',
        schema: 'SyntaxTreeSchema',
    ) -> 'tuple[Individual, Individual]':
        """
        Apply the crossover operator to the individuals.
        """
        # Get the common hyper-parameters
        hps1 = [node.symbol for node in parent1 if "::" in node.symbol]
        hps2 = [node.symbol for node in parent2 if "::" in node.symbol]
        common_hps = sorted(set(hps1) & set(hps2))

        # Apply one of the crossover operators
        if len(common_hps) > 1: # TODO: why > 1? should be >= 1?
            parent1, parent2 = self._cx_hps(parent1, parent2, common_hps)
        else:
            parent1, parent2 = self._cx_struct(parent1, parent2, schema)

        return parent1, parent2

    def __str__(self) -> str:
        return f"MultiCrossover(p_crossover={self.p_crossover})"

class BranchCrossover(CrossoverBase, BranchSwapperMixin):
    """
    Branch-crossover operator for syntax trees.

    The operator performs the crossover on the structure of the individuals.
    """

    def _select_symbol_excl(
        self,
        ind1: 'Individual',
        ind2: 'Individual',
        excl: 'Optional[list[str]]' = None,
    ) -> 'Union[tuple[int, int], tuple[None, None]]':
        """
        Select a common non-terminal node in the individuals.
        """
        # Individual length
        ind1_len = len(ind1)

        # Generate a tree position at random
        start_pos = random.randint(0, ind1_len)
        act_pos = start_pos

        for index1 in range(ind1_len):
            # Update the current position
            act_pos = (start_pos + index1) % ind1_len
            # Get the node
            node1 = ind1[act_pos]
            # Check symbol is non termianl
            if (isinstance(node1, NonTerminalNode) != 0 and
                    (excl is None or node1.symbol not in excl)):
                # Find the ocurrences of the symbol in the other individual
                indexes2 = [index2 for index2, node2 in enumerate(ind2)
                            if node1.symbol == node2.symbol]
                # Return a coincidence at random
                if len(indexes2) != 0:
                    return act_pos, random.choice(indexes2)

        # There are no occurrences
        return None, None

    def _apply_crossover(
        self,
        parent1: 'Individual',
        parent2: 'Individual',
        schema: 'SyntaxTreeSchema',
    ) -> 'tuple[Individual, Individual]':
        """
        Apply the crossover operator to the individuals.
        """
        # Select a common non terminal node (excluding the root)
        start1, start2 = self._select_symbol_excl(
            parent1,
            parent2,
            [schema.root],
        )

        if start1 is None or start2 is None:
            return parent1, parent2

        # Apply the crossover
        parent1, parent2 = self._swap_branches(schema, parent1, parent2, start1, start2)
        return parent1, parent2

    def __str__(self) -> str:
        return f"BranchCrossover(p_crossover={self.p_crossover})"
