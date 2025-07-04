#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

"""
Mutations operators for Genetic Programming.
"""

import random
from abc import ABC, abstractmethod
from inspect import isclass
from typing import TYPE_CHECKING

from pipegenie.syntax_tree._encoding import NonTerminalNode, TerminalNode

if TYPE_CHECKING:
    from pipegenie.syntax_tree._encoding import SyntaxTree, SyntaxTreeSchema
    from ._individual import Individual


class MutationBase(ABC):
    """
    Base class for mutation operators.

    Parameters
    ----------
    p_mutation : float, default=0.2
        The probability of mutation.
    """

    def __init__(self, p_mutation: float = 0.2):
        self.p_mutation = p_mutation

    @abstractmethod
    def _apply_mutation(
        self,
        ind: 'Individual',
        schema: 'SyntaxTreeSchema',
    ) -> 'Individual':
        """
        Apply mutation operator to the individual.
        """
        raise NotImplementedError("Method '_apply_mutation' must be implemented in subclass")

    def mutate(
        self,
        ind: 'Individual',
        schema: 'SyntaxTreeSchema',
    ) -> 'tuple[Individual, bool]':
        """
        Perform mutation on an individual.

        Parameters
        ----------
        ind : Individual
            The individual to mutate.

        schema : SyntaxTreeSchema
            The schema used to generate the individuals.

        Returns
        -------
        son : Individual
            The mutated individual.

        changed : bool
            True if the individual was changed, False otherwise.
        """
        if random.random() < self.p_mutation:
            ind_str = str(ind)
            ind = self._apply_mutation(ind, schema)
            return ind, str(ind) != ind_str

        return ind, False

class MultiMutation(MutationBase):
    """
    Multi-mutation operator for syntax trees.

    The operator, randomly, selects between mutation of hyper-parameters or structure.

    Parameters
    ----------
    p_mutation : float, default=0.2
        The probability of mutation.

    p_hyper : float, default=0.5
        The probability of mutating hyper-parameters.
    """

    def __init__(
        self,
        p_mutation: float = 0.2,
        p_hyper: float = 0.5,
    ):
        super().__init__(p_mutation)
        self.p_hyper = p_hyper
        self.hyper_mutation = HyperparameterMutation(1)
        self.branch_mutation = BranchMutation(1)

    def _apply_mutation(
        self,
        ind: 'Individual',
        schema: 'SyntaxTreeSchema',
    ) -> 'Individual':
        """
        Apply mutation operator to the individual.
        """
        if random.random() < self.p_hyper:
            return self.hyper_mutation._apply_mutation(ind, schema)

        return self.branch_mutation._apply_mutation(ind, schema)

    def __str__(self) -> str:
        return f"MultiMutation(p_mutation={self.p_mutation}, p_hyper={self.p_hyper})"

class BranchMutation(MutationBase):
    """
    Branch mutation operator for syntax trees.

    The operator mutates the structure of the individual
    by replacing a branch with a newly generated one.
    """

    def _rebuild_branch(
        self,
        ind: 'Individual',
        start: int,
        schema: 'SyntaxTreeSchema',
    ) -> 'Individual':
        tree_slice = ind.search_subtree(start)

        # Get branch depth (to check maximum size)
        p0_branch_depth = sum(1 for node in ind if isinstance(node, NonTerminalNode))
        p0_swap_branch = sum(1 for node in ind[tree_slice] if isinstance(node, NonTerminalNode))

        # Get the symbol
        symbol = ind[start].symbol

        # Save the fragment at the right of the subtree
        aux = ind[tree_slice.stop:]

        # Remove the subtree and the fragment at its right
        del ind[start:]

        # Create the son (second fragment) controlling the number of derivations
        max_derivations = schema.max_deriv_size - p0_branch_depth + p0_swap_branch
        min_derivations = schema.min_derivations(symbol)
        derivations = random.randint(min_derivations, max_derivations)
        schema.fill_tree_branch(ind, symbol, derivations)

        # Restore the fragment at the right of the subtree
        ind.extend(aux)
        return ind

    def _apply_mutation(
        self,
        ind: 'Individual',
        schema: 'SyntaxTreeSchema',
    ) -> 'Individual':
        """
        Apply mutation operator to the individual.
        """
        target_symbols = schema.non_terminals_map.keys()
        possible_starts = [index for index, node in enumerate(ind)
                           if node.symbol in target_symbols]
        start = random.choice(possible_starts)
        return self._rebuild_branch(ind, start, schema)

    def __str__(self) -> str:
        return f"BranchMutation(p_mutation={self.p_mutation})"

class HyperparameterMutation(MutationBase):
    """
    Hyper-parameter mutation operator for syntax trees.

    The operator mutates the hyper-parameters of the individual.
    The probability of mutating a preprocessing hyper-parameter is inversely proportional
    to the number of hyper-parameters in the preprocessing. The same applies to the estimator.
    """

    def _apply_mutation(
        self,
        ind: 'Individual',
        schema: 'SyntaxTreeSchema',
    ) -> 'Individual':
        """
        Apply mutation operator to the individual.
        """
        # get the start position of the estimator
        #target_symbols = ['classifier', 'regressor']
        last_alg_idx = next(idx for idx, node in enumerate(reversed(ind))
                             # Check if it is an algorithm
                             if isinstance(node, TerminalNode) and
                             "::" not in node.symbol and
                             # Ensure there is a next node
                             idx + 1 < len(ind) and
                             # Estimator hyperparameter special case
                             # -1 because we are iterating in reverse
                             # and another -1 because we are looking for the next node
                             "::" not in ind[len(ind) - idx - 2].symbol)
        # get parent symbol
        parent_symbols = schema.get_parent_symbols(ind[len(ind) - last_alg_idx - 1].symbol)
        parent_idx = next(idx for idx, node in enumerate(reversed(ind[:len(ind) - last_alg_idx]))
                          if node.symbol in parent_symbols)

        # get the target symbol of the operator and its position
        pos_estimator = len(ind) - last_alg_idx - parent_idx - 1  # -1 because of reversed index
        target_symbol = ind[pos_estimator].symbol

        # get the number of hyperparameters for the preprocessing and the estimator
        num_prep_hps = sum([1 for node in ind[:pos_estimator] if "::" in node.symbol])
        num_est_hps = sum([1 for node in ind[pos_estimator:] if "::" in node.symbol])

        # compute the probability of changing one hyperparameter
        mutpb_est = 1.0 / num_est_hps if num_est_hps > 0 else 0.0
        mutpb_prep = 1.0 / num_prep_hps if num_prep_hps > 0 else 0.0

        # mutate each hyperparameter with its corresponding probability
        for idx, node in enumerate(ind):
            if "::" in node.symbol:
                mutpb = mutpb_prep if idx < pos_estimator else mutpb_est

                if random.random() < mutpb:
                    if node.symbol in schema.terminals_map:
                        term = schema.terminals_map[node.symbol]
                        ind[idx] = TerminalNode(node.symbol, term.code()
                                                if isclass(term.code) else term.code)
                    else:
                        # estimator hyperparameter special case
                        min_derivations = schema.min_derivations(node.symbol)
                        max_derivations = schema.max_derivations(node.symbol)
                        derivations = random.randint(min_derivations, max_derivations)
                        new_node: 'SyntaxTree' = []
                        schema.fill_tree_branch(new_node, node.symbol, derivations)
                        tree_slice = ind.search_subtree(idx)
                        ind[idx:tree_slice.stop] = new_node

                        # update mutpb if the estimator hyperparameter was mutated
                        if idx >= pos_estimator:
                            num_est_hps = sum([1 for node in ind[pos_estimator:]
                                               if "::" in node.symbol])
                            mutpb_est = 1.0 / num_est_hps if num_est_hps > 0 else 0.0
                        else:
                            # if the hyperparameter was in the preprocessing part,
                            # the estimator position may have changed
                            pos_estimator = next(idx for idx, node in enumerate(ind)
                                                 if node.symbol == target_symbol)
                            num_prep_hps = sum([1 for node in ind[:pos_estimator]
                                                if "::" in node.symbol])
                            mutpb_prep = 1.0 / num_prep_hps if num_prep_hps > 0 else 0.0

        return ind

    def __str__(self) -> str:
        return f"HyperparameterMutation(p_mutation={self.p_mutation})"

class NodeMutation(MutationBase):
    """
    Node mutation operator for syntax trees.

    The operator mutates a node of the individual, either an algorithm or a hyper-parameter.

    Parameters
    ----------
    p_mutation : float, default=0.2
        The probability of mutation.

    p_hyper : float, default=0.5
        The probability of mutating hyper-parameters.
    """

    # target_symbol + algorithm + hyperparameters_root + hyperparameters is, at least, 4 nodes
    MIN_SLICE_SIZE_HYPERPARAMETERS = 4

    # target_symbol + algorithm + hyperparameters_root are 3 nodes
    HYPERPARAMETERS_OFFSET = 3

    def __init__(
        self,
        p_mutation: float = 0.2,
        p_hyper: float = 0.5,
    ):
        super().__init__(p_mutation)
        self.p_hyper = p_hyper

    def _mutate_algorithm(
        self,
        ind: 'Individual',
        schema: 'SyntaxTreeSchema',
        tree_slice: slice,
    ) -> 'Individual':
        new_node: 'SyntaxTree' = []
        # get the minimum and maximum number of derivations allowed for the symbol
        min_derivations = schema.min_derivations(ind[tree_slice.start].symbol)
        max_derivations = schema.max_derivations(ind[tree_slice.start].symbol)
        # Randomly select the number of derivations
        derivations = random.randint(min_derivations, max_derivations)
        schema.fill_tree_branch(new_node, ind[tree_slice.start].symbol, derivations)
        ind[tree_slice.start:tree_slice.stop] = new_node
        return ind

    def _mutate_hyperparameter(self,
        ind: 'Individual',
        schema: 'SyntaxTreeSchema',
        tree_slice: slice,
    ) -> 'Individual':
        hyper_index_start = tree_slice.start + self.HYPERPARAMETERS_OFFSET
        selected_index = random.randint(hyper_index_start, tree_slice.stop - 1)
        term = schema.terminals_map[ind[selected_index].symbol]
        ind[selected_index] = TerminalNode(ind[selected_index].symbol, term.code()
                                           if isclass(term.code) else term.code)
        return ind

    def _apply_mutation(
        self,
        ind: 'Individual',
        schema: 'SyntaxTreeSchema',
    ) -> 'Individual':
        """
        Apply mutation operator to the individual.
        """
        parent_symbols = [schema.get_parent_symbols(node.symbol) for node in ind
                              if isinstance(node, TerminalNode) and "::" not in node.symbol]
        target_symbols = set([symbol for symbols in parent_symbols for symbol in symbols])
        possible_indexes = [index for index, node in enumerate(ind)
                            if node.symbol in target_symbols]
        selected_index = random.choice(possible_indexes)
        tree_slice = ind.search_subtree(selected_index)
        has_hyperparameters = len(ind[tree_slice]) >= self.MIN_SLICE_SIZE_HYPERPARAMETERS

        if has_hyperparameters and random.random() < self.p_hyper:
            return self._mutate_hyperparameter(ind, schema, tree_slice)

        return self._mutate_algorithm(ind, schema, tree_slice)

    def __str__(self) -> str:
        return f"NodeMutation(p_mutation={self.p_mutation}, p_hyper={self.p_hyper})"
