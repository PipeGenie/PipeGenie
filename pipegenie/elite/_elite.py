#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

from bisect import bisect_right
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from pipegenie.evolutionary._individual import Individual


class DiverseElite:
    """
    Class responsible for maintaining the elite of the population.

    The elite is a subset of the population that is considered the best
    individuals. The elite is maintained based on the fitness and diversity
    of the individuals, so it is composed of diverse individuals. The elite
    is sorted based on the weighted sum of the diversity and the fitness of
    the individual.

    Parameters
    ----------
    max_size : int, default=1
        The maximum size of the elite.

    div_weight : float, default=0.2
        How much weight to give to the diversity of the individual.
        0 means that only the fitness is considered while 1 means
        that only the diversity is considered.
        If the elite size is 1, the diversity is not considered.
    """

    def __init__(self, max_size: int = 1, div_weight: float = 0.2):
        if max_size < 1:
            raise ValueError("The maximum size of the elite must be at least 1")

        self.max_size = max_size
        self.div_weight = div_weight if max_size > 1 else 0.0
        self.elite: list['Individual'] = []
        self.diversity: list[float] = []

    def add(self, individual: 'Individual') -> None:
        """
        Add an individual to the elite.

        Parameters
        ----------
        individual : Individual
            The individual to add to the elite.
        """
        ind = individual.clone()
        index = bisect_right(self.diversity, ind.diversity)
        self.elite.insert(len(self.elite) - index, ind)
        self.diversity.insert(index, ind.diversity)

    def update(self, population: 'Sequence[Individual]') -> None:
        """
        Update the elite with the population.

        Parameters
        ----------
        population : list of Individual
            The population to update the elite.
        """
        if len(self.elite) == 0:
            # If the elite is empty, we cannot calculate the diversity of the individuals
            for ind in population:
                ind.diversity = ind.fitness.weighted_value

            self.add(population[0])

            for ind in population[1:]:
                self._try_add_to_elite(ind)

            self.update_diversities()
            self.sort()
        else:
            for ind in population:
                if not ind.fitness.valid:
                    continue

                div = self.compute_diversity(ind)
                ind.diversity = (self.div_weight * div
                                 + (1 - self.div_weight) * ind.fitness.weighted_value)
                self._try_add_to_elite(ind)

    def _try_add_to_elite(self, ind: 'Individual') -> None:
        if ind.diversity > self.elite[-1].diversity or len(self.elite) < self.max_size:
            for member in self.elite:
                if ind == member:
                    # If the individual is already in the elite, we do not add it again
                    return

            if len(self.elite) >= self.max_size:
                self.remove(-1)

            self.add(ind)

    def sort(self) -> None:
        """
        Sort the elite.

        The elite is sorted based on the weighted sum of the diversity and the fitness of
        the individual.
        """
        elite = [ind.clone() for ind in self.elite]
        self.clear()
        self.add(elite[0])

        for individual in elite[1:]:
            self.add(individual)

    def compute_diversity(self, ind: 'Individual') -> float:
        """
        Compute the diversity of the individual against the elite.

        Parameters
        ----------
        ind : Individual
            The individual to compute the diversity.

        Returns
        -------
        diversity : float
            The diversity of the individual.
        """
        diversity = 0.0

        for member in self.elite:
            diff = np.sum(ind.prediction != member.prediction)
            diff /= len(ind.prediction)
            diversity += diff

        return diversity / len(self.elite)

    def update_diversities(self) -> None:
        """
        Compute the diversity of the individuals in the elite.
        """
        for ind in self.elite:
            div = self.compute_diversity(ind)
            ind.diversity = (self.div_weight * div
                             + (1 - self.div_weight) * ind.fitness.weighted_value)

    def best_ind(self) -> 'Individual':
        """
        Return the best individual in the elite based on its fitness.

        Returns
        -------
        best : Individual
            The best individual in the elite.
        """
        best = self.elite[0]
        best_fitness = best.fitness.weighted_value

        for individual in self.elite[1:]:
            if individual.fitness.weighted_value > best_fitness:
                best = individual
                best_fitness = individual.fitness.weighted_value

        return best.clone()

    def remove(self, index: int) -> None:
        """
        Remove an individual from the elite.

        Parameters
        ----------
        index : int
            The index of the individual to remove.
        """
        self.elite.pop(index)
        self.diversity.pop(len(self.diversity) - (index % len(self.diversity)) - 1)

    def clear(self) -> None:
        """
        Clear the elite.
        """
        self.elite.clear()
        self.diversity.clear()

    def __len__(self) -> int:
        return len(self.elite)

    def __iter__(self) -> 'Iterator[Individual]':
        return iter(self.elite)

    def __getitem__(self, index: int) -> 'Individual':
        return self.elite[index]

    def __repr__(self) -> str:
        return (
            f"DiverseElite(max_size={self.max_size}, "
            f"div_weight={self.div_weight}, "
            f"elite={self.elite})"
        )

    def __str__(self) -> str:
        return "\n".join([str(ind) for ind in self.elite])
