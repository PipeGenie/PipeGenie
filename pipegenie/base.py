#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

"""
Base class for all pipegenie estimators.
"""

import logging
import logging.handlers
import pickle
import random
import sys
import warnings
from abc import ABC, abstractmethod
from functools import partial
from math import ceil
from multiprocessing import Manager, cpu_count
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

import numpy as np
from loky import ProcessPoolExecutor

from pipegenie.elite._elite import DiverseElite
from pipegenie.evolutionary._individual import Fitness, Individual
from pipegenie.grammar import parse_pipe_grammar
from pipegenie.logging._logging import Logbook
from pipegenie.logging._stats import MultiStatistics, Statistics
from pipegenie.syntax_tree._encoding import SyntaxTreeSchema

if TYPE_CHECKING:
    from collections.abc import Callable
    from queue import Queue
    from typing import Optional

    from numpy.typing import ArrayLike

    from pipegenie.evolutionary.crossover import CrossoverBase
    from pipegenie.evolutionary.mutation import MutationBase
    from pipegenie.evolutionary.replacement import ReplacementBase
    from pipegenie.evolutionary.selection import SelectionBase
    from pipegenie.model_selection import BaseCrossValidator
    from pipegenie.pipeline import Pipeline
    from pipegenie.voting import BaseVoting

# Ignore all warnings
warnings.filterwarnings("ignore")

# Show warnings from the "pipegenie" package
warnings.filterwarnings("default", module="^pipegenie\.")


class BasePipegenie(ABC):
    r"""
    Base class for all pipegenie estimators.

    All estimators should specify all the parameters that can be set
    at the class level in their __init__ as explicit keyword arguments
    (no \*args or \*\*kwargs).

    This is a minimal interface that all estimators should implement.

    Parameters
    ----------
    grammar : str
        Path of the file containing the grammar used to generate the pipelines.

    grammar_type : str
        Format used to define the grammar.
        Supported formats: "evoflow-xml".
        You can add new formats by creating a new parser and registering it in the factory.

    pop_size : int
        Size of the population used in the evolutionary process.
        Must be greater than 0.

    generations : int
        Number of generations used in the evolutionary process.
        Must be greater or equal than 0.
        If 0, only the initial population will be evaluated.

    fitness : function
        Fitness function used to evaluate the pipelines.

    nderiv : int
        Number of derivations used to generate the pipelines.
        Must be greater than 0.

    selection : SelectionBase instance
        Selection method used to select the individuals that
        will be used to generate the offspring.

    crossover : CrossoverBase instance
        Crossover function used to generate new individuals.

    mutation : MutationBase instance
        Mutation function used to generate new individuals.

    mutation_elite : MutationBase instance
        Mutation function used to generate new individuals from the elite.

    replacement : ReplacementBase instance
        Replacement function used to create the new population.

    use_double_mutation : bool
        Indicates if the mutation_elite function should be used to generate new individuals
        from the elite instead of the mutation function.
        No crossover will be applied.
        If False, the crossover function will be applied and then the mutation function.

    elite_size : int
        Maximun number of pipelines to be stored in the elite.
        must be in the range [1, pop_size].

    timeout : int
        Maximun time allowed for the evolutionary process.
        Must be greater than 0.

    eval_timeout : int or None
        Maximun time allowed for the evaluation of a pipeline.
        It has to be greater than 0 or None.
        If None, the evaluation of a pipeline will not have a time limit.

    cross_validator : BaseCrossValidator instance
        Cross-validator used to evaluate the pipelines.

    maximization : bool
        Indicates if the fitness function should be maximized or minimized.

    early_stopping_threshold : float
        Threshold used to determine if there is improvement in the elite average fitness.
        Must be greater or equal than 0.0.

    early_stopping_generations : int or None
        Number of generations without improvement in the elite average fitness
        before stopping the evolutionary process.
        It has to be greater than 0 or None.
        If None, the evolutionary process will not stop based on generations without improvement.

    early_stopping_time : int or None
        Maximun time without improvement in the elite average fitness
        before stopping the evolutionary process.
        It has to be greater than 0 or None.
        If None, the evolutionary process will not stop based on time without improvement.

    seed : int or None
        Seed used to initialize the random number generator.
        If None, a random seed will be used.

    outdir : str
        Path of the directory where the results will be stored.

    n_jobs : int
        Number of processes used to evaluate the population.
        Must be greater than 0.
        If -1, the number of processes will be equal to the number of cores of the machine.
    """

    def __init__(
        self,
        grammar: str,
        *,
        grammar_type: str,
        pop_size: int,
        generations: int,
        fitness: 'Callable[..., object]',
        nderiv: int,
        selection: 'SelectionBase',
        crossover: 'CrossoverBase',
        mutation: 'MutationBase',
        mutation_elite: 'MutationBase',
        replacement: 'ReplacementBase',
        use_double_mutation: bool,
        elite_size: int,
        timeout: int,
        eval_timeout: 'Optional[int]',
        cross_validator: 'BaseCrossValidator',
        maximization: bool,
        early_stopping_threshold: float,
        early_stopping_generations: 'Optional[int]',
        early_stopping_time: 'Optional[int]',
        seed: 'Optional[int]',
        outdir: str,
        n_jobs: int,
        **kwargs: object,
    ):
        self.grammar = grammar
        self.grammar_type = grammar_type
        self.pop_size = pop_size
        self.generations = generations
        self.fitness = fitness
        self.nderiv = nderiv
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.mutation_elite = mutation_elite
        self.replacement = replacement
        self.use_double_mutation = use_double_mutation
        self.elite_size = elite_size
        self.timeout = timeout
        self.eval_timeout = eval_timeout
        self.cross_validator = cross_validator
        self.maximization = maximization
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_generations = early_stopping_generations
        self.early_stopping_time = early_stopping_time
        self.outdir = outdir
        self.n_jobs = n_jobs

        self.seed = seed if seed is not None else random.randint(0, 2**32)
        random.seed(self.seed)

        # Ensure that the cross-validator is initialized with the same seed
        self.cross_validator.set_random_state(random_state=self.seed)

        arguments = vars(self).copy()
        arguments.update(kwargs)

        self.outdir_path = Path(self.outdir)
        self.outdir_path.mkdir(parents=True, exist_ok=True)

        self.cpu_count = cpu_count() if self.n_jobs == -1 else self.n_jobs

        # log the configuration
        with self.outdir_path.joinpath("config.txt").open("w", encoding="utf-8") as log:
            for key, value in arguments.items():
                if hasattr(value, '__name__'):
                    log.write(f"{key}: {value.__name__}\n")
                else:
                    log.write(f"{key}: {value}\n")

        self._create_loggers()
        self._init_statistics()

        root, terms, non_terms, self.pset, _ = parse_pipe_grammar(
            grammar,
            grammar_type,
            self.seed,
        )
        self.schema = SyntaxTreeSchema(root, nderiv, terms, non_terms)

    def _create_loggers(self) -> None:
        """
        Create the loggers used to store the results of the evolutionary process.
        """
        # create a logger for individual evaluation
        self.individuals_logger = logging.getLogger("pipegenie_individuals")
        self.individuals_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.outdir_path.joinpath("individuals.tsv"), mode="w")
        self.individuals_logger.addHandler(handler)
        self.individuals_logger.info("pipeline\tfitness\tfit_time")

        # create a logger for the evolutionary process
        self.evolution_logger = logging.getLogger("pipegenie_evolution")
        self.evolution_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.outdir_path.joinpath("evolution.txt"), mode="w")
        self.evolution_logger.addHandler(handler)
        console = logging.StreamHandler(sys.stdout)
        self.evolution_logger.addHandler(console)

    def _create_worker_logger(self, queue: 'Queue[object]') -> logging.Logger:
        """
        Create a logger used to store the logs generated by the worker processes.

        Parameters
        ----------
        queue : Queue
            Queue used to store the logs generated by the worker processes.

        Returns
        -------
        worker_logger : Logger
            Logger used to store the logs generated by the worker processes.
        """
        worker_logger = logging.getLogger("pipegenie_worker")

        if not worker_logger.hasHandlers():
            handler = logging.handlers.QueueHandler(queue)
            worker_logger.addHandler(handler)

        worker_logger.setLevel(logging.INFO)
        return worker_logger

    def _close_loggers(self) -> None:
        """
        Close the loggers used to store the results of the evolutionary process.
        """
        for handler in self.evolution_logger.handlers[:]:
            handler.close()
            self.evolution_logger.removeHandler(handler)

        for handler in self.individuals_logger.handlers[:]:
            handler.close()
            self.individuals_logger.removeHandler(handler)

    def _init_statistics(self) -> None:
        """
        Initialize the statistics used to evaluate the population and the elite.
        """
        stat_fit = Statistics(key=lambda ind: ind.fitness.values)
        # use string representation of the individual to calculate
        # the size because the pipeline object may not be valid
        stat_size = Statistics(key=lambda ind: len(str(ind).split(";")))
        stats = MultiStatistics(fitness=stat_fit, size=stat_size)
        stats.register('min', np.nanmin)
        stats.register('max', np.nanmax)
        stats.register('avg', np.nanmean)
        stats.register('std', np.nanstd)
        self.stats = stats

        stat_fit_elite = Statistics(key=lambda ind: ind.fitness.values)
        stat_size_elite = Statistics(key=lambda ind: len(str(ind).split(";")))
        stats_elite = MultiStatistics(fitness_elite=stat_fit_elite, size_elite=stat_size_elite)
        stats_elite.register('min', np.nanmin)
        stats_elite.register('max', np.nanmax)
        stats_elite.register('avg', np.nanmean)
        stats_elite.register('std', np.nanstd)
        self.stats_elite = stats_elite

    def _fit(self, data: 'dict[str, ArrayLike]') -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        data : dict
            Dictionary containing the training input samples and the target values.
        """
        self.elite_avg_fitness = np.nan
        self.generations_without_improvement = 0
        self.last_improvement_time = time()
        self.elite = self._create_elite_object()

        start_time = time()
        self._evolve(data)
        exec_time = str(time() - start_time)

        self.evolution_logger.info("--- " + exec_time + " sec ---")

        if self.elite is None or len(self.elite) == 0:
            raise RuntimeError("The elite is empty when it should have, at least, one member")

        for ind in self.elite:
            if not hasattr(ind, 'pipeline') or ind.pipeline is None:
                ind.create_pipeline(self.pset)

        self._generate_ensemble(data)
        self._close_loggers()

    @abstractmethod
    def _evaluate(
        self,
        ind: 'Individual',
        data: 'dict[str, ArrayLike]',
        start: float,
        queue: 'Queue',
    ) -> tuple:
        """
        Evaluate an individual.

        Parameters
        ----------
        ind : Individual
            Individual to be evaluated.

        data : dict
            Dictionary containing the training input samples and the target values.

        start : float
            Time when the evaluation started.

        queue : Queue
            Queue used to store the logs generated by the worker processes.

        Returns
        -------
        result : tuple
            Tuple containing the fitness values, the prediction and the runtime of the individual.
        """
        raise NotImplementedError("Method '_evaluate' must be implemented in subclass")

    def _evolve(self, data: 'dict[str, ArrayLike]') -> None:
        """
        Evolve the population using the evolutionary process.

        Parameters
        ----------
        data : dict
            Dictionary containing the training input samples and the target values.
        """
        start = time() # To control the timeout

        headers = ["gen", "nevals"]
        headers.extend(self.stats.fields)
        headers.extend(self.stats_elite.fields)

        chapter_headers = {category: self.stats[category].fields
                           for category in self.stats.fields}
        chapter_headers.update({category: self.stats_elite[category].fields
                                for category in self.stats_elite.fields})

        logbook = Logbook(headers=headers, chapter_headers=chapter_headers)

        population = [
            Individual(
                self.schema.create_syntax_tree(),
                Fitness((1.0,) if self.maximization else (-1.0,)),
            ) for _ in range(self.pop_size)
        ]

        chunksize = 1 if self.cpu_count == 1 else ceil((self.pop_size / self.cpu_count) * 0.25)

        manager = Manager()
        q = manager.Queue()
        listener = logging.handlers.QueueListener(q, *self.individuals_logger.handlers)
        listener.start()

        evaluate = partial(self._evaluate, data=data, start=start, queue=q)

        with ProcessPoolExecutor(max_workers=self.cpu_count) as pool:
            results = pool.map(evaluate, population, chunksize=chunksize)

        for ind, result in zip(population, results, strict=True):
            ind.fitness.values, ind.prediction, ind.runtime = result

        valid_population = [ind for ind in population if ind.fitness.valid]

        if len(valid_population) > 0:
            self.elite.update(valid_population)

        record = self.stats.compile(population)
        record_elite = self.stats_elite.compile(self.elite)
        logbook.record(gen=0, nevals=len(population), **record, **record_elite)
        report = logbook.stream

        self.evolution_logger.info(report)

        apply_operators = (self._apply_operators_double_mut if self.use_double_mutation
                           else self._apply_operators_cx_mut)

        for gen in range(1, self.generations + 1):
            if (time() - start) > self.timeout:
                return

            offspring = apply_operators(population)
            evals = [ind for ind in offspring if not ind.fitness.valid]

            with ProcessPoolExecutor(max_workers=self.cpu_count) as pool:
                results = pool.map(evaluate, offspring, chunksize=chunksize)

            for ind, result in zip(offspring, results, strict=True):
                ind.fitness.values, ind.prediction, ind.runtime = result

            population = self.replacement.replace(population, offspring, self.elite.elite)
            self.elite.update(population)

            record = self.stats.compile(population)
            record_elite = self.stats_elite.compile(self.elite)
            logbook.record(gen=gen, nevals=len(evals), **record, **record_elite)
            report = logbook.stream

            self.evolution_logger.info(report)

            if self._should_early_stop():
                return

        listener.stop()

    @abstractmethod
    def _create_ensemble_object(
        self,
        estimators: 'list[tuple[str, Pipeline]]',
        weights: 'ArrayLike',
    ) -> 'BaseVoting':
        """
        Create the ensemble object used to combine the predictions of the estimators.

        Parameters
        ----------
        estimators : list of (str, estimator) tuples
            List of tuples with the estimators to be included in the ensemble.

        weights : list of floats
            List of weights used to combine the predictions of the estimators.

        Returns
        -------
        ensemble : BaseVoting instance
            The ensemble model.
        """
        raise NotImplementedError("Method '_create_ensemble' must be implemented in subclass")

    @abstractmethod
    def _generate_ensemble(self, data: 'dict[str, ArrayLike]') -> None:
        """
        Generate the ensemble model using the individuals of the elite.

        Parameters
        ----------
        data : dict
            Dictionary containing the training input samples and the target values.
        """
        raise NotImplementedError("Method '_generate_ensemble' must be implemented in subclass")

    def _create_elite_object(self) -> 'DiverseElite':
        """
        Create the elite object used to store the best individuals of the population.

        By default, the elite object is a DiverseElite object with a diversity weight of 0.

        Returns
        -------
        elite : DiverseElite
            The elite object.
        """
        return DiverseElite(self.elite_size, div_weight=0)

    def _should_early_stop(self) -> bool:
        """
        Check if the evolutionary process should stop based on the early stopping criteria.

        Returns
        -------
        stop : bool
            Indicates if the evolutionary process should stop.
        """
        if self.early_stopping_generations is None and self.early_stopping_time is None:
            return False

        current_avg_fitness = float(np.mean([ind.fitness.values[0] for ind in self.elite]))

        if (current_avg_fitness - self.elite_avg_fitness) < self.early_stopping_threshold:
            self.generations_without_improvement += 1

            if self.early_stopping_generations is not None and \
                self.generations_without_improvement >= self.early_stopping_generations:
                self.evolution_logger.info(
                    f"Early stopping due to no improvement in the elite average fitness for "
                    f"{self.generations_without_improvement} generations",
                )
                return True

            current_time = time()

            if self.early_stopping_time is not None and \
                (current_time - self.last_improvement_time) >= self.early_stopping_time:
                self.evolution_logger.info(
                    f"Early stopping due to no improvement in the elite average fitness for "
                    f"{current_time - self.last_improvement_time} sec",
                )
                return True
        else:
            self.generations_without_improvement = 0
            self.elite_avg_fitness = current_avg_fitness
            self.last_improvement_time = time()

        return False

    def _apply_operators_cx_mut(self, population: list['Individual']) -> list['Individual']:
        offspring = self.selection.select(population, len(population))
        offspring = [ind.clone() for ind in offspring]

        for i in range(1, len(offspring), 2):
            offspring[i - 1], offspring[i], modified = self.crossover.cross(
                offspring[i - 1],
                offspring[i],
                self.schema,
            )

            if modified:
                offspring[i - 1].reset()
                offspring[i].reset()

        for i, _ in enumerate(offspring):
            offspring[i], modified = self.mutation.mutate(offspring[i], self.schema)

            if modified and hasattr(offspring[i], "pipeline"):
                offspring[i].reset()

        return offspring

    def _apply_operators_double_mut(self, population: list['Individual']) -> list['Individual']:
        offspring = [ind.clone() for ind in population]

        for i, _ in enumerate(offspring):
            if offspring[i] in self.elite:
                offspring[i], modified = self.mutation_elite.mutate(offspring[i], self.schema)
            else:
                offspring[i], modified = self.mutation.mutate(offspring[i], self.schema)

            if modified and hasattr(offspring[i], "pipeline"):
                offspring[i].reset()

        return offspring

    def save_model(self, filename: str) -> None:
        """
        Save the created ensemble to a file.

        Parameters
        ----------
        filename : str
            Name of the file where the ensemble will be saved.
        """
        # Check if the ensemble model has been created
        if not hasattr(self, "ensemble"):
            raise RuntimeError("The ensemble model has not been created yet")

        # Ensure the directory exists
        file_path = Path(self.outdir_path) / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with self.outdir_path.joinpath(filename).open("wb") as file:
            pickle.dump(self.ensemble, file)
