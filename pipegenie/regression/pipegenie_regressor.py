#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

"""
Regressor that uses evolutionary algorithms to optimize pipelines for a given dataset.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from pipegenie._classification_regression import BaseClassificationRegression
from pipegenie.evolutionary.crossover import MultiCrossover
from pipegenie.evolutionary.mutation import HyperparameterMutation, MultiMutation
from pipegenie.evolutionary.replacement import ElitistGenerationalReplacement
from pipegenie.evolutionary.selection import TournamentSelection
from pipegenie.metrics import r2_score, root_mean_squared_error
from pipegenie.model_selection import KFold
from pipegenie.voting import VotingRegressor

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.connection import Connection
    from typing import Optional, Union

    from numpy.typing import ArrayLike, NDArray

    from pipegenie.evolutionary._individual import Individual
    from pipegenie.evolutionary.crossover import CrossoverBase
    from pipegenie.evolutionary.mutation import MutationBase
    from pipegenie.evolutionary.replacement import ReplacementBase
    from pipegenie.evolutionary.selection import SelectionBase
    from pipegenie.model_selection import BaseCrossValidator
    from pipegenie.pipeline import Pipeline

current_directory = Path(__file__).resolve().parent

class PipegenieRegressor(BaseClassificationRegression):
    """
    Regressor that uses evolutionary algorithms to optimize pipelines for a given dataset.

    Parameters
    ----------
    grammar : str
        Path of the file containing the grammar used to generate the pipelines.

    grammar_type : str, default="evoflow-xml"
        Format used to define the grammar.
        Supported formats: "evoflow-xml".
        You can add new formats by creating a new parser and registering it in the factory.

    pop_size : int, default=100
        Size of the population used in the evolutionary process.
        Must be greater than 0.

    generations : int, default=10
        Number of generations used in the evolutionary process.
        Must be greater or equal than 0.
        If 0, only the initial population will be evaluated.

    fitness : function, default=pipegenie.metrics.root_mean_squared_error
        Fitness function used to evaluate the pipelines.

    nderiv : int, default=13
        Number of derivations used to generate the pipelines.
        Must be greater than 0.

    selection : SelectionBase subclass instance, default=TournamentSelection(2)
        Selection function used to select the individuals
        that will be used to generate the offspring.

    crossover : CrossoverBase subclass instance, default=MultiCrossover(0.8)
        Crossover function used to generate new individuals.

    mutation : MutationBase subclass instance, default=MultiMutation(0.2)
        Mutation function used to generate new individuals.

    mutation_elite : MutationBase subclass instance, default=HyperparameterMutation(0.2)
        Mutation function used to generate new individuals from the elite.

    replacement : ReplacementBase subclass instance, default=ElitistGenerationalReplacement(10)
        Replacement function used to create the new population.

    use_double_mutation : bool, default=False
        Indicates if the mutation_elite function should be used
        to generate new individuals from the elite.
        No crossover will be applied.
        If False, the mutation function will be used instead and
        the crossover function will be applied.

    elite_size : int, default=10
        Maximun number of pipelines to be stored in the elite.
        Must be in the range [1, pop_size].

    timeout : int, default=3600
        Maximun time allowed for the evolutionary process.
        Must be greater than 0.

    eval_timeout : int, default=360
        Maximun time allowed for the evaluation of a pipeline.
        It can be None, in which case the evaluation of the individuals
        will not be stopped, or greater than 0.

    cross_validator : BaseCrossValidator subclass instance, default=KFold(5, True)
        Cross-validator used to evaluate the pipelines.

    maximization : bool, default=False
        Indicates if the fitness function should be maximized or minimized.

    early_stopping_threshold : float, default=0.0001
        Threshold used to determine if there is improvement in the elite average fitness.
        Must be greater or equal than 0.0.

    early_stopping_generations : int, default=None
        Number of generations without improvement in the elite average fitness
        before stopping the evolutionary process.
        It can be None, in which case the evolutionary process will not stop
        based on generations without improvement, or greater than 0.

    early_stopping_time : int, default=None
        Maximun time without improvement in the elite average fitness
        before stopping the evolutionary process.
        It can be None, in which case the evolutionary process will not stop
        based on time without improvement, or greater than 0.

    seed : int, default=None
        Seed used to initialize the random number generator.
        If None, a random seed will be used.

    outdir : str, default="results"
        Path of the directory where the results will be stored.

    n_jobs : int, default=1
        Number of processes used to evaluate the population.
        Must be greater than 0.
        If -1, the number of processes will be equal to the number of cores of the machine.
    """

    def __init__(
        self,
        grammar: str = current_directory / "assets/grammar-regression.xml",
        *,
        grammar_type: str = "evoflow-xml",
        pop_size: int = 100,
        generations: int = 10,
        fitness: 'Callable[..., object]' = root_mean_squared_error,
        nderiv: int = 13,
        selection: 'SelectionBase' = TournamentSelection(2),
        crossover: 'CrossoverBase' = MultiCrossover(0.8),
        mutation: 'MutationBase' = MultiMutation(0.2),
        mutation_elite: 'MutationBase' = HyperparameterMutation(0.2),
        replacement: 'ReplacementBase' = ElitistGenerationalReplacement(10),
        use_double_mutation: bool = False,
        elite_size: int = 10,
        timeout: int = 3600,
        eval_timeout: int = 360,
        cross_validator: 'BaseCrossValidator' = KFold(5, shuffle=True),
        maximization: bool = False,
        early_stopping_threshold: float = 0.0001,
        early_stopping_generations: 'Optional[int]' = None,
        early_stopping_time: 'Optional[int]' = None,
        seed: 'Optional[int]' = None,
        outdir: str = "results",
        n_jobs: str = 1,
    ):
        super().__init__(
            grammar,
            grammar_type=grammar_type,
            pop_size=pop_size,
            generations=generations,
            fitness=fitness,
            nderiv=nderiv,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            mutation_elite=mutation_elite,
            replacement=replacement,
            use_double_mutation=use_double_mutation,
            elite_size=elite_size,
            timeout=timeout,
            eval_timeout=eval_timeout,
            cross_validator=cross_validator,
            maximization=maximization,
            early_stopping_threshold=early_stopping_threshold,
            early_stopping_generations=early_stopping_generations,
            early_stopping_time=early_stopping_time,
            seed=seed,
            outdir=outdir,
            n_jobs=n_jobs,
        )

    def fit(self, X: 'ArrayLike', y: 'ArrayLike') -> 'PipegenieRegressor':
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : PipegenieRegressor
            The fitted model.
        """
        return super().fit(X=X, y=y)

    def _create_ensemble_object(
        self,
        estimators: 'list[tuple[str, Pipeline]]',
        weights: 'ArrayLike',
    ) -> 'VotingRegressor':
        return VotingRegressor(estimators, weights, n_jobs=self.cpu_count)

    def _evaluate_pipeline(
        self,
        ind: 'Individual',
        X_train: 'ArrayLike',
        y_train: 'ArrayLike',
        X_test: 'ArrayLike',
        pid_conn: 'Optional[Connection]',
        memory_limit: int,
    ) -> 'Union[NDArray, str]':
        if pid_conn is not None:
            pid_conn.send(os.getpid())
            pid_conn.close()

        try:
            if not hasattr(ind, "pipeline") or ind.pipeline is None:
                # Create the pipeline in the child to avoid issues when using GPU and multiprocessing
                ind.create_pipeline(self.pset, memory_limit=memory_limit)

            ind.pipeline.fit(X_train, y_train)
            return ind.pipeline.predict(X_test)
        except Exception as e:
            return "eval_error (" + str(e) + ")"

    def predict(self, X: 'ArrayLike') -> 'NDArray':
        """
        Predict the target values of the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted target values.
        """
        return super().predict(X)

    def score(self, X: 'ArrayLike', y: 'ArrayLike') -> float:
        """
        Return the coefficient of determination (R²) score of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        score : float
            The R² score of the model.
        """
        return r2_score(y, self.predict(X))
