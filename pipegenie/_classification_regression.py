#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

"""
Base class for pipegenie classification and regression models.
"""

import os
import signal
from abc import ABC, abstractmethod
from copy import deepcopy
from multiprocessing import Pipe
from time import time
from typing import TYPE_CHECKING

import numpy as np
from loky import TimeoutError, get_reusable_executor

from pipegenie.base import BasePipegenie

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from queue import Queue
    from typing import Optional, Union

    from numpy.typing import ArrayLike, NDArray

    from pipegenie.evolutionary._individual import Individual


class BaseClassificationRegression(BasePipegenie, ABC):
    """
    Base class for pipegenie classification and regression models.

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

    use_double_mutation : bool
        Indicates if the mutation_elite function should be used to generate new individuals
        from the elite instead of the mutation function.
        No crossover will be applied.
        If False, the crossover function will be applied and then the mutation function.

    elite_size : int
        Maximun number of pipelines to be stored in the elite.
        must be in the range [1, pop_size].

    intergenerational_elite : int
        Number of individuals from the elite that will be included in the next generation.
        Must be in the range [0, elite_size].
        If 0, no individuals from the elite will be included in the next generation.

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

    @abstractmethod
    def fit(
        self,
        X: 'ArrayLike',
        y: 'ArrayLike',
        **kwargs: object,
    ) -> 'BaseClassificationRegression':
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if y is None or len(y) == 0:
            raise ValueError("The target values are missing")

        if X is None or len(X) == 0 or (X is not None and len(X) != len(y)):
            raise ValueError("The training input samples are missing or have an invalid shape")

        X_copy = np.array(X)
        y_copy = np.array(y)

        # Ensure y is unidimensional
        if y_copy.ndim > 1:
            if y_copy.shape[1] == 1:
                y_copy = y_copy.ravel()
            else:
                raise ValueError("The 'y' parameter should be unidimensional")

        data = {
            "X": X_copy,
            "y": y_copy,
            "kwargs": kwargs,
        }

        self._fit(data)
        return self

    def _evaluate(
        self,
        ind: 'Individual',
        data: 'dict[str, ArrayLike]',
        start: float,
        queue: 'Queue',
    ) -> 'tuple[float, Optional[NDArray], Optional[float]]':
        if "X" not in data or "y" not in data:
            raise ValueError("Missing evaluation data. Either 'X' or 'y' is missing.")

        individuals_logger = self._create_worker_logger(queue)

        if ind.fitness.valid:
            # Use cached fitness
            individuals_logger.info(f"{ind}\t{ind.fitness.values[0]}\tcached")
            return ind.fitness.values, ind.prediction, ind.runtime

        if (time() - start) > self.timeout:
            individuals_logger.info(f"{ind}\ttimeout_error\ttimeout_error")

            return (np.nan,), None, None

        start_eval = time()
        fitness, predictions = self._evaluate_cv(ind, data["X"], data["y"])
        elapsed_time = time() - start_eval

        if isinstance(fitness, str):
            individuals_logger.info(f"{ind}\t{fitness}\t{fitness}")
            return (np.nan,), None, None

        individuals_logger.info(f"{ind}\t{fitness}\t{elapsed_time}")
        return (fitness,), predictions, elapsed_time

    def _evaluate_cv(
        self,
        ind: 'Individual',
        X: 'ArrayLike',
        y: 'ArrayLike',
        memory_limit: int = 3072,
    ) -> 'tuple[float, Optional[NDArray]]':
        pipe_str = str(ind).split(";")
        ops_str = [op[:op.index("(")] for op in pipe_str]

        if len(ops_str) != len(set(ops_str)):
            return "invalid_ind (duplicated operator)", None

        if self.eval_timeout is not None:
            fold_timeout = self.eval_timeout / self.cross_validator.n_splits

        fitness_values = []
        predictions = np.empty(0)
        executor = get_reusable_executor(max_workers=1)

        # Create copy of cross_validator to avoid issues with multiprocessing
        cross_validator = deepcopy(self.cross_validator)

        for train_idx, test_idx in cross_validator.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if self.eval_timeout is None:
                future = executor.submit(
                    self._evaluate_pipeline,
                    ind,
                    X_train,
                    y_train,
                    X_test,
                    None,
                    memory_limit,
                )
                result = future.result()

                if isinstance(result, str):
                    executor.shutdown(wait=False)
                    return result, None

                y_pred = result
            else:
                parent_conn, child_conn = Pipe(duplex=False)
                future = executor.submit(
                    self._evaluate_pipeline,
                    ind,
                    X_train,
                    y_train,
                    X_test,
                    child_conn,
                    memory_limit,
                )
                result = None

                try:
                    worker_pid = parent_conn.recv()
                    result = future.result(timeout=fold_timeout)
                except TimeoutError:
                    os.kill(worker_pid, signal.SIGINT)
                    result = "eval_timeout_error"
                finally:
                    parent_conn.close()

                if isinstance(result, str):
                    executor.shutdown(wait=False)
                    return result, None

                y_pred = result

            try:
                y_pred = y_pred[:, 1] if y_pred.ndim == 2 and y_pred.shape[1] <= 2 else y_pred
                predictions = np.concatenate((predictions, y_pred))

                fitness = self.fitness(y_test, y_pred)
                fitness_values.append(fitness)
            except Exception as e:
                executor.shutdown(wait=False)
                return "metric_error (" + str(e) + ")", None

        executor.shutdown(wait=False)
        return np.mean(fitness_values), predictions

    @abstractmethod
    def _evaluate_pipeline(
        self,
        ind: 'Individual',
        X_train: 'ArrayLike',
        y_train: 'ArrayLike',
        X_test: 'ArrayLike',
        pid_conn: 'Optional[Connection]',
        memory_limit: int,
    ) -> 'Union[NDArray, str]':
        raise NotImplementedError("Method '_evaluate_pipeline' must be implemented in a subclass.")

    def _generate_ensemble(self, data: 'dict[str, ArrayLike]') -> None:
        X = data["X"]
        y = data["y"]

        best_ind = self.elite.best_ind()
        best_ind.pipeline.fit(X, y)
        best_fitness = best_ind.fitness.values[0]

        weights = [float(ind.fitness.values[0] / best_fitness) if self.maximization
                   else float(best_fitness / ind.fitness.values[0])
                   for ind in self.elite]

        estimators = [(str(idx), est.pipeline) for idx, est in enumerate(self.elite)]
        self.ensemble = self._create_ensemble_object(estimators, weights)
        # Reset ramdon state
        self.cross_validator.set_random_state(self.seed)
        fitness_values = []
        
        # Evaluate the ensemble with the cross-validator
        for train_idx, test_idx in self.cross_validator.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self.ensemble.fit(X_train, y_train)
            y_pred = self.ensemble.predict(X_test)

            fitness = self.fitness(y_test, y_pred)
            fitness_values.append(fitness)

        # Retrain the ensemble with the whole dataset
        self.ensemble.fit(X, y)

        with self.outdir_path.joinpath("best_pipeline.txt").open("w", encoding="utf-8") as log:
            log.write(str(best_ind) + "\n")
            log.write(f"Fitness: {best_fitness}\n")
            log.write(f"Prediction: {best_ind.pipeline.predict(X)}\n")

        with self.outdir_path.joinpath("ensemble.txt").open("w", encoding="utf-8") as log:
            for idx, (name, est) in enumerate(estimators):
                log.write(f"{name}: {est} -> Fitness: {self.elite[idx].fitness.values[0]}\n\n")

            log.write(f"Ensemble fitness: {np.mean(fitness_values)}\n")
            log.write(f"Weights: {weights}\n")
            log.write(f"Prediction: {self.ensemble.predict(X)}\n")

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
        return self.ensemble.predict(X)

    @abstractmethod
    def score(self, X: 'ArrayLike', y: 'ArrayLike') -> float:
        """
        Return the score of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        score : float
            The score of the model.
        """
        raise NotImplementedError("Method 'score' must be implemented in a subclass.")
