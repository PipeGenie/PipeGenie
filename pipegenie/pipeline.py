#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

"""
Pipeline for chaining data transformers and resamplers with a final estimator.
"""

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from pipegenie._available_if import available_if

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Optional, Union

    from numpy.typing import ArrayLike, NDArray

def _final_estimator_has(attr: str) -> 'Callable[[object], bool]':
    """
    Check if the final estimator has a given attribute.

    Parameters
    ----------
    attr : str
        The attribute to check for.

    Returns
    -------
    check : callable
        A function that returns True if the final estimator has the given attribute.
    """
    def _check(self: object) -> bool:
        if isinstance(self, Pipeline):
            return hasattr(self.final_estimator, attr)

        raise ValueError("The object is not a Pipeline")

    return _check

class Pipeline:
    """
    A sequence of data transformers and resamplers followed by a final estimator.

    A pipeline allows to sequentially apply a list of transformers to preprocess the data
    and resamplers to resample the data, and conclude the sequence with a model prediction.

    Intermediate steps of the pipeline must be transformers (i.e., they must
    implement `fit` and `transform`, or `fit_transform` and `transform` methods),
    or resamplers (i.e., they must implement `fit_resample` method).
    The final estimator needs to implement `fit` method and a prediction method (e.g., `predict`).

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters.

    Parameters
    ----------
    steps : list of (str, object) tuples
        List of (name, transform) tuples that are chained in sequential order.
        All the steps in the pipeline must be transformers or resamplers except
        the final estimator, which can be any object that implements `fit` method
        and a prediction method (e.g., `predict`).

    memory_limit : int, default=None
        Maximum memory usage in MB. If `None`, memory usage is not limited.
    """

    def __init__(self, steps: list[tuple[str, object]], memory_limit: 'Optional[int]' = None):
        self.steps = steps
        self.memory_limit = memory_limit
        self._validate_steps()

    def _validate_names(self, names: tuple[str, ...]) -> None:
        """
        Validate the step names.

        Check that the step names are unique.
        """
        if len(set(names)) != len(names):
            raise ValueError("Step names are not unique")

    def _validate_steps(self) -> None:
        """
        Validate the pipeline steps.

        Check that the final estimator implements `fit` and intermediate steps
        implement at least one of `fit` and `transform`, `fit_transform` and `transform`,
        or `fit_resample`.
        """
        names, estimators = zip(*self.steps, strict=True)
        # validate names
        self._validate_names(names)

        def has_attribute(estimator: object, attr: 'Union[str, Sequence[str]]') -> bool:
            if isinstance(attr, str):
                return hasattr(estimator, attr)

            return all(hasattr(estimator, a) for a in attr)

        # validate steps
        required_combinations = [
            ("fit", "transform"),
            ("fit_transform", "transform"),
            "fit_resample",
        ]

        for estimator in estimators[:-1]:
            if not any(has_attribute(estimator, required_combination)
                       for required_combination in required_combinations):
                raise ValueError("All intermediate steps should implement at least ",
                                 f"one of the following combinations: {required_combinations}")

        predictors = ["predict", "predict_proba", "predict_log_proba"]

        if (not hasattr(estimators[-1], "fit") or
                not any(hasattr(estimators[-1], predictor) for predictor in predictors)):
            raise ValueError("The final estimator should implement 'fit' and a prediction ",
                             "method ('predict', 'predict_proba', or 'predict_log_proba')")

    @property
    def named_steps(self) -> dict[str, object]:
        """
        Get a dictionary mapping step names to their instances.

        Returns
        -------
        named_steps : dict of str: object
            Dictionary mapping step names to their instances.
        """
        return dict(self.steps)

    @property
    def final_estimator(self) -> object:
        """
        Get the final estimator in the pipeline.

        Returns
        -------
        estimator : object
            The final estimator in the pipeline.
        """
        return self.steps[-1][1]
    
    def __getitem__(self, index: int) -> object:
        """
        Get the algorithm at the given index.

        Parameters
        ----------
        index : int
            The index of the step.

        Returns
        -------
        estimator : object
            The algorithm at the given index.
        """
        return self.steps[index][1]

    def __repr__(self) -> str:
        """
        Return the string representation of the pipeline.
        """
        return f"Pipeline(steps={self.steps})"

    def __str__(self) -> str:
        """
        Return the string representation of the pipeline in a readable format.
        """
        steps = [f"\t{name}: {step}" for name, step in self.steps]
        pipeline_str = "\n".join(steps)
        return f"Pipeline(\n{pipeline_str}\n)"

    def fit(self, X: 'ArrayLike', y: 'ArrayLike') -> 'Pipeline':
        """
        Fit the model according to the given training data.

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
        X_copy = np.array(X)
        y_copy = np.array(y)

        for _, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                X_copy = step.fit_transform(X_copy, y_copy)
            elif hasattr(step, "transform"):
                step.fit(X_copy, y_copy)
                X_copy = step.transform(X_copy)
            elif hasattr(step, "fit_resample"):
                X_copy, y_copy = step.fit_resample(X_copy, y_copy)

            if self.memory_limit is not None and (X_copy.nbytes / (1024 ** 2)) > self.memory_limit:
                raise MemoryError("Memory limit exceeded")

        self.steps[-1][1].fit(X_copy, y_copy)
        return self

    @available_if(_final_estimator_has("predict"))
    def predict(self, X: 'ArrayLike') -> 'NDArray':
        """
        Transform the input samples and apply `predict` with the final estimator.

        Call `transform` on each transformer in the pipeline. The transformed
        data is passed to the `predict` method of the final estimator. Only available
        if the final estimator implements `predict`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
             Result of the final estimator's `predict` method.
        """
        X_copy = np.array(X)

        for _, step in self.steps[:-1]:
            if hasattr(step, "transform"):
                X_copy = step.transform(X_copy)

        return self.steps[-1][1].predict(X_copy)

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X: 'ArrayLike') -> 'NDArray':
        """
        Transform the input samples and apply `predict_proba` with the final estimator.

        Call `transform` on each transformer in the pipeline. The transformed
        data is passed to the `predict_proba` method of the final estimator. Only available
        if the final estimator implements `predict_proba`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Result of the final estimator's `predict_proba` method.
        """
        X_copy = np.array(X)

        for _, step in self.steps[:-1]:
            if hasattr(step, "transform"):
                X_copy = step.transform(X_copy)

        return self.steps[-1][1].predict_proba(X_copy)

    @available_if(_final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X: 'ArrayLike') -> 'NDArray':
        """
        Transform the input samples and apply `predict_log_proba` with the final estimator.

        Call `transform` on each transformer in the pipeline. The transformed
        data is passed to the `predict_log_proba` method of the final estimator. Only available
        if the final estimator implements `predict_log_proba`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Result of the final estimator's `predict_log_proba` method.
        """
        X_copy = np.array(X)

        for _, step in self.steps[:-1]:
            if hasattr(step, "transform"):
                X_copy = step.transform(X_copy)

        return self.steps[-1][1].predict_log_proba(X_copy)

    @available_if(_final_estimator_has("score"))
    def score(self, X: 'ArrayLike', y: 'ArrayLike') -> float:
        """
        Transform the input samples and apply `score` with the final estimator.

        Call `transform` on each transformer in the pipeline. The transformed
        data is passed to the `score` method of the final estimator. Only available
        if the final estimator implements `score`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        score : float
            Result of the final estimator's `score` method.
        """
        X_copy = np.array(X)

        for _, step in self.steps[:-1]:
            if hasattr(step, "transform"):
                X_copy = step.transform(X_copy)

        return self.steps[-1][1].score(X_copy, y)

    @available_if(_final_estimator_has("decision_function"))
    def decision_function(self, X: 'ArrayLike') -> 'NDArray':
        """
        Transform the input samples and apply `decision_function` with the final estimator.

        Call `transform` on each transformer in the pipeline. The transformed
        data is passed to the `decision_function` method of the final estimator. Only available
        if the final estimator implements `decision_function`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Result of the final estimator's `decision_function` method.
        """
        X_copy = np.array(X)

        for _, step in self.steps[:-1]:
            if hasattr(step, "transform"):
                X_copy = step.transform(X_copy)

        return self.steps[-1][1].decision_function(X_copy)

def _name_estimators(estimators: list[object]) -> list[tuple[str, object]]:
    """
    Generate names for the estimators.

    Parameters
    ----------
    estimators : list of objects
        List of estimators.

    Returns
    -------
    estimators : list of (str, object) tuples
        List of (name, estimator) tuples.
    """
    names = [type(estimator).__name__.lower() for estimator in estimators]
    names_count: dict[str, int] = defaultdict(int)

    for name in names:
        names_count[name] += 1

    for k, v in names_count.items():
        if v == 1:
            del names_count[k]

    for i in reversed(range(len(names))):
        name = names[i]

        if name in names_count:
            names[i] = "{0}-{1}".format(name, names_count[name])
            names_count[name] -= 1

    return list(zip(names, estimators, strict=True))

def make_pipeline(steps: list[object]) -> 'Pipeline':
    """
    Construct a pipeline from the given transformers/resamplers and final estimator.

    Parameters
    ----------
    steps : list of objects
        List of transformers/resamplers and a final estimator. The intermediate steps should
        implement `fit` and `transform`, `fit_transform` and `transform`, or `fit_resample`.
        The final estimator should implement `fit`.

    Returns
    -------
    pipeline : Pipeline
        A pipeline that sequentially applies the given transformers and concludes
        with the final estimator.
    """
    return Pipeline(_name_estimators(steps))
