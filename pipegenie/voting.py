#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

"""
Soft voting / majority rule voting classifier and voting regressor.
"""

import warnings
from abc import ABC, abstractmethod
from itertools import repeat
from multiprocessing import cpu_count
from typing import TYPE_CHECKING

import numpy as np
from loky import ProcessPoolExecutor

from pipegenie._available_if import available_if
from pipegenie.metrics import accuracy_score, r2_score

warnings.filterwarnings("ignore")

if TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import ArrayLike, NDArray

# Ignore all warnings
warnings.filterwarnings("ignore")

# Show warnings from the "pipegenie" package
warnings.filterwarnings("default", module="^pipegenie\.")


class BaseVoting(ABC):
    """
    Base class for all voting ensembles.

    Ensemble methods combine the predictions of several base estimators
    in order to improve generalization / robustness over a single estimator.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        The list of estimators to fit and vote on.

    weights : array-like of shape (n_estimators,) or None, default=None
        The weights of the estimators. If None, then uniform weights are used.

    n_jobs : int, default=1
        The number of jobs to run in parallel.
        If -1, then the number of jobs is set to the number of CPU cores.
    """

    def __init__(
        self,
        estimators: list[tuple[str, object]],
        weights: 'Optional[ArrayLike]' = None,
        n_jobs: int = 1,
    ):
        self.estimators = estimators
        self.weights = weights
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    @abstractmethod
    def fit(self, X: 'ArrayLike', y: 'ArrayLike') -> 'BaseVoting':
        """
        Fit the model to the data.
        """
        if self.weights is not None and len(self.weights) != len(self.estimators):
            raise ValueError("The number of weights should be equal to the number of estimators.")

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = executor.map(
                self._fit_single_estimator,
                [est[1] for est in self.estimators],
                repeat(X),
                repeat(y),
            )

        self.estimators = list(zip([est[0] for est in self.estimators], results, strict=True))

        return self

    def _fit_single_estimator(self, estimator: object, X: 'ArrayLike', y: 'ArrayLike') -> object:
        return estimator.fit(X, y)

    @abstractmethod
    def predict(self, X: 'ArrayLike') -> 'NDArray':
        """
        Predict using the model.
        """
        raise NotImplementedError("predict method should be implemented by the subclass")

    @abstractmethod
    def score(self, X: 'ArrayLike', y: 'ArrayLike') -> float:
        """
        Return the score of the model on the input data.
        """
        raise NotImplementedError("score method should be implemented by the subclass")


class VotingClassifier(BaseVoting):
    """
    A voting classifier that uses the majority rule voting strategy.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        The list of estimators to fit and vote on.

    weights : array-like of shape (n_classifiers,) or None, default=None
        The weights of the classifiers. If None, then uniform weights are used.

    n_jobs : int, default=1
        The number of jobs to run in parallel.
        If -1, then the number of jobs is set to the number of CPU cores.
    """

    def __init__(
        self,
        estimators: list[tuple[str, object]],
        weights: 'Optional[ArrayLike]' = None,
        n_jobs: int = 1,
    ):
        self.estimators = estimators
        self.weights = weights
        self.n_jobs = n_jobs

    def fit(self, X: 'ArrayLike', y: 'ArrayLike') -> 'VotingClassifier':
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
        return super().fit(X, y)

    def predict(self, X: 'ArrayLike') -> 'NDArray':
        """
        Predict using the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        predictions = np.array([estimator.predict(X) for _, estimator in self.estimators])
        return np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x, weights=self.weights)),
            axis=0,
            arr=predictions,
        )

    def _has_predict_proba(self) -> bool:
        """
        Check if the estimators have a predict_proba method.

        Returns
        -------
        has_proba : bool
            True if all estimators have a predict_proba method, False otherwise.
        """
        return all(hasattr(est[-1], "predict_proba") for _, est in self.estimators)

    @available_if(_has_predict_proba)
    def predict_proba(self, X: 'ArrayLike') -> 'NDArray':
        """
        Predict the probabilities of the classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array-like of shape (n_samples, n_classes)
            The predicted probabilities of the classes.
        """
        proba_predictions = [est.predict_proba(X) for _, est in self.estimators]
        return np.average(proba_predictions, axis=0, weights=self.weights)

    def score(self, X: 'ArrayLike', y: 'ArrayLike') -> float:
        """
        Return the accuracy of the model on the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        score : float
            The accuracy score of the model on the input data.
        """
        return accuracy_score(y, self.predict(X))


class VotingRegressor(BaseVoting):
    """
    A voting regressor that uses the average of the predictions.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        The list of estimators to fit and vote on.

    weights : array-like of shape (n_classifiers,) or None, default=None
        The weights of the classifiers. If None, then uniform weights are used.

    n_jobs : int, default=1
        The number of jobs to run in parallel.
        If -1, then the number of jobs is set to the number of CPU cores.
    """

    def __init__(
        self,
        estimators: list[tuple[str, object]],
        weights: 'Optional[ArrayLike]' = None,
        n_jobs: int = 1,
    ):
        self.estimators = estimators
        self.weights = weights
        self.n_jobs = n_jobs

    def fit(self, X: 'ArrayLike', y: 'ArrayLike') -> 'VotingRegressor':
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
        return super().fit(X, y)

    def predict(self, X: 'ArrayLike') -> 'NDArray':
        """
        Predict using the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        predictions = [est.predict(X) for _, est in self.estimators]
        return np.average(predictions, axis=0, weights=self.weights)

    def score(self, X: 'ArrayLike', y: 'ArrayLike') -> float:
        """
        Return the R2 score of the model on the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        score : float
            The R2 score of the model on the input data.
        """
        return r2_score(y, self.predict(X))
