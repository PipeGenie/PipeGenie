import pytest
from pipegenie.model_selection import KFold, StratifiedKFold, train_test_split
import numpy as np
import math
from unittest.mock import patch

def _calculate_thresholds(n_samples, y_counts, n_folds):
    group_size = n_samples / n_folds
    thresholds = {}
    
    for i, count in enumerate(y_counts):
        p_target = count / n_samples
        sigma_target = math.sqrt(p_target * (1 - p_target) / group_size)
        percentage_threshold = sigma_target * 100
        absolute_deviation = sigma_target * group_size
        thresholds[i] = (percentage_threshold, absolute_deviation)

    return thresholds

class TestKFold:
    @pytest.fixture()
    def setup(self):
        X = np.arange(50).reshape((25, 2))
        y = np.arange(50)
        n_splits = 5
        return X, y, n_splits

    def test_fold_sizes(self, setup):
        X, y, n_splits = setup
        kfold = KFold(n_splits=n_splits)
        assert kfold.n_splits == n_splits
        assert len(list(kfold.split(X))) == n_splits

        for train_indices, test_indices in kfold.split(X):
            assert len(train_indices) == 20
            assert len(test_indices) == 5

    def test_shuffle(self, setup):
        X, y, n_splits = setup
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        first_run = list(kfold.split(X))
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        second_run = list(kfold.split(X))
        
        # same seed should produce the same results
        for (train_indices1, test_indices1), (train_indices2, test_indices2) in zip(first_run, second_run):
            assert np.array_equal(train_indices1, train_indices2)
            assert np.array_equal(test_indices1, test_indices2)

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
        third_run = list(kfold.split(X))
        
        # different seeds should produce different results if data is big enough
        for (train_indices1, test_indices1), (train_indices2, test_indices2) in zip(first_run, third_run):
            assert not np.array_equal(train_indices1, train_indices2)
            assert not np.array_equal(test_indices1, test_indices2)

class TestStratifiedKFold:
    @pytest.fixture
    def setup(self):
        X = np.arange(30).reshape((15, 2))
        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) # 10 zeros and 5 ones
        n_splits = 3
        return X, y, n_splits

    def test_fold_sizes(self, setup):
        X, y, n_splits = setup
        skfold = StratifiedKFold(n_splits=n_splits)
        assert skfold.n_splits == n_splits
        assert len(list(skfold.split(X, y))) == n_splits

        for train_indices, test_indices in skfold.split(X, y):
            assert len(train_indices) == 10
            assert len(test_indices) == 5

    def test_shuffle(self, setup):
        X, y, n_splits = setup
        skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        first_run = list(skfold.split(X, y))
        skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        second_run = list(skfold.split(X, y))
        
        # same seed should produce the same results
        for (train_indices1, test_indices1), (train_indices2, test_indices2) in zip(first_run, second_run):
            assert np.array_equal(train_indices1, train_indices2)
            assert np.array_equal(test_indices1, test_indices2)

        skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
        third_run = list(skfold.split(X, y))
        
        # different seeds should produce different results if data is big enough
        for (train_indices1, test_indices1), (train_indices2, test_indices2) in zip(first_run, third_run):
            assert not np.array_equal(train_indices1, train_indices2)
            assert not np.array_equal(test_indices1, test_indices2)

    def test_stratification(self, setup):
        X, y, n_splits = setup
        skfold = StratifiedKFold(n_splits=n_splits)
        y_counts = np.bincount(y)
        y_counts = np.sort(y_counts)
        class_samples_per_test_fold = np.round(y_counts / n_splits).astype(int)
        thresholds = _calculate_thresholds(len(y), y_counts, n_splits)
        
        for train_indices, test_indices in skfold.split(X, y):
            train_y = y[train_indices]
            test_y = y[test_indices]
            train_y_counts = np.bincount(train_y)
            train_y_counts = np.sort(train_y_counts)
            test_y_counts = np.bincount(test_y)
            test_y_counts = np.sort(test_y_counts)

            for i, (train_count, test_count) in enumerate(zip(train_y_counts, test_y_counts)):
                test_class_samples = class_samples_per_test_fold[i]
                train_class_samples = test_class_samples * (n_splits - 1)
                threshold = thresholds[i][1]
                assert abs(train_count - train_class_samples) <= threshold
                assert abs(test_count - test_class_samples) <= threshold

    def test_stratification_shuffle(self, setup):
        X, y, n_splits = setup
        skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        y_counts = np.bincount(y)
        y_counts = np.sort(y_counts)
        class_samples_per_test_fold = np.round(y_counts / n_splits).astype(int)
        thresholds = _calculate_thresholds(len(y), y_counts, n_splits)
        
        for train_indices, test_indices in skfold.split(X, y):
            train_y = y[train_indices]
            test_y = y[test_indices]
            train_y_counts = np.bincount(train_y)
            train_y_counts = np.sort(train_y_counts)
            test_y_counts = np.bincount(test_y)
            test_y_counts = np.sort(test_y_counts)

            for i, (train_count, test_count) in enumerate(zip(train_y_counts, test_y_counts)):
                test_class_samples = class_samples_per_test_fold[i]
                train_class_samples = test_class_samples * (n_splits - 1)
                threshold = thresholds[i][1]
                assert abs(train_count - train_class_samples) <= threshold
                assert abs(test_count - test_class_samples) <= threshold

class TestTrainTestSplit:
    class MockRandomState:
        def __init__(self, seed=None):
            pass
        def shuffle(self, x):
            x[:] = x[::-1]  # Reverse to simulate the shuffle
 
    @pytest.fixture
    def setup(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        y = np.array([0, 0, 0, 0, 1, 1])
        return X, y
    
    def test_split_sizes(self, setup):
        X, y = setup
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        assert len(X_train) == 4
        assert len(X_test) == 2
        assert len(y_train) == 4
        assert len(y_test) == 2

    def test_no_shuffle(self, setup):
        X, y = setup
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
        assert np.array_equal(X_train, [[1, 2], [3, 4], [5, 6], [7, 8]])
        assert np.array_equal(X_test, [[9, 10], [11, 12]])
        assert np.array_equal(y_train, [0, 0, 0, 0])
        assert np.array_equal(y_test, [1, 1])

    def test_shuffle(self, setup):
        X, y = setup
        
        # We cannot use the shuffle parameter directly as RandomState is inmutable
        # We need to mock the RandomState class to simulate the shuffle
        with patch('numpy.random.RandomState', self.MockRandomState):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None, shuffle=True)
            assert np.array_equal(X_train, [[11, 12], [9, 10], [7, 8], [5, 6]])
            assert np.array_equal(X_test, [[3, 4], [1, 2]])
            assert np.array_equal(y_train, [1, 1, 0, 0])
            assert np.array_equal(y_test, [0, 0])

    def test_stratify_no_shuffle(self, setup):
        X, y = setup
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0, shuffle=False)
        assert np.array_equal(X_train, [[1, 2], [3, 4], [5, 6], [9, 10]])
        assert np.array_equal(X_test, [[7, 8], [11, 12]])
        assert np.array_equal(y_train, [0, 0, 0, 1])
        assert np.array_equal(y_test, [0, 1])

    def test_stratify_shuffle(self, setup):
        X, y = setup
        
        # We cannot use the shuffle parameter directly as RandomState is inmutable
        # We need to mock the RandomState class to simulate
        with patch('numpy.random.RandomState', self.MockRandomState):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=None, shuffle=True)
            assert np.array_equal(X_train, [[11, 12], [3, 4], [5, 6], [7, 8]])
            assert np.array_equal(X_test, [[9, 10], [1, 2]])
            assert np.array_equal(y_train, [1, 0, 0, 0])
            assert np.array_equal(y_test, [1, 0])

    def test_invalid_test_size(self, setup):
        X, y = setup
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size=0, stratify=y)
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size=1, stratify=y)

    def test_reproducibility(self, setup):
        rng = np.random.RandomState(0)
        X = rng.rand(100, 10)
        y = rng.randint(0, 2, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.25, random_state=0)
        assert np.array_equal(X_train, X_train2)
        assert np.array_equal(X_test, X_test2)
        assert np.array_equal(y_train, y_train2)
        assert np.array_equal(y_test, y_test2)

        X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.25, random_state=1)
        assert not np.array_equal(X_train, X_train3)
        assert not np.array_equal(X_test, X_test3)
        assert not np.array_equal(y_train, y_train3)
        assert not np.array_equal(y_test, y_test3)

    def test_stratify_reproducibility(self, setup):
        rng = np.random.RandomState(0)
        X = rng.rand(100, 10)
        y = rng.randint(0, 2, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
        assert np.array_equal(X_train, X_train2)
        assert np.array_equal(X_test, X_test2)
        assert np.array_equal(y_train, y_train2)
        assert np.array_equal(y_test, y_test2)

        X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)
        assert not np.array_equal(X_train, X_train3)
        assert not np.array_equal(X_test, X_test3)
        assert not np.array_equal(y_train, y_train3)
        assert not np.array_equal(y_test, y_test3)