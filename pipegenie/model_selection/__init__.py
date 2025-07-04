#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

"""
Model selection module for the Genetic Programming algorithm.
"""

from ._split import BaseCrossValidator, KFold, StratifiedKFold, train_test_split

__all__ = [
    'BaseCrossValidator',
    'KFold',
    'StratifiedKFold',
    'train_test_split',
]
