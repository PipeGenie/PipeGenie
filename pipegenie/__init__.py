#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

"""
PipeGenie: A Genetic Programming library for pipeline optimization.
"""
import sys

if sys.version_info > (3, 13):
    raise RuntimeError("Python < 3.13 is required.")
elif sys.version_info < (3, 10):
    raise RuntimeError("Python >= 3.10 is required.")
