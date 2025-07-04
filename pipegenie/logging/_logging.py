#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

# This file contains a class based on DEAP's tools.Logbook to log data.
# Source:
# - tools.Logbook: https://github.com/DEAP/deap/blob/master/deap/tools/support.py#L259
# Modified by: Ángel Fuentes Almoguera
# Date: 25/05/2024
# Modifications:
# - Added header and chapter_headers attributes to the constructor.
# - Floating point values are formatted to use up to 10 characters
#   and scientific notation if needed.
# - Switched from \t to 4 spaces for indentation.
#   More consistency when showing in terminal and text editors.
# - If a key does not have a value associated with it, it is filled with NaN.
# - Reduced complexity of the __txt__ method by splitting it into smaller methods.

from collections import defaultdict
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Optional, Union


class Logbook(list['dict[str, Union[int, float, dict[str, Union[int, float]]]]']):
    """
    Cronological records of the evolution.

    Contains the statistics of the evolution in a list of dictionaries.
    Each dictionary represents a new record with the statistics of the
    population at a given generation.

    Parameters
    ----------
    headers : list of str, default=None
        List of keys for the main header.

    chapter_headers : dict of str: list of str, default=None
        Dictionary with the name of the chapters as keys (may appear in the main header)
        and a list of keys for the headers of each chapter.
    """

    def __init__(
        self,
        headers: 'Optional[list[str]]' = None,
        chapter_headers: 'Optional[dict[str, list[str]]]' = None,
    ):
        self.buffer_index: int = 0
        self.headers = headers
        self.chapters: dict[str, Logbook] = defaultdict(Logbook)

        if chapter_headers is None:
            chapter_headers = {}

        self.chapter_headers = chapter_headers

        for key, value in chapter_headers.items():
            self.chapters[key].headers = value

        self._columns_len: list[int] = []

    def record(
        self,
        **info: 'Union[int, float, dict[str, Union[int, float]]]',
    ) -> None:
        apply_to_all = {key: value for key, value in info.items() if not isinstance(value, dict)}

        # Ensure all chapter headers are initialized
        self._initialize_chapter_headers(**info)

        for key, value in list(info.items()):
            if isinstance(value, dict):
                chapter_info = value.copy()
                chapter_info.update(apply_to_all)

                # Add default NaN values for missing keys in chapter headers
                if key in self.chapter_headers:
                    for subkey in self.chapter_headers[key]:
                        if subkey not in chapter_info:
                            chapter_info[subkey] = np.nan

                self.chapters[key].record(**chapter_info)
                del info[key]

        # Add default NaN values for missing keys in the main header
        if self.headers:
            for key in self.headers:
                if key not in info:
                    info[key] = np.nan

        super().append(info)

    def _initialize_chapter_headers(
        self,
        **info: 'Union[int, float, dict[str, Union[int, float]]]',
    ) -> None:
        for chapter_key in self.chapter_headers:
            if chapter_key not in info:
                info[chapter_key] = {}

    @property
    def stream(self) -> str:
        start_index, self.buffer_index = self.buffer_index, len(self)
        return self.__str__(start_index)

    def __txt__(self, start_index: int) -> list[str]:
        columns = self.headers

        if not columns:
            columns = sorted(self[0].keys()) + sorted(self.chapters.keys())

        if not self._columns_len or len(self._columns_len) != len(columns):
            self._columns_len = list(map(len, columns))

        chapters_txt, offsets = self._get_chapters_txt(start_index)
        str_matrix = self._create_str_matrix(start_index, columns, chapters_txt, offsets)

        if start_index == 0:
            str_matrix = self._add_headers(columns, chapters_txt, offsets, str_matrix)

        template = "    ".join("{%i:<%i}" % (i, col_l)
                               for i, col_l in enumerate(self._columns_len))
        return [template.format(*line) for line in str_matrix]

    def _get_chapters_txt(self, start_index: int) -> tuple[dict[str, list[str]], dict[str, int]]:
        chapters_txt: dict[str, list[str]] = {}
        offsets = defaultdict(int)

        for name, chapter in list(self.chapters.items()):
            chapters_txt[name] = chapter.__txt__(start_index)

            if start_index == 0:
                offsets[name] = len(chapters_txt[name]) - len(self)

        return chapters_txt, offsets

    def _create_str_matrix(
        self,
        start_index: int,
        columns: list[str],
        chapters_txt: dict[str, list[str]],
        offsets: dict[str, int],
    ) -> list[list[str]]:
        str_matrix: list[list[str]] = []

        for i, line in enumerate(self[start_index:]):
            str_line = []

            for j, name in enumerate(columns):
                if name in chapters_txt:
                    column = chapters_txt[name][i + offsets[name]]
                else:
                    value = line.get(name, "")
                    string = "{0:.5g}" if isinstance(value, float) else "{0}"
                    column = string.format(value)
                    column = column.ljust(10) if isinstance(value, float) else column

                self._columns_len[j] = max(self._columns_len[j], len(column))
                str_line.append(column)

            str_matrix.append(str_line)

        return str_matrix

    def _add_headers(
        self,
        columns: list[str],
        chapters_txt: dict[str, list[str]],
        offsets: dict[str, int],
        str_matrix: list[list[str]],
    ) -> list[list[str]]:
        nlines = 1

        if len(self.chapters) > 0:
            nlines += max(list(map(len, list(chapters_txt.values())))) - len(self) + 1

        header_lines: 'list[list[str]]' = [[] for i in range(nlines)]

        for j, name in enumerate(columns):
            if name in chapters_txt:
                length = max(len(line.expandtabs()) for line in chapters_txt[name])
                blanks = nlines - 2 - offsets[name]

                for i in range(blanks):
                    header_lines[i].append(" " * length)

                header_lines[blanks].append(name.center(length))
                header_lines[blanks + 1].append("-" * length)

                for i in range(offsets[name]):
                    header_lines[blanks + 2 + i].append(chapters_txt[name][i])
            else:
                length = max(len(line[j].expandtabs()) for line in str_matrix)

                for header_line in header_lines[:-1]:
                    header_line.append(" " * length)

                header_lines[-1].append(name)

        return list(chain(header_lines, str_matrix))

    def __str__(self, start_index: int = 0) -> str:
        text = self.__txt__(start_index)
        return "\n".join(text)
