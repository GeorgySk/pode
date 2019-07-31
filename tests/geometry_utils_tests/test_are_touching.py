from typing import Tuple

from hypothesis import (given,
                        note)
from shapely.geometry import LineString

from pode.geometry_utils import are_touching
from tests.strategies import (disjoint_lines,
                              lines_one_in_another,
                              overlapping_lines,
                              touching_lines)


@given(lines_one_in_another)
def test_one_in_another(lines: Tuple[LineString, LineString]) -> None:
    note(f'Line 1: {lines[0]}\n'
         f'Line 2: {lines[1]}')
    assert are_touching(*lines)


@given(overlapping_lines)
def test_overlapping(lines: Tuple[LineString, LineString]) -> None:
    note(f'Line 1: {lines[0]}\n'
         f'Line 2: {lines[1]}')
    assert are_touching(*lines)


@given(touching_lines)
def test_touching(lines: Tuple[LineString, LineString]) -> None:
    note(f'Line 1: {lines[0]}\n'
         f'Line 2: {lines[1]}')
    assert not are_touching(*lines)


@given(disjoint_lines)
def test_disjoint(lines: Tuple[LineString, LineString]) -> None:
    note(f'Line 1: {lines[0]}\n'
         f'Line 2: {lines[1]}')
    assert not are_touching(*lines)
