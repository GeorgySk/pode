from math import isclose

import pytest
from hypothesis import (given,
                        note)
from shapely.geometry import (LineString,
                              Point)

from pode.geometry_utils import midpoint
from tests.strategies import (empty_linestrings,
                              segments)
from tests.utils import no_trim_wkt


@given(segments)
def test_distance(line: LineString) -> None:
    note(f'Line: {no_trim_wkt(line)}')
    point = Point(midpoint(line))
    # It is not clear how to calculate the upper boundary for abs_tol
    # but values less than 1e-13 can fail the test
    assert isclose(line.distance(point), 0, abs_tol=1e-13)


@given(segments)
def test_lengths(line: LineString) -> None:
    note(f'Line: {no_trim_wkt(line)}')
    point = Point(midpoint(line))
    half = LineString([line.boundary[0], point])
    other = LineString([point, line.boundary[1]])
    assert isclose(half.length, other.length)


@given(empty_linestrings)
def test_empty(line: LineString) -> None:
    with pytest.raises(ValueError):
        list(midpoint(line))
