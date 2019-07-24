from operator import itemgetter

import pytest
from hypothesis import (given,
                        note)
from lz.functional import compose
from lz.iterating import capacity
from shapely.geometry import (LinearRing,
                              LineString)

from pode.geometry_utils import segments
from tests.strategies import linear_rings


@given(linear_rings)
def test_numbers_of_points_in_each_segment(ring: LinearRing) -> None:
    note(f"Ring: {ring.wkt}")
    assert all(len(line.coords) == 2 for line in segments(ring))


@given(linear_rings)
def test_number_of_segments(ring: LinearRing) -> None:
    note(f"Ring: {ring.wkt}")
    assert capacity(segments(ring)) == len(ring.coords) - 1


@given(linear_rings)
def test_endpoints(ring: LinearRing) -> None:
    lines = list(segments(ring))
    note(f"Ring: {ring.wkt}")
    assert lines[0].coords[0] == lines[-1].coords[-1]


@pytest.mark.skip(reason="Upstream bug for LinearRing.equals. "
                         "Fails for rings with overlapping collinear segments")
@given(linear_rings)
def test_recreation(ring: LinearRing) -> None:
    lines = list(segments(ring))
    first_coordinate = compose(itemgetter(0), LineString.coords.fget)
    first_points = map(first_coordinate, lines)
    recreated_ring = LinearRing(first_points)
    note(f"Ring: {ring.wkt}")
    assert ring.equals(recreated_ring)


@given(linear_rings)
def test_points_equality(ring: LinearRing) -> None:
    lines = list(segments(ring))
    first_coordinate = compose(itemgetter(0), LineString.coords.fget)
    first_points = list(map(first_coordinate, lines))
    note(f"Ring: {ring.wkt}")
    assert first_points == ring.coords[:-1]