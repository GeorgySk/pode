from typing import Tuple

from gon.base import (Multipoint,
                      Segment)
from hypothesis import given

from pode.pode import order_by_edge
from tests.strategies.geometry.composite import multipoints_and_segments


@given(multipoints_and_segments)
def test_last_point(multipoint_and_edge: Tuple[Multipoint, Segment]) -> None:
    multipoint, edge = multipoint_and_edge
    vertices = order_by_edge(multipoint.points, edge)
    assert vertices[-1] in edge
