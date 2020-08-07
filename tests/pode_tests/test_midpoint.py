from typing import Tuple

from gon.linear import Segment
from gon.primitive import Point
from hypothesis import given

from pode.pode import midpoint
from tests.strategies.geometry.base import unique_points_pairs


@given(unique_points_pairs)
def test_inclusion(points_pair: Tuple[Point, Point]) -> None:
    assert midpoint(*points_pair) in Segment(*points_pair)
