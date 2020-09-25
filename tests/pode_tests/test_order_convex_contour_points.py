from typing import List

from gon.primitive import Point
from hypothesis import given

from tests.utils import cyclic_equivalence
from tests.strategies.geometry.composite import convex_contour_points
from utils import order_convex_contour_points


@given(convex_contour_points)
def test_rotation(points: List[Point]) -> None:
    ordered_points = order_convex_contour_points(points)
    assert cyclic_equivalence(points, ordered_points)
