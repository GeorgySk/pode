import operator
from functools import reduce
from itertools import combinations
from typing import (List,
                    Tuple)

from gon.base import (Point,
                      Polygon)
from hypothesis import (assume,
                        given)

from pode.hints import ConvexDivisorType
from pode.pode import to_graph
from tests.strategies.geometry.base import convex_divisors
from tests.strategies.geometry.composite import polygons_and_points


@given(polygon_and_points=polygons_and_points,
       convex_divisor=convex_divisors)
def test_polygon_reconstruction(
        polygon_and_points: Tuple[Polygon, List[Point]],
        convex_divisor: ConvexDivisorType) -> None:
    polygon, points = polygon_and_points
    graph = to_graph(polygon, points, convex_divisor=convex_divisor)
    union = reduce(operator.or_, graph)
    assert polygon == union


@given(polygon_and_points=polygons_and_points,
       convex_divisor=convex_divisors)
def test_parts_intersections(
        polygon_and_points: Tuple[Polygon, List[Point]],
        convex_divisor: ConvexDivisorType) -> None:
    polygon, points = polygon_and_points
    graph = to_graph(polygon, points, convex_divisor=convex_divisor)
    assert all(not isinstance(part & other, Polygon)
               for part, other in combinations(list(graph), 2))


@given(polygon_and_points=polygons_and_points,
       convex_divisor=convex_divisors)
def test_nodes_connections(
        polygon_and_points: Tuple[Polygon, List[Point]],
        convex_divisor: ConvexDivisorType) -> None:
    polygon, points = polygon_and_points
    graph = to_graph(polygon, points, convex_divisor=convex_divisor)
    is_not_triangle = len(polygon.border.vertices) > 3
    has_extra_points = any(point not in set(polygon.border.vertices)
                           for point in points)
    assume(is_not_triangle or has_extra_points or polygon.holes)
    assert all(graph[node] for node in graph)
