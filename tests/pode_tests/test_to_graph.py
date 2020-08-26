import operator
from functools import reduce
from itertools import combinations
from typing import (List,
                    Tuple)

from gon.shaped import Polygon
from hypothesis import (assume,
                        given)

from pode.hints import ConvexDivisorType
from pode.pode import (Site,
                       to_graph)
from tests.strategies.geometry.base import convex_divisors
from tests.strategies.geometry.composite import polygons_and_sites


@given(polygon_and_sites=polygons_and_sites,
       convex_divisor=convex_divisors)
def test_polygon_reconstruction(polygon_and_sites: Tuple[Polygon, List[Site]],
                                convex_divisor: ConvexDivisorType) -> None:
    polygon, sites = polygon_and_sites
    graph = to_graph(polygon,
                     [site.location for site in sites],
                     convex_divisor=convex_divisor)
    union = reduce(operator.or_, graph)
    assert polygon == union


@given(polygon_and_sites=polygons_and_sites,
       convex_divisor=convex_divisors)
def test_parts_intersections(polygon_and_sites: Tuple[Polygon, List[Site]],
                             convex_divisor: ConvexDivisorType) -> None:
    polygon, sites = polygon_and_sites
    graph = to_graph(polygon,
                     [site.location for site in sites],
                     convex_divisor=convex_divisor)
    assert all(not isinstance(part & other, Polygon)
               for part, other in combinations(list(graph), 2))


@given(polygon_and_sites=polygons_and_sites,
       convex_divisor=convex_divisors)
def test_nodes_connections(polygon_and_sites: Tuple[Polygon, List[Site]],
                           convex_divisor: ConvexDivisorType) -> None:
    polygon, sites = polygon_and_sites
    graph = to_graph(polygon,
                     [site.location for site in sites],
                     convex_divisor=convex_divisor)
    is_not_triangle = len(polygon.border.vertices) > 3
    has_extra_points = any(site.location not in set(polygon.border.vertices)
                           for site in sites)
    assume(is_not_triangle or has_extra_points or polygon.holes)
    assert all(graph[node] for node in graph)
