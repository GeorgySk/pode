import operator
from functools import reduce
from itertools import combinations
from typing import (List,
                    Tuple)

from gon.shaped import Polygon
from hypothesis import given
from sect.triangulation import constrained_delaunay_triangles

from pode.pode import (Site,
                       to_graph)
from tests.strategies.geometry.composite import polygons_and_sites


@given(polygons_and_sites)
def test_polygon_reconstruction(polygon_and_sites: Tuple[Polygon, List[Site]]
                                ) -> None:
    polygon, sites = polygon_and_sites
    graph = to_graph(polygon,
                     [site.location for site in sites],
                     constrained_delaunay_triangles)
    union = reduce(operator.or_, graph)
    assert polygon == union


@given(polygons_and_sites)
def test_parts_intersections(polygon_and_sites: Tuple[Polygon, List[Site]]
                             ) -> None:
    polygon, sites = polygon_and_sites
    graph = to_graph(polygon,
                     [site.location for site in sites],
                     constrained_delaunay_triangles)
    assert all(not isinstance(part & other, Polygon)
               for part, other in combinations(list(graph), 2))


@given(polygons_and_sites)
def test_nodes_connections(polygon_and_sites: Tuple[Polygon, List[Site]]
                           ) -> None:
    polygon, sites = polygon_and_sites
    graph = to_graph(polygon,
                     [site.location for site in sites],
                     constrained_delaunay_triangles)
    is_not_triangle = len(polygon.border.vertices) > 3
    has_extra_points = any(site.location not in set(polygon.border.vertices)
                           for site in sites)
    is_split_into_several_parts = (is_not_triangle or has_extra_points
                                   or polygon.holes)
    if is_split_into_several_parts:
        assert all(graph.nodes[node] for node in graph)
