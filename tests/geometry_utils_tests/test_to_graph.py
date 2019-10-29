from typing import (List,
                    Tuple)

from hypothesis import (given,
                        note)
from lz.functional import compose
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from pode.geometry_utils import to_graph
from tests.strategies import (disjoint_polygons_lists,
                              polygons_grids_and_dimensions)
from tests.utils import no_trim_wkt

# The following monkey-patch should be removed
# when Shapely will make the geometries hashable again
# (should be in Shapely 2.0).
# Currently I use wkb representation to get hashes,
# but it means that, for example, two LineString's consisting of the same
# points but with different directions will have different hash
BaseGeometry.__hash__ = Polygon.__hash__ = compose(hash, BaseGeometry.wkb.fget)


@given(disjoint_polygons_lists)
def test_disjoint(polygons: List[Polygon]) -> None:
    note(f'Polygons: {list(map(no_trim_wkt, polygons))}')
    graph = to_graph(polygons)
    assert not graph.edges


@given(polygons_grids_and_dimensions)
def test_grids(polygon_grid_and_shape: Tuple[List[Polygon], Tuple[int, int]]
               ) -> None:
    polygons, (rows_count, columns_count) = polygon_grid_and_shape
    note(f'Polygons: {list(map(no_trim_wkt, polygons))}')
    polygons_count = len(polygons)
    graph = to_graph(polygons)
    if polygons_count < 2:
        assert not graph.edges
    else:
        assert len(graph.nodes) == polygons_count
        assert len(graph.edges) == (2 * rows_count * columns_count
                                    - rows_count - columns_count)
