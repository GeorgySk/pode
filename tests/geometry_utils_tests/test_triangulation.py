from hypothesis import (given,
                        note)
from shapely.geometry import Polygon
from shapely.ops import unary_union

from pode.geometry_utils import triangulation
from tests.strategies import nonempty_polygons
from tests.utils import (is_close,
                         no_trim_wkt)


@given(nonempty_polygons)
def test_area(polygon: Polygon) -> None:
    note(f'Polygon: {no_trim_wkt(polygon)}')
    parts = triangulation(polygon)
    assert is_close(polygon.convex_hull.area, unary_union(parts).area)
