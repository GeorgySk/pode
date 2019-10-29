from hypothesis import given
from shapely.geometry import Polygon
from shapely.ops import unary_union

from pode.geometry_utils import triangulation
from tests.strategies import polygons
from tests.utils import is_close


@given(polygons)
def test_area(polygon: Polygon) -> None:
    parts = triangulation(polygon)
    assert is_close(polygon.convex_hull.area, unary_union(parts).area)
