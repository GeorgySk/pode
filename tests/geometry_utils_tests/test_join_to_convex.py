from hypothesis import (given,
                        note)
from lz.iterating import capacity
from shapely.geometry import Polygon

from pode.geometry_utils import (join_to_convex,
                                 to_convex_parts)
from tests.strategies import (convex_polygons,
                              polygons)
from tests.utils import (no_trim_wkt,
                         is_close)


@given(convex_polygons)
def test_convex(polygon: Polygon) -> None:
    note(f"Polygon: {no_trim_wkt(polygon)}")
    parts = to_convex_parts(polygon)
    assert capacity(join_to_convex(parts)) == 1


@given(polygons)
def test_area(polygon: Polygon) -> None:
    note(f"Polygon: {no_trim_wkt(polygon)}")
    parts = to_convex_parts(polygon)
    large_parts = join_to_convex(parts)
    assert is_close(polygon.area, sum(part.area for part in large_parts))
