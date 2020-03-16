from hypothesis import (given,
                        note)
from shapely.geometry import Polygon

from pode.geometry_utils import to_convex_parts
from tests.strategies import area_greater_than_tolerance_polygons
from tests.utils import (is_close,
                         no_trim_wkt)


@given(area_greater_than_tolerance_polygons)
def test_quantity(polygon: Polygon) -> None:
    note(f'Polygon: {no_trim_wkt(polygon)}')
    parts = to_convex_parts(polygon)
    assert len(list(parts)) > 0


@given(area_greater_than_tolerance_polygons)
def test_area(polygon: Polygon) -> None:
    note(f'Polygon: {no_trim_wkt(polygon)}')
    parts = to_convex_parts(polygon)
    assert is_close(polygon.area, sum(part.area for part in parts))
