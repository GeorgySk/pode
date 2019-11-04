from hypothesis import (given,
                        note)
from shapely.geometry import Polygon

from pode.geometry_utils import is_convex
from tests.strategies import (convex_polygons,
                              nonconvex_polygons)
from tests.utils import no_trim_wkt


@given(convex_polygons)
def test_convex(polygon: Polygon) -> None:
    note(f"Polygon: {no_trim_wkt(polygon)}")
    assert is_convex(polygon)


@given(nonconvex_polygons)
def test_nonconvex(polygon: Polygon) -> None:
    note(f"Polygon: {no_trim_wkt(polygon)}")
    assert not is_convex(polygon)
