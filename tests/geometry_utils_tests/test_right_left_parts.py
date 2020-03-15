from math import isclose
from typing import Tuple

from hypothesis import (given,
                        note)
from shapely.geometry import (LineString,
                              Polygon)

from pode.geometry_utils import right_left_parts
from tests.strategies import (
    convex_polygons_and_segments_not_passing_through_centroids)


@given(convex_polygons_and_segments_not_passing_through_centroids)
def test_area(polygon_and_segment: Tuple[Polygon, LineString]) -> None:
    polygon, segment = polygon_and_segment
    note(f"Polygon: {polygon.wkt}\n"
         f"LineString: {segment.wkt}")
    part, other_part = right_left_parts(polygon, segment)
    assert isclose(polygon.area,
                   part.area + other_part.area,
                   rel_tol=1e-08)  # smaller tolerance can result in errors
