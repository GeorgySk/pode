from typing import Tuple

from hypothesis import (given,
                        note)
from shapely.geometry import (LineString,
                              Polygon)

from geometry_utils import right_part
from tests.strategies import (
    convex_polygons_and_segments_not_passing_through_centroids)


@given(convex_polygons_and_segments_not_passing_through_centroids)
def test_area(polygon_and_segment: Tuple[Polygon, LineString]) -> None:
    polygon, segment = polygon_and_segment
    note(f"Polygon: {polygon.wkt}\n"
         f"LineString: {segment.wkt}")
    part = right_part(polygon, segment)
    assert part.area <= polygon.area 