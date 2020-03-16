from typing import Tuple

import pytest
from hypothesis import (given,
                        note)
from shapely.geometry import (LineString,
                              Polygon)

from pode.geometry_utils import right_part
from tests.configs import ABS_TOL
from tests.strategies import (
    convex_polygons_and_segments_not_passing_through_centroids,
    empty_polygons,
    non_segments,
    nonempty_polygons,
    segments)


@given(convex_polygons_and_segments_not_passing_through_centroids)
def test_area(polygon_and_segment: Tuple[Polygon, LineString]) -> None:
    polygon, segment = polygon_and_segment
    note(f"Polygon: {polygon.wkt}\n"
         f"LineString: {segment.wkt}")
    part = right_part(polygon, segment)
    assert part.area <= polygon.area + ABS_TOL


@given(polygon=empty_polygons,
       segment=segments)
def test_empty_polygons(polygon: Polygon,
                        segment: LineString) -> None:
    with pytest.raises(ValueError):
        right_part(polygon, segment)


@given(polygon=nonempty_polygons,
       line=non_segments)
def test_nonsegments(polygon: Polygon,
                     line: LineString) -> None:
    with pytest.raises(ValueError):
        right_part(polygon, line)
