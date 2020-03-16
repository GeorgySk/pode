from math import isclose
from typing import Tuple

import pytest
from hypothesis import (given,
                        note)
from shapely.geometry import (LineString,
                              Polygon)

from pode.geometry_utils import right_left_parts
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
    part, other_part = right_left_parts(polygon, segment)
    assert isclose(polygon.area,
                   part.area + other_part.area,
                   rel_tol=1e-08)  # smaller tolerance can result in errors


@given(convex_polygons_and_segments_not_passing_through_centroids)
def test_intersection(polygon_and_segment: Tuple[Polygon, LineString]) -> None:
    polygon, segment = polygon_and_segment
    note(f"Polygon: {polygon.wkt}\n"
         f"LineString: {segment.wkt}")
    part, other_part = right_left_parts(polygon, segment)
    assert any(small_part.intersects(polygon)
               for small_part in (part, other_part))


@given(polygon=empty_polygons,
       segment=segments)
def test_empty_polygons(polygon: Polygon,
                        segment: LineString) -> None:
    note(f"Polygon: {polygon.wkt}\n"
         f"LineString: {segment.wkt}")
    with pytest.raises(ValueError):
        right_left_parts(polygon, segment)


@given(polygon=nonempty_polygons,
       line=non_segments)
def test_nonsegments(polygon: Polygon,
                     line: LineString) -> None:
    note(f"Polygon: {polygon.wkt}\n"
         f"LineString: {line.wkt}")
    with pytest.raises(ValueError):
        right_left_parts(polygon, line)
