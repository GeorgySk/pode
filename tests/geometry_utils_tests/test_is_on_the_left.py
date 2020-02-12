from typing import Tuple

import pytest
from hypothesis import (given,
                        note)
from shapely.geometry import (LineString,
                              Polygon)

from pode.geometry_utils import is_on_the_left
from tests.strategies import (
    empty_linestrings,
    empty_polygons,
    non_segments,
    nonempty_polygons,
    polygons_and_segments_not_passing_through_centroids,
    segments)


@given(polygons_and_segments_not_passing_through_centroids)
def test_opposite_lines(polygon_and_line: Tuple[Polygon, LineString]) -> None:
    polygon, line = polygon_and_line
    note(f"Geometry: {polygon.wkt}\n"
         f"LineString: {line.wkt}")
    reversed_line = LineString(line.coords[::-1])
    assert (is_on_the_left(polygon, line)
            is not is_on_the_left(polygon, reversed_line))


@given(polygon=nonempty_polygons,
       line=segments)
def test_crossing_centroid(polygon: Polygon,
                           line: LineString) -> None:
    line = LineString([polygon.centroid, *line.coords])
    note(f"Geometry: {polygon.wkt}\n"
         f"LineString: {line.wkt}")
    with pytest.raises(ValueError):
        is_on_the_left(polygon, line)


@given(polygon=nonempty_polygons,
       line=non_segments)
def test_non_segments(polygon: Polygon,
                      line: LineString) -> None:
    note(f"Geometry: {polygon.wkt}\n"
         f"LineString: {line.wkt}")
    with pytest.raises(ValueError):
        is_on_the_left(polygon, line)


@given(polygon=empty_polygons,
       line=segments)
def test_empty_geometry(polygon: Polygon,
                        line: LineString) -> None:
    note(f"Geometry: {polygon.wkt}\n"
         f"LineString: {line.wkt}")
    with pytest.raises(ValueError):
        is_on_the_left(polygon, line)


@given(polygon=nonempty_polygons,
       line=empty_linestrings)
def test_empty_line(polygon: Polygon,
                    line: LineString) -> None:
    note(f"Geometry: {polygon.wkt}\n"
         f"LineString: {line.wkt}")
    with pytest.raises(ValueError):
        is_on_the_left(polygon, line)
