from typing import Tuple

from gon.linear import (Contour,
                        Segment)
from gon.primitive import Point
from hypothesis import given

from pode.utils import cut
from tests.strategies.geometry.composite import contours_and_points


@given(contours_and_points)
def test_vertices(contour_and_points: Tuple[Contour, Point, Point]) -> None:
    contour, start, end = contour_and_points
    assert all(point in contour for point in cut(contour, start, end))


@given(contours_and_points)
def test_segments(contour_and_points: Tuple[Contour, Point, Point]) -> None:
    contour, start, end = contour_and_points
    vertices = cut(contour, start, end)
    segments = map(Segment, vertices[:-1], vertices[1:])
    assert all(segment < contour for segment in segments)


@given(contours_and_points)
def test_duplicates(contour_and_points: Tuple[Contour, Point, Point]) -> None:
    contour, start, end = contour_and_points
    vertices = cut(contour, start, end)
    assert len(set(vertices)) == len(vertices)


@given(contours_and_points)
def test_both_directions(contour_and_points: Tuple[Contour, Point, Point]
                         ) -> None:
    contour, start, end = contour_and_points
    right_vertices = cut(contour, start, end)
    left_vertices = cut(contour, end, start)
    assert {*right_vertices, *left_vertices} == {*contour.vertices, start, end}
