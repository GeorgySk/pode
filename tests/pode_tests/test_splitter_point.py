from fractions import Fraction

from gon.linear import (Contour,
                        Segment)
from gon.shaped import Polygon
from hypothesis import (assume,
                        given)
from hypothesis.strategies import (booleans,
                                   fractions)

from pode.pode import splitter_point
from tests.strategies.geometry.base import fraction_triangles


@given(triangle=fraction_triangles,
       is_counterclockwise=booleans(),
       fraction=fractions(0, 1))
def test_area(triangle: Polygon,
              is_counterclockwise: bool,
              fraction: Fraction) -> None:
    assume(0 < fraction < 1)
    requirement = triangle.area * fraction
    if not is_counterclockwise:
        triangle = Polygon(triangle.border.to_clockwise())
    pivot, low_area_point, high_area_point = triangle.border.vertices
    new_point = splitter_point(requirement=requirement,
                               pivot=pivot,
                               low_area_point=low_area_point,
                               high_area_point=high_area_point)
    new_triangle = Polygon(Contour((pivot, low_area_point, new_point)))
    assert new_triangle.area == requirement


@given(triangle=fraction_triangles,
       is_counterclockwise=booleans(),
       fraction=fractions(0, 1))
def test_point(triangle: Polygon,
               is_counterclockwise: bool,
               fraction: Fraction) -> None:
    assume(0 < fraction < 1)
    requirement = triangle.area * fraction
    if not is_counterclockwise:
        triangle = Polygon(triangle.border.to_clockwise())
    pivot, low_area_point, high_area_point = triangle.border.vertices
    new_point = splitter_point(requirement=requirement,
                               pivot=pivot,
                               low_area_point=low_area_point,
                               high_area_point=high_area_point)
    assert new_point in Segment(low_area_point, high_area_point)
