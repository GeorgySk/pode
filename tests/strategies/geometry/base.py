from numbers import Real
from typing import TypeVar

from gon.discrete import Multipoint
from gon.linear import Contour
from gon.primitive import Point
from gon.shaped import Polygon
from hypothesis import strategies as st
from hypothesis_geometry import planar
from sect.triangulation import constrained_delaunay_triangles

from tests.pode_tests.config import (MAX_CONTOUR_SIZE,
                                     MAX_COORDINATE,
                                     MAX_HOLES_SIZE,
                                     MIN_CONTOUR_SIZE,
                                     MIN_COORDINATE,
                                     MIN_HOLES_SIZE,
                                     coordinates_strategies_factories)
from pode.utils import joined_constrained_delaunay_triangles

T = TypeVar('T')

TRIANGULAR_CONTOUR_SIZE = 3
RECTANGLE_CONTOUR_SIZE = 4

fractions = st.fractions(MIN_COORDINATE, MAX_COORDINATE)
raw_fraction_contours = planar.contours(fractions)
fractions_contours = st.builds(Contour.from_raw, raw_fraction_contours)

coordinates_strategies = st.sampled_from([
    factory(MIN_COORDINATE, MAX_COORDINATE)
    for factory in coordinates_strategies_factories.values()])


def coordinates_to_polygons(coordinates: st.SearchStrategy[Real]
                            ) -> st.SearchStrategy[Polygon]:
    return st.builds(Polygon.from_raw, planar.polygons(
        coordinates,
        min_size=MIN_CONTOUR_SIZE,
        max_size=MAX_CONTOUR_SIZE,
        min_holes_size=MIN_HOLES_SIZE,
        max_holes_size=MAX_HOLES_SIZE))


def coordinates_to_triangles(coordinates: st.SearchStrategy[Real]
                             ) -> st.SearchStrategy[Polygon]:
    return st.builds(Polygon.from_raw, planar.polygons(
        coordinates,
        max_size=TRIANGULAR_CONTOUR_SIZE))


def coordinates_to_multipoints(coordinates: st.SearchStrategy[Real]
                               ) -> st.SearchStrategy[Polygon]:
    return st.builds(Multipoint.from_raw, planar.multipoints(coordinates,
                                                             min_size=1))


polygons = coordinates_strategies.flatmap(coordinates_to_polygons)
fraction_triangles = st.builds(Polygon.from_raw, planar.polygons(
    fractions, max_size=TRIANGULAR_CONTOUR_SIZE))
multipoints = coordinates_strategies.flatmap(coordinates_to_multipoints)
_fraction_points = st.builds(Point, fractions, fractions)
unique_points_pairs = st.builds(tuple,
                                st.sets(_fraction_points,
                                        min_size=2,
                                        max_size=2))
convex_divisors = st.sampled_from([constrained_delaunay_triangles,
                                   joined_constrained_delaunay_triangles])
