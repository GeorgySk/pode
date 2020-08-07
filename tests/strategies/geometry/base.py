from numbers import Real
from typing import TypeVar

from gon.discrete import Multipoint
from gon.primitive import Point
from gon.shaped import Polygon
from hypothesis.strategies import (SearchStrategy,
                                   builds,
                                   fractions,
                                   sampled_from,
                                   sets)
from hypothesis_geometry import planar

from tests.pode_tests.config import (MAX_CONTOUR_SIZE,
                                     MAX_COORDINATE,
                                     MAX_HOLES_SIZE,
                                     MIN_CONTOUR_SIZE,
                                     MIN_COORDINATE,
                                     coordinates_strategies_factories)

T = TypeVar('T')

TRIANGULAR_CONTOUR_SIZE = 3
RECTANGLE_CONTOUR_SIZE = 4

coordinates_strategies = sampled_from([
    factory(MIN_COORDINATE, MAX_COORDINATE)
    for factory in coordinates_strategies_factories.values()])


def coordinates_to_polygons(coordinates: SearchStrategy[Real]
                            ) -> SearchStrategy[Polygon]:
    return builds(Polygon.from_raw, planar.polygons(
        coordinates,
        min_size=MIN_CONTOUR_SIZE,
        max_size=MAX_CONTOUR_SIZE,
        max_holes_size=MAX_HOLES_SIZE))


def coordinates_to_triangles(coordinates: SearchStrategy[Real]
                             ) -> SearchStrategy[Polygon]:
    return builds(Polygon.from_raw, planar.polygons(
        coordinates,
        max_size=TRIANGULAR_CONTOUR_SIZE))


def coordinates_to_multipoints(coordinates: SearchStrategy[Real]
                               ) -> SearchStrategy[Polygon]:
    return builds(Multipoint.from_raw, planar.multipoints(coordinates,
                                                          min_size=1))


polygons = coordinates_strategies.flatmap(coordinates_to_polygons)
fraction_triangles = builds(Polygon.from_raw, planar.polygons(
    fractions(MIN_COORDINATE, MAX_COORDINATE),
    max_size=TRIANGULAR_CONTOUR_SIZE))
multipoints = coordinates_strategies.flatmap(coordinates_to_multipoints)
_fraction_coordinates = fractions(MIN_COORDINATE, MAX_COORDINATE)
_fraction_points = builds(Point, _fraction_coordinates, _fraction_coordinates)
unique_points_pairs = builds(tuple,
                             sets(_fraction_points,
                                  min_size=2,
                                  max_size=2))
