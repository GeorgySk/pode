import math
from functools import partial
from itertools import (chain,
                       combinations,
                       starmap)
from typing import (Iterable,
                    Tuple)

from shapely.geometry import (LineString,
                              Point,
                              Polygon)

from tests.configs import ABS_TOL

is_close = partial(math.isclose,
                   abs_tol=ABS_TOL)


def points_are_sparse(lines: Iterable[LineString],
                      *,
                      abs_tol: float = ABS_TOL) -> bool:
    """
    Checks each pair of points in lines if they are too close.
    Used to reject geometries that can lead to precision errors.
    """
    coordinates = chain.from_iterable(map(LineString.coords.fget, lines))
    points = map(Point, set(coordinates))
    distances = starmap(Point.distance, combinations(points, 2))
    return all(distance > abs_tol for distance in distances)


def polygons_are_sparse(polygon: Polygon,
                        other: Polygon,
                        *,
                        abs_tol: float = ABS_TOL) -> bool:
    """
    Checks if pair of polygons are too close.
    Used to reject geometries that can lead to precision errors.
    """
    return polygon.distance(other) > abs_tol


def form_object_with_area(vertices: Iterable[Tuple[float, float]]) -> bool:
    return Polygon(vertices).area > 0
