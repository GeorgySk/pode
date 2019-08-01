import math
from functools import partial
from itertools import (combinations,
                       starmap,
                       chain)
from typing import Iterable

from shapely.geometry import (Point,
                              LineString)

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
