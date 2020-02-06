import math
from functools import (partial,
                       singledispatch)
from itertools import (chain,
                       combinations,
                       starmap)
from typing import (Iterable,
                    Iterator,
                    Tuple)

from lz.iterating import flatten
from shapely import wkt
from shapely.geometry import (GeometryCollection,
                              Point,
                              Polygon)
from shapely.geometry.base import (BaseGeometry,
                                   BaseMultipartGeometry)

from tests.configs import ABS_TOL

is_close = partial(math.isclose,
                   abs_tol=ABS_TOL)

no_trim_wkt = partial(wkt.dumps, trim=False)


@singledispatch
def to_coords(geometry: BaseGeometry) -> Iterator[Tuple[float, float]]:
    yield from geometry.coords


@to_coords.register(Polygon)
def _(geometry) -> Iterator[Tuple[float, float]]:
    interiors_coords_sequences = map(to_coords, geometry.interiors)
    interiors_coords = flatten(interiors_coords_sequences)
    exterior_coords = to_coords(geometry.exterior)
    yield from chain(interiors_coords, exterior_coords)


@to_coords.register(BaseMultipartGeometry)
def _(geometry) -> Iterator[Tuple[float, float]]:
    coords_sequences = map(to_coords, geometry.geoms)
    yield from flatten(coords_sequences)


def has_no_close_points(geometry: BaseGeometry,
                        *,
                        abs_tol: float = ABS_TOL) -> bool:
    coords = set(to_coords(geometry))
    points = map(Point, coords)
    distances = starmap(Point.distance, combinations(points, 2))
    return all(distance > abs_tol for distance in distances)


def have_no_close_points(geometries: Iterable[BaseGeometry],
                         *,
                         abs_tol: float = ABS_TOL) -> bool:
    collection = GeometryCollection(list(geometries))
    return has_no_close_points(collection, abs_tol=abs_tol)


def are_sparse(geometries: Iterable[BaseGeometry],
               *,
               abs_tol: float = ABS_TOL) -> bool:
    """
    Checks if all geometries are not too close to each other.
    Used to reject geometries that can lead to precision errors.
    """
    pairs = combinations(geometries, 2)
    distances = starmap(BaseGeometry.distance, pairs)
    return all(distance > abs_tol for distance in distances)
