from itertools import chain
from operator import itemgetter
from typing import (Callable,
                    Iterator,
                    Tuple)

from lz.functional import compose
from lz.iterating import pairwise
from shapely.geometry import (LineString,
                              LinearRing,
                              Polygon)
from shapely.ops import split


def segments(ring: LinearRing) -> Iterator[LineString]:
    """Yields consecutive lines from the given ring"""
    pairs = pairwise(ring.coords)
    yield from map(LineString, pairs)


def right_left_parts(polygon: Polygon,
                     line: LineString) -> Tuple[Polygon, Polygon]:
    """Splits polygon by a line and returns two parts: right and left"""
    part, other_part = safe_split(polygon, line)
    if is_on_the_left(part, line):
        return other_part, part
    return part, other_part


right_part: Callable[[Polygon, LineString], Polygon] = compose(
    itemgetter(0), right_left_parts)
right_part.__doc__ = "Splits polygon by a line and returns the right part"


def is_on_the_left(polygon: Polygon,
                   line: LineString) -> bool:
    """
    Determines if the polygon is on the left side of the line
    according to:
    https://stackoverflow.com/questions/50393718/determine-the-left-and-right-side-of-a-split-shapely-geometry
    Doesn't work for 3D geometries:
    https://github.com/Toblerity/Shapely/issues/709
    """
    ring = LinearRing(chain(line.coords, polygon.centroid.coords))
    return ring.is_ccw


def safe_split(polygon: Polygon,
               line: LineString) -> Tuple[Polygon, Polygon]:
    parts = tuple(split(polygon, line))
    if len(parts) == 1:
        return parts[0], Polygon()
    return parts


def midpoint(line: LineString) -> Tuple[float, float]:
    """Returns coordinates of the middle of the input line"""
    return line.interpolate(0.5, normalized=True).coords[0]
