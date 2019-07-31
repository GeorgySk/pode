from functools import partial
from itertools import (chain,
                       combinations,
                       product,
                       starmap)
from math import isclose
from operator import itemgetter
from typing import (Callable,
                    Iterator,
                    Sequence,
                    Tuple)

import networkx as nx
from lz.functional import compose
from lz.iterating import pairwise
from shapely.geometry import (LineString,
                              LinearRing,
                              Polygon)
from shapely.ops import split

from pode.utils import starfilter


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


def to_graph(polygons: Sequence[Polygon]) -> nx.Graph:
    """
    Returns graph representation of input polygons.
    Edges are defined by polygons with touching sides.
    """
    graph = nx.Graph()
    for pair in neighbors(polygons):
        graph.add_edge(*pair)
    return graph


def neighbors(polygons: Sequence[Polygon]
              ) -> Iterator[Tuple[Polygon, Polygon]]:
    """Returns those polygons that have touching sides"""
    pairs = combinations(polygons, 2)
    yield from starfilter(side_touches, pairs)


def side_touches(polygon: Polygon,
                 other: Polygon) -> bool:
    """Determines if polygons have touching sides"""
    sides = segments(polygon.exterior)
    other_sides = segments(other.exterior)
    sides_combinations = product(sides, other_sides)
    return any(starmap(are_touching, sides_combinations))


def are_touching(segment: LineString,
                 other: LineString,
                 *,
                 buffer: float = 1e-10) -> bool:
    """
    Checks if two segments lie on the same line
    and touch in more than one point
    """
    # Buffering a line into a polygon to account for precision errors
    buffered_segment = segment.buffer(buffer)
    intersection = buffered_segment.intersection(other)
    return (isinstance(intersection, LineString)
            # check for lines touching initially in one point
            and intersection.length > buffer * 2)
