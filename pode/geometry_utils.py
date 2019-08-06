from itertools import (chain,
                       combinations,
                       product,
                       starmap)
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
                 buffer: float = 1e-8) -> bool:
    """
    Checks if two segments lie on the same line
    and touch in more than one point.
    Buffer values less than 1e-8 can lead to errors
    """
    if isinstance(segment.intersection(other), LineString):
        return True
    # lines are touching only in one point
    if any(map(other.boundary.contains, segment.boundary)):
        return False
    # Buffering a line into a polygon to account for precision errors
    left_side_offset = line_offset(segment, buffer, side='left')
    right_side_offset = line_offset(segment, buffer, side='right')
    buffered_segment = Polygon([*left_side_offset.coords,
                                *right_side_offset.coords])
    intersection = buffered_segment.intersection(other)
    return (isinstance(intersection, LineString)
            # check for lines touching initially in one point
            and intersection.length > buffer)


def line_offset(line: LineString,
                distance: float,
                *,
                side: str) -> LineString:
    """
    This is a wrapper for Shapely's LineString.parallel_offset method.
    As due to GEOS bugs it can produce MultiLineString's sometimes,
    we need to construct the offset manually in such cases.
    Bug report: https://github.com/Toblerity/Shapely/issues/746
    """
    if len(line.coords) > 2:
        raise ValueError('Can offset only lines consisting of 2 points')

    result = line.parallel_offset(distance, side=side)
    if isinstance(result, LineString) and len(result.coords) == 2:
        return result

    dy = line.boundary[1].y - line.boundary[0].y
    dx = line.boundary[1].x - line.boundary[0].x
    direction = 1 if side == 'left' else -1
    if dx == 0:
        x_offset = distance
        y_offset = 0
    elif dy == 0:
        x_offset = 0
        y_offset = distance
    else:
        m = dy / dx
        # based on solution of the following equations system:
        # 𝛿y = -𝛿x * (1 / m)  # slope of orthogonal line
        # 𝛿x**2 + 𝛿y**2 = distance**2
        x_offset = direction * distance / (1 + 1 / m ** 2) ** 0.5
        y_offset = -distance / (1 + m ** 2) ** 0.5
    # reorienting for 'right' as in LineString.parallel_offset
    return LineString([(line.boundary[0].x + x_offset,
                        line.boundary[0].y + y_offset),
                       (line.boundary[1].x + x_offset,
                        line.boundary[1].y + y_offset)][::direction])
