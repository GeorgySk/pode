from itertools import (chain,
                       combinations,
                       product)
from operator import itemgetter
from typing import (Callable,
                    Iterable,
                    Iterator,
                    List,
                    Optional,
                    Sequence,
                    Tuple)

import networkx as nx
import numpy as np
from lz.functional import compose
from lz.iterating import (flatten,
                          pairwise)
from matplotlib.tri.triangulation import Triangulation
from shapely.geometry import (LineString,
                              LinearRing,
                              Polygon)
from shapely.geometry.base import BaseGeometry
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


# TODO: add overlapping check
def is_on_the_right(geometry: BaseGeometry,
                    line: LineString) -> bool:
    """
    Determines if the geometry object is on the right side of the line
    according to:
    https://stackoverflow.com/questions/50393718/determine-the-left-and-right-side-of-a-split-shapely-geometry
    Doesn't work for 3D geometries:
    https://github.com/Toblerity/Shapely/issues/709
    """
    ring = LinearRing(chain(line.coords, geometry.centroid.coords))
    return not ring.is_ccw


def safe_split(polygon: Polygon,
               line: LineString) -> Tuple[Polygon, Polygon]:
    parts = tuple(split(polygon, line))
    if len(parts) == 1:
        return polygon, Polygon()
    return parts


def midpoint(line: LineString) -> Tuple[float, float]:
    """Returns coordinates of the middle of the input line"""
    return line.interpolate(0.5, normalized=True).coords[0]


def to_graph(polygons: Sequence[Polygon]) -> nx.Graph:
    """
    Returns graph representation of input polygons.
    Edges are defined by polygons with touching sides.
    Nodes have attributes with information on touching sides.
    """
    graph = nx.Graph()
    if len(polygons) == 1:
        graph.add_nodes_from(polygons)
        return graph
    for pair, sides in neighbors_and_sides(polygons):
        polygon, other_polygon = pair
        side, other_side = sides
        graph.add_edge(polygon, other_polygon, side=side)
    return graph


def neighbors_and_sides(polygons: Iterable[Polygon]
                        ) -> Iterator[Tuple[Tuple[Polygon, Polygon],
                                            Tuple[LineString, LineString]]]:
    """Returns those polygons that have touching sides"""
    pairs = combinations(polygons, 2)
    for pair in pairs:
        sides = touching_sides(*pair)
        if sides is not None:
            yield pair, sides


def touching_sides(polygon: Polygon,
                   other: Polygon) -> Optional[Tuple[LineString, LineString]]:
    """Returns the first found touching sides of two polygons"""
    sides = segments(polygon.exterior)
    other_sides = segments(other.exterior)
    sides_combinations = product(sides, other_sides)
    return next(starfilter(are_touching, sides_combinations), None)


def are_touching(segment: LineString,
                 other: LineString,
                 *,
                 buffer: float = 1e-7) -> bool:
    """
    Checks if two segments lie on the same line
    and touch in more than one point.
    Buffer values less than 1e-8 can lead to errors
    """
    if isinstance(segment.intersection(other), LineString):
        return True
    a, b = segment.boundary
    c, d = other.boundary
    distances = [a.distance(other), b.distance(other),
                 c.distance(segment), d.distance(segment)]
    close_points_count = sum(distance <= buffer for distance in distances)
    if any(point in other.boundary for point in segment.boundary):
        return close_points_count >= 3
    else:
        return close_points_count >= 2


def to_convex_parts(polygon: Polygon) -> Iterator[Polygon]:
    """
    Splits a polygon to convex parts.
    Implemented by simple Delaunay triangulation.
    """
    yield from filter(polygon.contains, triangulation(polygon))


def triangulation(polygon: Polygon) -> List[Polygon]:
    """
    This is an alternative to Shapely's ops.triangulate function.
    As due to some bugs it can return wrong results sometimes,
    we need another implementation.
    Polygons to reproduce the issues:
        empty list:
            Polygon([(0, 0), (1, 0), (-1, 0.05)])
        missing part:
            Polygon([(0, 0), (0, 486), (1, 486), (1, 22), (2, 22), (2, 0)])
    Bug is submitted to GitHub.
    """
    exterior_coords_set = set(polygon.exterior.coords)
    interiors_coords_sequences = map(LinearRing.coords.fget, polygon.interiors)
    interiors_coords_set = set(flatten(interiors_coords_sequences))
    coords_set = exterior_coords_set | interiors_coords_set
    coords_set = np.array([*coords_set])
    xy = coords_set.T
    triangulation_object = Triangulation(*xy)
    triangles_points_indices = triangulation_object.triangles
    triangles_coords = coords_set[triangles_points_indices]
    return list(map(Polygon, triangles_coords))
