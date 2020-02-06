from itertools import (chain,
                       combinations,
                       product)
from math import (atan2,
                  tan)
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
from shapely.affinity import (rotate,
                              scale,
                              translate)
from shapely.geometry import (GeometryCollection,
                              LineString,
                              LinearRing,
                              Point,
                              Polygon,
                              MultiPolygon)
from shapely.geometry.base import BaseGeometry
from shapely.ops import split

from pode.utils import (next_index,
                        starfilter)


def segments(ring: LinearRing) -> Iterator[LineString]:
    """
    Yields consecutive lines from the given ring.
    Doesn't work for degenerate geometries.
    """
    pairs = pairwise(ring.coords)
    yield from map(LineString, pairs)


def right_left_parts(polygon: Polygon,
                     line: LineString) -> Tuple[Polygon, Polygon]:
    """Splits polygon by a line and returns two parts: right and left"""
    if len(line.coords) != 2:
        raise ValueError("Only lines consisting of 2 points are supported")
    part, other_part = safe_split(polygon, line)
    # sometimes due to precision errors a site point lying on a polygon's
    # segment can be lying a bit inside of it which will make the
    # polygon nonconvex:
    if (part.area < 1e-16 and is_on_the_left(part, line)
            and is_on_the_left(other_part, line)):
        return part, other_part
    if is_on_the_left(part, line):
        return other_part, part
    return part, other_part


right_part: Callable[[Polygon, LineString], Polygon] = compose(
    itemgetter(0), right_left_parts)
right_part.__doc__ = "Splits polygon by a line and returns the right part"


def is_on_the_left(geometry: BaseGeometry,
                   line: LineString) -> bool:
    """
    Determines if the geometry is on the left side of the line
    according to:
    https://stackoverflow.com/questions/50393718/determine-the-left-and-right-side-of-a-split-shapely-geometry
    Doesn't work for 3D geometries:
    https://github.com/Toblerity/Shapely/issues/709
    """
    ring = LinearRing(chain(line.coords, geometry.centroid.coords))
    return ring.is_ccw


def safe_split(polygon: Polygon,
               line: LineString) -> Tuple[Polygon, Polygon]:
    """
    Splits the polygon by the given line and returns resulting parts.
    If splitting resulted in only one polygon, it is returned first,
    and an empty polygon is returned after it.
    """
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
    parts = triangulation(polygon)
    intersections = map(polygon.intersection, parts)
    for intersection in intersections:
        if isinstance(intersection, Polygon):
            yield intersection
        elif isinstance(intersection, (GeometryCollection, MultiPolygon)):
            for part in intersection.geoms:
                if isinstance(part, Polygon):
                    yield part


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


def join_to_convex(polygons: Iterable[Polygon]) -> Iterator[Polygon]:
    """Joins polygons to form convex parts of greater size"""
    polygons = list(polygons)
    initial_polygon = polygons.pop()
    while True:
        resulting_polygon = initial_polygon
        for index, polygon in enumerate(iter(polygons)):
            union = resulting_polygon.union(polygon)
            if isinstance(union, Polygon) and is_convex(union):
                polygons.pop(index)
                resulting_polygon = union
        if resulting_polygon is not initial_polygon:
            initial_polygon = resulting_polygon
            continue
        yield resulting_polygon
        if not polygons:
            return
        initial_polygon = polygons.pop()


def is_convex(polygon: Polygon) -> bool:
    return not polygon.interiors and polygon.convex_hull.equals(polygon)


def slope_intercept(line: LineString) -> Tuple[float, float]:
    slope = slope_angle(line)
    start, end = line.boundary
    intercept = start.y - tan(slope) * start.x
    return slope, intercept


def slope_angle(line: LineString) -> float:
    start, end = line.boundary
    return atan2(end.y - start.y, end.x - start.x)


def rotating_splitter(fraction: float,
                      source_line: LineString,
                      target_line: LineString,
                      length: float) -> LineString:
    scale_factor_by_axis = length / source_line.length
    source_line_slope, source_line_intercept = slope_intercept(source_line)
    target_line_slope, target_line_intercept = slope_intercept(target_line)
    line = scale(source_line,
                 xfact=scale_factor_by_axis,
                 yfact=scale_factor_by_axis)
    if source_line_slope == target_line_slope:
        source_line_center = source_line.centroid
        target_line_center = target_line.centroid
        dx = (target_line_center.x - source_line_center.x) * fraction
        dy = (target_line_center.y - source_line_center.y) * fraction
        return translate(line,
                         xoff=dx,
                         yoff=dy)
    # checking if the slopes have different signs in order to get a correct
    # splitter
    resulting_slope = (source_line_slope + fraction * (target_line_slope
                                                       - source_line_slope))
    if source_line_slope * target_line_slope < 0 and resulting_slope != 0:
        resulting_slope = - 1 / resulting_slope
    rotation_origin = rays_intersection(line_slope=tan(source_line_slope),
                                        line_intercept=source_line_intercept,
                                        other_slope=tan(target_line_slope),
                                        other_intercept=target_line_intercept)
    return rotate(line,
                  angle=resulting_slope - source_line_slope,
                  origin=rotation_origin,
                  use_radians=True)


def rays_intersection(line_slope: float,
                      line_intercept: float,
                      other_slope: float,
                      other_intercept: float) -> Point:
    if line_slope == other_slope:
        raise ValueError("Parallel lines do not intersect")
    dbdk = (other_intercept - line_intercept) / (other_slope - line_slope)
    return Point(-dbdk, -line_slope * dbdk + line_intercept)


def insert_between(point: Point,
                   vertices: Iterable[Point],
                   polygon: Polygon) -> Polygon:
    """Inserts point between the vertices of polygon's exterior"""
    polygon_edges = segments(polygon.exterior)
    line_to_split = LineString(vertices)
    index = next_index(polygon_edges, line_to_split.equals)
    coords = list(polygon.exterior.coords)
    return Polygon([*coords[:index + 1], to_tuple(point), *coords[index + 1:]])


def to_tuple(point: Point) -> Tuple[float, float]:
    return point.x, point.y
