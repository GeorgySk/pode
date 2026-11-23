"""Geometric helpers used by the decomposition algorithm."""

import operator
from collections.abc import (Iterator,
                             Sequence)
from fractions import Fraction
from functools import (reduce,
                       singledispatch)
from typing import TypeVar

from gon.base import (Contour,
                      EMPTY,
                      Empty,
                      Geometry,
                      Orientation,
                      Point,
                      Polygon,
                      Segment,
                      Shaped,
                      Triangulation)
from ground.base import Context

from pode.hints import Scalar

GeometryType = TypeVar("GeometryType", bound=Geometry)
T = TypeVar("T")

TRIANGLE_VERTICES_COUNT = 3


def cut(contour: Contour,
        start: Point,
        end: Point) -> list[Point]:
    """
    Returns the points of the contour from the start to the end point.

    The contour is traversed in the order of its vertices. The result
    starts with the start point, ends with the end point, and contains
    the contour vertices passed in between. The start and end points
    don't have to be contour vertices. If they coincide, the whole
    contour is traversed, so the point is both the first and the last
    element.

    :param contour: contour to traverse
    :param start: point on the contour to start from
    :param end: point on the contour to stop at
    :return: start point, contour vertices in between, end point
    :raises ValueError: if the start or end point doesn't lie on the
                        contour
    """
    if any(point not in contour for point in (start, end)):
        raise ValueError("Both start point and end point should lie on "
                         "the contour")
    vertices = [*contour.vertices, contour.vertices[0]]
    sides = list(edges(contour))
    start_is_vertex = start in vertices
    end_is_vertex = end in vertices
    # A vertex start is the start of the side that follows it,
    # while a vertex end is the end of the side that precedes it.
    start_index = (vertices.index(start) if start_is_vertex
                   else next(index for index, side in enumerate(sides)
                             if start in side))
    end_index = (vertices.index(end, 1) if end_is_vertex
                 else next(index for index, side in enumerate(sides)
                           if end in side))
    side_start = vertices[start_index]
    goes_forward = (start_index < end_index
                    or (start_index == end_index
                        and (squared_distance(side_start, start)
                             < squared_distance(side_start, end))))
    result = (vertices[start_index + 1:end_index + 1] if goes_forward
              else vertices[start_index + 1:-1] + vertices[:end_index + 1])
    # A vertex end is already the last element of the result.
    tail = [] if end_is_vertex else [end]
    return [start, *result, *tail]


def edges(contour: Contour) -> Iterator[Segment]:
    """Yields the sides of the contour in the order of its vertices."""
    vertices = [*contour.vertices, contour.vertices[0]]
    yield from (Segment(vertices[index], vertices[index + 1])
                for index in range(len(vertices) - 1))


def squared_distance(first: Point,
                     second: Point) -> Scalar:
    """
    Returns the squared Euclidean distance between the points.

    Unlike the distance itself, it needs no square root, so it stays
    exact for rational coordinates.

    :param first: one point
    :param second: other point
    :return: squared distance
    """
    return ((first.x - second.x) ** 2  # type: ignore[no-any-return]
            + (first.y - second.y) ** 2)


def shrink_collinear_vertices(contour: Contour) -> list[Point]:
    """Returns the contour vertices except those collinear with neighbors."""
    vertices = list(contour.vertices)
    index = -len(vertices) + 1
    while index < 0:
        while (max(2, -index) < len(vertices)
               and (Contour([vertices[index + 2],
                             vertices[index + 1],
                             vertices[index]]).orientation
                    is Orientation.COLLINEAR)):
            del vertices[index + 1]
        index += 1
    while index < len(vertices):
        while (max(2, index) < len(vertices)
               and (Contour([vertices[index - 2],
                             vertices[index - 1],
                             vertices[index]]).orientation
                    is Orientation.COLLINEAR)):
            del vertices[index - 1]
        index += 1
    return vertices


@singledispatch
def to_fractions(geometry: GeometryType) -> GeometryType:
    """Converts coordinates of the geometry to fractions."""
    raise TypeError(f"Unsupported type: {type(geometry)}")


@to_fractions.register
def _(geometry: Point) -> Point:
    return Point(Fraction(geometry.x), Fraction(geometry.y))


@to_fractions.register
def _(geometry: Contour) -> Contour:
    return Contour(list(map(to_fractions, geometry.vertices)))


@to_fractions.register
def _(geometry: Polygon) -> Polygon:
    return Polygon(to_fractions(geometry.border),
                   list(map(to_fractions, geometry.holes)))


def unite(*geometries: Shaped) -> Shaped | Empty:
    """Returns the union of the geometries, or `EMPTY` if none given."""
    return reduce(operator.or_, geometries, EMPTY)


def splitter_point(requirement: Scalar,
                   pivot: Point,
                   low_area_point: Point,
                   high_area_point: Point) -> Point:
    """Alternative to bisection search since we always have triangles."""
    if requirement <= 0:
        raise ValueError("Can't have a zero or negative requirement")
    low = low_area_point
    high = high_area_point
    contour = Contour([pivot, low, high])
    triangle = Polygon(contour)
    if triangle.area < requirement:
        raise ValueError("Can't have a requirement greater than the area of "
                         "the triangle")
    r = requirement
    dx = high.x - low.x
    if dx == 0:
        if contour.orientation is Orientation.COUNTERCLOCKWISE:
            a = (pivot.y - low.y) / (pivot.x - low.x)
            b = (pivot.x * low.y - low.x * pivot.y - 2 * r) / (pivot.x - low.x)
            x = low.x
            y = a * low.x + b
            return Point(x, y)
        t = triangle.area - r
        a = (pivot.y - high.y) / (pivot.x - high.x)
        b = (pivot.x * high.y - high.x * pivot.y - 2 * t) / (pivot.x - high.x)
        x = high.x
        y = a * high.x + b
        return Point(x, y)
    k = (high.y - low.y) / dx
    m = high.y - k * high.x
    if contour.orientation is Orientation.COUNTERCLOCKWISE:
        dx = pivot.x - low.x
        if dx == 0:
            x = (2 * r + low.x * pivot.y - pivot.x * low.y) / (pivot.y - low.y)
            y = k * x + m
            return Point(x, y)
        a = (pivot.y - low.y) / dx
        b = (pivot.x * low.y - low.x * pivot.y - 2 * r) / dx
    else:
        t = triangle.area - r
        dx = pivot.x - high.x
        if dx == 0:
            x = ((2 * t + high.x * pivot.y - pivot.x * high.y)
                 / (pivot.y - high.y))
            y = k * x + m
            return Point(x, y)
        a = (pivot.y - high.y) / dx
        b = (pivot.x * high.y - high.x * pivot.y - 2 * t) / dx
    x = (m - b) / (a - k)
    y = a * x + b
    return Point(x, y)


def orient(polygon: Polygon) -> Polygon:
    """To counterclockwise. No holes."""
    return Polygon(polygon.border.to_counterclockwise())


def joined_constrained_delaunay_triangles(
        polygon: Polygon,
        *,
        extra_points: Sequence[Point] = (),
        extra_constraints: Sequence[Segment] = (),
        context: Context) -> list[Contour]:
    """Joins polygons to form convex parts of greater size."""
    triangles = Triangulation.constrained_delaunay(
        polygon,
        extra_points=extra_points,
        extra_constraints=extra_constraints,
        context=context).triangles()
    polygons = list(map(Polygon, triangles))
    initial_polygon = polygons.pop()
    result = []
    while True:
        resulting_polygon = initial_polygon
        for index, part in enumerate(iter(polygons)):
            part_sides = set(edges(part.border))
            common_side = next((edge
                                for edge in edges(resulting_polygon.border)
                                if edge in part_sides), None)
            if common_side is None:
                continue
            has_point_on_edge = any(point in common_side
                                    for point in extra_points)
            if has_point_on_edge:
                continue
            union_ = unite(resulting_polygon, part)
            if isinstance(union_, Polygon) and union_.is_convex:
                polygons.pop(index)
                resulting_polygon = union_
        if resulting_polygon is not initial_polygon:
            initial_polygon = resulting_polygon
            continue
        result.append(resulting_polygon.border)
        if not polygons:
            return result
        initial_polygon = polygons.pop()


def rotate(sequence: list[T], index: int) -> list[T]:
    """Cyclically shifts the sequence to start from the given index."""
    return sequence[index:] + sequence[:index]


def order_convex_contour_points(points: list[Point]) -> list[Point]:
    """
    Orders points on the border of a convex polygon counterclockwise.

    The lowest leftmost point goes first.
    """
    lowest_leftmost_point_index = 0
    lowest_leftmost_point = points[lowest_leftmost_point_index]
    for index, point in enumerate(points[1:], start=1):
        if (point.y < lowest_leftmost_point.y
                or point.y == lowest_leftmost_point.y
                and point.x < lowest_leftmost_point.x):
            lowest_leftmost_point_index = index
            lowest_leftmost_point = point
    points = (points[:lowest_leftmost_point_index]
              + points[lowest_leftmost_point_index + 1:])

    def cotangent(point: Point) -> Fraction | float:
        dx = point.x - lowest_leftmost_point.x
        dy = point.y - lowest_leftmost_point.y
        return (-dx / dy if dy
                else (float("-inf") if dx > 0 else float("inf")))

    cotangents_per_point = {point: cotangent(point) for point in points}
    max_tangent = max(cotangents_per_point.values())

    def sorting_key(point: Point) -> tuple[Fraction | float, Fraction]:
        tangent = cotangents_per_point[point]
        distance = lowest_leftmost_point.distance_to(point)
        factor = -1 if tangent == max_tangent else 1
        return tangent, factor * distance

    sorted_points = sorted(points, key=sorting_key)
    return [lowest_leftmost_point, *sorted_points]
