import operator
from fractions import Fraction
from functools import (reduce,
                       singledispatch)
from statistics import mean
from typing import (Iterator,
                    List,
                    Sequence,
                    TypeVar,
                    Union)

from gon.angular import Orientation
from gon.compound import Shaped
from gon.degenerate import (EMPTY,
                            Empty)
from gon.discrete import Multipoint
from gon.geometry import Geometry
from gon.linear import (Contour,
                        Segment)
from gon.primitive import Point
from gon.shaped import Polygon
from sect.triangulation import constrained_delaunay_triangles

from pode.hints import (ContourType,
                        ConvexPartsType,
                        PointType,
                        SegmentType)

GeometryType = TypeVar('GeometryType', bound=Geometry)


def cut(contour: Contour,
        start: Point,
        end: Point) -> List[Point]:
    """
    Returns all points between the start point and the end point lying
    on the counter including the endpoints themselves.
    The points do not necessarily match with contour vertices.
    """
    if any(point not in contour for point in (start, end)):
        raise ValueError(f"Both start point and end point should lie on "
                         f"the contour")
    vertices = [*contour.vertices, contour.vertices[0]]
    start_index = (vertices.index(start) if start in vertices
                   else next((index
                              for index, edge in enumerate(edges(contour))
                              if start in edge)))
    end_index = (vertices[1:].index(end) + 1 if end in vertices
                 else next((index for index, edge in enumerate(edges(contour))
                           if end in edge)))
    if start_index < end_index:
        result = vertices[start_index + 1:end_index + 1]
    elif start_index > end_index:
        result = vertices[start_index + 1:-1] + vertices[:end_index + 1]
    else:
        segment_to_start = Segment(vertices[start_index], start)
        segment_to_end = Segment(vertices[start_index], end)
        if segment_to_start.length < segment_to_end.length:
            return [start, end]
        else:
            result = vertices[start_index + 1:-1] + vertices[:end_index + 1]
    head = [vertices[start_index]] if start in vertices else [start]
    tail = [] if end in vertices else [end]
    return head + result + tail


def edges(contour: Contour) -> Iterator[Segment]:
    vertices = [*contour.vertices, contour.vertices[0]]
    yield from (Segment(vertices[index], vertices[index + 1])
                for index in range(len(vertices) - 1))


def shrink_collinear_vertices(contour: Contour) -> List[Point]:
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


def union(*geometries: Shaped) -> Union[Shaped, Empty]:
    if not geometries:
        return EMPTY
    return reduce(operator.or_, geometries)


def midpoint(start: Point,
             end: Point) -> Point:
    """Returns coordinates of the middle of the input line"""
    return Point((start.x + end.x) / 2, (start.y + end.y) / 2)


def splitter_point(requirement: float,
                   pivot: Point,
                   low_area_point: Point,
                   high_area_point: Point) -> Point:
    """Alternative to bisection search since we always have triangles"""
    if requirement <= 0:
        raise ValueError("Can't have a zero or negative requirement")
    p = pivot
    l = low_area_point
    h = high_area_point
    contour = Contour([p, l, h])
    triangle = Polygon(contour)
    if triangle.area < requirement:
        raise ValueError("Can't have a requirement greater than the area of "
                         "the triangle")
    r = requirement
    dx = h.x - l.x
    if dx == 0:
        if contour.orientation is Orientation.COUNTERCLOCKWISE:
            a = (p.y - l.y) / (p.x - l.x)
            b = (p.x * l.y - l.x * p.y - 2 * r) / (p.x - l.x)
            x = l.x
            y = a * l.x + b
            return Point(x, y)
        else:
            t = triangle.area - r
            a = (p.y - h.y) / (p.x - h.x)
            b = (p.x * h.y - h.x * p.y - 2 * t) / (p.x - h.x)
            x = h.x
            y = a * h.x + b
            return Point(x, y)
    k = (h.y - l.y) / dx
    m = h.y - k * h.x
    if contour.orientation is Orientation.COUNTERCLOCKWISE:
        dx = p.x - l.x
        if dx == 0:
            x = (2 * r + l.x * p.y - p.x * l.y) / (p.y - l.y)
            y = k * x + m
            return Point(x, y)
        a = (p.y - l.y) / dx
        b = (p.x * l.y - l.x * p.y - 2 * r) / dx
    else:
        t = triangle.area - r
        dx = p.x - h.x
        if dx == 0:
            x = (2 * t + h.x * p.y - p.x * h.y) / (p.y - h.y)
            y = k * x + m
            return Point(x, y)
        a = (p.y - h.y) / dx
        b = (p.x * h.y - h.x * p.y - 2 * t) / dx
    x = (m - b) / (a - k)
    y = a * x + b
    return Point(x, y)


def orient(polygon: Polygon) -> Polygon:
    """To counterclockwise. No holes"""
    return Polygon(polygon.border.to_counterclockwise())


def centroid(points: Multipoint) -> Point:
    """Arithmetic mean position of the given points"""
    mean_x = mean(point.x for point in points.points)
    mean_y = mean(point.y for point in points.points)
    return Point(mean_x, mean_y)


def joined_constrained_delaunay_triangles(
        border: ContourType,
        holes: Sequence[ContourType] = (),
        *,
        extra_points: Sequence[PointType] = (),
        extra_constraints: Sequence[SegmentType] = ()) -> ConvexPartsType:
    """Joins polygons to form convex parts of greater size"""
    triangles = constrained_delaunay_triangles(
        border, holes,
        extra_points=extra_points,
        extra_constraints=extra_constraints)
    polygons = [Polygon.from_raw((list(triangle), []))
                for triangle in triangles]
    initial_polygon = polygons.pop()
    result = []
    while True:
        resulting_polygon = initial_polygon
        for index, polygon in enumerate(iter(polygons)):
            polygon_sides = set(edges(polygon.border))
            common_side = next((edge
                                for edge in edges(resulting_polygon.border)
                                if edge in polygon_sides), None)
            if common_side is None:
                continue
            has_point_on_edge = any(Point.from_raw(raw_point) in common_side
                                    for raw_point in extra_points)
            if has_point_on_edge:
                continue
            union_ = union(resulting_polygon, polygon)
            if isinstance(union_, Polygon) and union_.is_convex:
                polygons.pop(index)
                resulting_polygon = union_
        if resulting_polygon is not initial_polygon:
            initial_polygon = resulting_polygon
            continue
        result.append(resulting_polygon.border.raw())
        if not polygons:
            return result
        initial_polygon = polygons.pop()
