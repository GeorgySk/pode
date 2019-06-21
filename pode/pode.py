"""
Implementation of the `ConvexDivide` procedure from the paper
"Polygon Area Decomposition for Multiple-Robot Workspace Division"
"""
import math
from itertools import repeat
from operator import (contains,
                      itemgetter)
from typing import (Callable,
                    Dict,
                    Iterable,
                    Iterator,
                    List,
                    NamedTuple,
                    Tuple)

# As functools.partial has a highlighting bug:
# https://youtrack.jetbrains.com/issue/PY-35363
# I'm using `lz.left.applier` instead
from lz import left
from lz.functional import compose
from shapely.geometry import (LineString,
                              LinearRing,
                              MultiPoint,
                              Point,
                              Polygon)
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

from .utils import (cached_sum,
                    difference_by_key,
                    find_if_or_last,
                    iff,
                    midpoint,
                    next_enumerate,
                    last_index,
                    next_index,
                    right_left_parts,
                    right_part,
                    scalar_multiplication,
                    segments)


# The following monkey-patch should be removed
# when Shapely will make the geometries hashable again
# (should be in Shapely 2.0).
# Currently I use wkb representation to get hashes,
# but it means that, for example, two LineString's consisting of the same
# points but with different directions will have different hash
BaseGeometry.__hash__ = Polygon.__hash__ = compose(hash, BaseGeometry.wkb.fget)

SitesType = Dict[Point, float]


class Partition(NamedTuple):
    """
    Defines a `polygon` and its sites.
    """
    sites: SitesType
    polygon: Polygon


def area_partition(sites_points: MultiPoint,
                   fractions: Iterable[float],
                   vertices: LinearRing,
                   *,
                   tolerance: float = 1e-4
                   ) -> Iterator[Tuple[Polygon, Tuple[Point, float]]]:
    """
    Divides a polygon to as many parts as there are sites
    on the perimeter of the polygon.
    :param sites_points: special points on the perimeter of the polygon
    that define a number of areas, with each area containing one site
    :param fractions: fractions of the resulting areas
    :param vertices: defining a polygon
    :param tolerance: for area check
    :return: sub-polygons with their sites
    """
    if not vertices.is_ccw:
        raise ValueError('Algorithm works only for CCW vertices')
    polygon = Polygon(vertices)
    area_requirements = scalar_multiplication(fractions, polygon.area)
    sites: SitesType = dict(zip(sites_points.geoms, area_requirements))
    distance_along_vertices: Callable[[Point], float] = vertices.project

    multisite_partitions = []
    while True:
        vertices_points = MultiPoint(vertices.coords)
        sites_points = MultiPoint([*sites])
        vertices_and_sites = vertices_points.union(sites_points)
        vertices_and_sites = sorted(vertices_and_sites,
                                    key=distance_along_vertices)
        partitions = divide(vertices_and_sites=vertices_and_sites,
                            sites=sites,
                            tolerance=tolerance)
        for partition in partitions:
            if len(partition.sites) == 1:
                yield (partition.polygon, *partition.sites.items())
            else:
                multisite_partitions.append(partition)
        if not multisite_partitions:
            break
        partition = multisite_partitions.pop()
        polygon = orient(partition.polygon)
        vertices: LinearRing = polygon.exterior
        sites = partition.sites


def divide(vertices_and_sites: List[Point],
           sites: SitesType,
           *,
           tolerance: float) -> Tuple[Partition, Partition]:
    """
    Splits input polygon in two parts
    :param vertices_and_sites: vertices and sites points in CCW order
    :param sites: mapping of sites points to their requirements
    :param tolerance: for area check
    :return: right and left partition containing their sites
    """
    polygon = Polygon(MultiPoint(vertices_and_sites))

    is_site = left.applier(contains, sites)
    first_site_index = next_index(vertices_and_sites, is_site)
    last_site_index = last_index(vertices_and_sites, is_site)
    endpoints = vertices_and_sites[first_site_index:last_site_index]
    first_site = endpoints[0]

    first_partition, line = get_first_partition(
        polygon=polygon,
        start_point=vertices_and_sites[0],
        endpoints=endpoints,
        sites=sites)
    line_end: Point = line.boundary[1]
    area_requirement = cached_sum(*first_partition.sites.values())

    if not line_end.equals(first_site) and has_enough_area(first_partition):
        search_line = previous_side(line_end, polygon.exterior)
        fixed_point = line.boundary[0]
        is_head_fixed = False
    else:
        if line_end.equals(first_site):
            endpoints = (vertices_and_sites[first_site_index - 1::-1]
                         + vertices_and_sites[-1:])
        else:
            endpoints = (vertices_and_sites[:1]
                         + vertices_and_sites[:last_site_index - 1:-1])
        search_line = get_new_search_line(polygon=polygon,
                                          line_end=line_end,
                                          endpoints=endpoints,
                                          sites=first_partition.sites)
        fixed_point = line_end
        is_head_fixed = True

    right_poly, left_poly = bisection_search(fixed_point=fixed_point,
                                             search_line=search_line,
                                             polygon=polygon,
                                             is_head_fixed=is_head_fixed,
                                             area_requirement=area_requirement,
                                             area_tolerance=tolerance)
    left_sites = difference_by_key(sites, first_partition.sites)
    return (Partition(sites=first_partition.sites, polygon=right_poly),
            Partition(sites=left_sites, polygon=left_poly))


def get_first_partition(polygon: Polygon,
                        start_point: Point,
                        endpoints: List[Point],
                        sites: SitesType) -> Tuple[Partition, LineString]:
    """Step Nº3 of the ConvexDivide procedure"""
    lines = list(map(LineString, zip(repeat(start_point), endpoints)))
    right_parts = map(right_part, repeat(polygon), lines)
    right_sites_per_vertex = sites_per_vertex(endpoints, sites)
    partitions = map(Partition, right_sites_per_vertex, right_parts)
    first_has_enough_area = compose(has_enough_area, itemgetter(0))
    return find_if_or_last(first_has_enough_area, zip(partitions, lines))


def has_enough_area(partition: Partition) -> bool:
    return partition.polygon.area >= cached_sum(*partition.sites.values())


def get_new_search_line(polygon: Polygon,
                        line_end: Point,
                        endpoints: List[Point],
                        sites: SitesType) -> LineString:
    """
    Searches for a line to be used in interpolating
    as in the step Nº4 of the algorithm
    """
    lines = list(map(LineString, zip(endpoints, repeat(line_end))))
    right_parts = map(right_part, repeat(polygon), lines)
    partitions = map(Partition, repeat(sites), right_parts)
    partitions_and_lines = zip(partitions, lines)
    partition_has_enough_area = compose(has_enough_area, itemgetter(0))
    _, line = next(filter(partition_has_enough_area, partitions_and_lines))
    return next_side(line.boundary[0], polygon.exterior)


def previous_side(point: Point,
                  ring: LinearRing) -> LineString:
    """Returns a line on the input ring that ends with the input point"""
    lines = segments(ring)
    line_endpoint = compose(itemgetter(1), LineString.boundary.fget)
    is_end_matching = compose(point.equals, line_endpoint)
    try:
        return next(filter(is_end_matching, lines))
    except StopIteration as error:
        raise ValueError(f"Couldn't find a line on the ring {ring.wkt} "
                         f"ending with point {point.wkt}") from error


def next_side(point: Point,
              ring: LinearRing) -> LineString:
    """Returns a line on the input ring starting with the input point"""
    lines = segments(ring)
    line_endpoint = compose(itemgetter(0), LineString.boundary.fget)
    is_start_matching = compose(point.equals, line_endpoint)
    try:
        return next(filter(is_start_matching, lines))
    except StopIteration as error:
        raise ValueError(f"Couldn't find a line on the ring {ring.wkt} "
                         f"starting with point {point.wkt}") from error


def sites_per_vertex(domain_vertices: List[Point],
                     sites: SitesType) -> Iterator[SitesType]:
    """
    When rotating a line head over `domain_vertices`
    we need to get on each step a list of all previously encountered
    sites, so that later we can extract corresponding area requirement.
    """
    yield {domain_vertices[0]: sites[domain_vertices[0]]}
    result: SitesType = {}
    # previous points for S1, S2, S3, ...
    for vertex in domain_vertices[:-1]:
        if vertex in sites:
            result[vertex] = sites[vertex]
        yield result.copy()


def bisection_search(*,
                     fixed_point: Point,
                     search_line: LineString,
                     polygon: Polygon,
                     is_head_fixed: bool,
                     area_requirement: float,
                     area_tolerance: float,
                     max_iterations_count: int = 10**9
                     ) -> Tuple[Polygon, Polygon]:
    """
    Splits a polygon to right and left parts by a bisection search.
    :param fixed_point: line splitting a polygon rotates around it
    :param search_line: initial guess - side of the polygon
    :param polygon: input polygon
    :param is_head_fixed: determines if the hear or tail
    of the line will rotate
    :param area_requirement: area of the right part after splitting
    :param area_tolerance: valid difference between `area_requirement`
    and obtained value
    :param max_iterations_count: if exceeded, will raise an error
    :return: right and left parts of the input polygon
    """
    polygon_coordinates = list(polygon.exterior.coords)
    polygon_sides = segments(polygon.exterior)
    start_search_index, search_line = next_enumerate(polygon_sides,
                                                     search_line.equals)
    for _ in range(max_iterations_count):
        endpoint = midpoint(search_line)
        division_line = (LineString([endpoint, fixed_point]) if is_head_fixed
                         else LineString([fixed_point, endpoint]))
        # adding a "snap"-point on the search line - prevents precision errors
        polygon_coordinates.insert(start_search_index + 1, endpoint)
        polygon = Polygon(polygon_coordinates)
        right_poly, left_poly = right_left_parts(polygon, division_line)
        if math.isclose(right_poly.area,
                        area_requirement,
                        rel_tol=area_tolerance):
            return right_poly, left_poly
        # removing "snap"-point
        polygon_coordinates.pop(start_search_index + 1)
        if iff(right_poly.area > area_requirement, is_head_fixed):
            search_line = LineString([endpoint, search_line.coords[1]])
        else:
            search_line = LineString([search_line.coords[0], endpoint])
    raise ValueError(f"Failed to find partition "
                     f"after {max_iterations_count} steps.")
