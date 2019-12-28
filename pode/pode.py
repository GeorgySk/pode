"""
Implementation of the `ConvexDivide` procedure from the paper
"Polygon Area Decomposition for Multiple-Robot Workspace Division"
"""
import math
from collections import (Counter,
                         defaultdict)
from collections.abc import Hashable
from functools import partial
from itertools import (chain,
                       dropwhile,
                       filterfalse,
                       islice,
                       repeat,
                       takewhile)
from math import isclose
from operator import (contains,
                      itemgetter,
                      ne,
                      not_,
                      xor)
from typing import (Callable,
                    Dict,
                    Iterable,
                    Iterator,
                    List,
                    NamedTuple,
                    Optional,
                    Set,
                    Tuple,
                    TypeVar)

import networkx as nx
# As functools.partial has a highlighting bug:
# https://youtrack.jetbrains.com/issue/PY-35363
# I'm using `lz.left.applier` instead
from lz import left
from lz.functional import compose
from lz.iterating import (first,
                          flatten,
                          pairwise)
from shapely.geometry import (LineString,
                              LinearRing,
                              MultiPoint,
                              MultiPolygon,
                              Point,
                              Polygon)
from shapely.geometry.base import (BaseGeometry,
                                   BaseMultipartGeometry)
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from .geometry_utils import (are_touching,
                             insert_between,
                             is_on_the_right,
                             join_to_convex,
                             midpoint,
                             right_left_parts,
                             right_part,
                             rotating_splitter,
                             segments,
                             to_convex_parts,
                             to_graph,
                             to_tuple,
                             touching_sides)
from .utils import (cached_sum,
                    difference_by_key,
                    find_if_or_last,
                    iff,
                    last_index,
                    next_enumerate,
                    next_index,
                    scalar_multiplication)

# The following monkey-patch should be removed
# when Shapely will make the geometries hashable again
# (should be in Shapely 2.0).
# Currently I use wkb representation to get hashes,
# but it means that, for example, two LineString's consisting of the same
# points but with different directions will have different hash
BaseGeometry.__hash__ = BaseMultipartGeometry.__hash__ = Polygon.__hash__ = (
    compose(hash, BaseGeometry.wkb.fget))

Node = TypeVar('Node', bound=Hashable)
SitesType = Dict[Point, float]


class Partition(NamedTuple):
    """
    Defines a `polygon` and its sites.
    """
    sites: SitesType
    polygon: Polygon


def divide(polygon: Polygon,
           *,
           sites: Optional[SitesType] = None,
           requirements: Optional[List[float]] = None,
           joined: bool = False,
           gradual: bool = False) -> Iterator[Tuple[Polygon,
                                                    Tuple[Point, float]]]:
    """
    Divides given polygon for the given sites or requirements
    :param polygon: input Shapely polygon
    :param sites: an optional mapping of Shapely points
    to area requirements for each point
    :param requirements: an optional list of area requirements
    (should sum up to the area of the polygon)
    :param joined: if `True`, will join consecutive triangles
    of Delaunay triangulation
    :param gradual: if `True` will search the split-lines
    by gradually shifting the edge between two polygon parts
    :return: parts of the polygon
    along with corresponding sites points and its requirements
    """
    if not xor(sites is None, requirements is None):
        raise ValueError('At least one of `sites` or `requirements` '
                         'is required.')
    if gradual and requirements is None:
        raise ValueError('Gradual edge search can be performed only for '
                         'flexible sites positioning. Either provide `sites` '
                         'instead of `requirements`, or turn off '
                         'the `gradual` option.')
    if (requirements is not None
            and not isclose(sum(requirements), polygon.area)):
        raise ValueError('Provided requirements do not sum up '
                         'to the area of the given polygon.')
    detach = divide_on_the_go if sites is None else detach_and_assign
    parts = list(to_convex_parts(polygon))
    if joined:
        parts = list(join_to_convex(parts))
    graph = to_graph(parts)
    graph = to_ordered(graph)
    if sites is None:
        yield from detach(graph,
                          requirements,
                          with_gradual_search=gradual)
    else:
        graph = attach_sites(graph, sites)
        yield from detach(graph)


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
        partitions = divide_in_two(vertices_and_sites=vertices_and_sites,
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


def divide_in_two(vertices_and_sites: List[Point],
                  sites: SitesType,
                  *,
                  tolerance: float) -> Tuple[Partition, Partition]:
    """
    Splits input polygon in two parts
    :param vertices_and_sites: vertices and sites points in CCW order_by_site
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
    # TODO: check if this should be `has_more_than_enough_area`
    partition_has_enough_area = compose(has_enough_area, itemgetter(0))
    _, line = next(filter(partition_has_enough_area, partitions_and_lines))
    return next_side(line.boundary[0], polygon.exterior)


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


def bisection_search(*,
                     fixed_point: Point,
                     search_line: LineString,
                     polygon: Polygon,
                     is_head_fixed: bool,
                     area_requirement: float,
                     area_tolerance: float,
                     max_iterations_count: int = 10 ** 9
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


def detach_and_assign(graph: nx.Graph
                      ) -> Iterator[Tuple[Polygon, Tuple[Point, float]]]:
    graph = graph.copy()
    area_incomplete_polygons = defaultdict(list)
    pseudo_sites_relations = {}

    graph_iterator = iter(graph)
    while True:
        if not graph:
            return
        polygon: Polygon = next(graph_iterator)
        sites: SitesType = graph.nodes[polygon]
        if not sites:
            continue
        pred_polygons = pred_polys(polygon, graph=graph)
        pred_polygon = unary_union(pred_polygons)
        if is_area_complete(pred_polygon, sites=sites):
            if len(sites) == 1:
                graph.remove_nodes_from(pred_polygons)
                site_point, requirement = first(sites.items())
                if site_point not in pseudo_sites_relations:
                    yield (pred_polygon, *sites.items())
                else:
                    original_site = pseudo_sites_relations[site_point]
                    point, requirement = first(original_site.items())
                    all_related_polygons = [*area_incomplete_polygons[point],
                                            pred_polygon]
                    resulting_polygon = unary_union(all_related_polygons)
                    yield resulting_polygon, (point, requirement)
            else:
                neighbor = next_neighbor(polygon, graph=graph)
                graph = update_graph(polygon=polygon,
                                     sites=sites,
                                     edge=None,
                                     neighbor=neighbor,
                                     graph=graph)
        elif is_area_incomplete(pred_polygon, sites=sites):
            neighbor = next_neighbor(polygon, graph=graph)
            edge = graph[polygon][neighbor]['side']
            if len(sites) == 1:
                site_point, requirement = first(sites.items())
                site_point, initial_requirement = first(
                    pseudo_sites_relations.get(site_point, sites).items())
                pseudo_requirement = requirement - pred_polygon.area
                pseudo_site_point = Point(midpoint(edge))
                pred_polygon = insert_between(point=pseudo_site_point,
                                              vertices=edge.boundary,
                                              polygon=pred_polygon)
                area_incomplete_polygons[site_point].append(pred_polygon)
                graph.remove_nodes_from(pred_polygons)
                pseudo_sites_relations[pseudo_site_point] = (
                    {site_point: initial_requirement})
                graph.nodes[neighbor].update(
                    {pseudo_site_point: pseudo_requirement})
                original_neighbor = neighbor
                neighbor = insert_between(point=pseudo_site_point,
                                          vertices=edge.boundary,
                                          polygon=neighbor)
                graph = nx.relabel_nodes(graph, {original_neighbor: neighbor})
            else:
                graph = update_graph(polygon=polygon,
                                     sites=sites,
                                     edge=edge,
                                     neighbor=neighbor,
                                     graph=graph)
        else:
            neighbor = next_neighbor(polygon, graph=graph)
            edge = graph[polygon][neighbor]['side']
            graph = update_graph(polygon=polygon,
                                 sites=sites,
                                 edge=edge,
                                 neighbor=neighbor,
                                 graph=graph)
        graph_iterator = iter(graph)


def add_point_to_neighbor(point: Point,
                          vertices: Tuple[Point, Point],
                          polygon: Polygon,
                          graph: nx.Graph) -> nx.Graph:
    """
    Adds a point between given vertices
    to the neighbor of the given polygon.
    """
    for neighbor in graph.neighbors(polygon):
        if set(map(to_tuple, vertices)) <= set(neighbor.exterior.coords):
            new_polygon = insert_between(point=point,
                                         vertices=vertices,
                                         polygon=neighbor)
            return nx.relabel_nodes(graph, {neighbor: new_polygon})
    return graph


def pred_polys(node: Polygon,
               graph: nx.Graph) -> Set[Polygon]:
    """
    Returns a set with all connected predecessors
    including the node itself
    """
    # sometimes node has reordered vertices or additional points:
    if node not in graph:
        node = get_original_node(node, graph=graph)
    nodes_to_remove = set(successors(node, graph))
    is_not_successor = compose(not_, nodes_to_remove.__contains__)
    no_successors_graph = nx.subgraph_view(graph, filter_node=is_not_successor)
    for component in nx.connected_components(no_successors_graph):
        if node in component:
            return component
    else:
        raise ValueError('The node is not in the graph')


def is_area_complete(polygon: Polygon,
                     sites: SitesType) -> bool:
    """Checks if area of polygon equals total sites requirement"""
    return isclose(polygon.area, sum(sites.values()), rel_tol=1e-3)


def next_neighbor(polygon: Polygon,
                  graph: nx.Graph) -> Optional[Polygon]:
    """Most immediate successor which is a neighbor of a given node"""
    neighbors_successors = (node for node in successors(polygon, graph)
                            if node in graph.neighbors(polygon))
    return next(neighbors_successors, None)


def successors(node: Node,
               graph: nx.Graph) -> Iterator[Node]:
    """
    For a graph (N1, N2, ..., Nm) and its node Nj
    returns all Nj's successor nodes (Nj+1, Nj+2, ..., Nm).
    Note that this definition is different
    from the definition of successors for directed graphs.
    """
    not_a_node = partial(ne, node)
    node_and_successors = dropwhile(not_a_node, graph.nodes)
    yield from islice(node_and_successors, 1, None)


def update_graph(polygon: Polygon,
                 sites: SitesType,
                 edge: Optional[LineString],
                 neighbor: Polygon,
                 graph: nx.Graph,
                 with_gradual_search: bool = False) -> nx.Graph:
    original_polygon = polygon
    polygon = add_sites(orient(polygon), sites=sites)
    if edge is not None:
        polygon = order_by_edge(polygon, edge=edge)
    else:
        polygon = order_by_site(polygon, sites=sites)
    graph = nx.relabel_nodes(graph, {original_polygon: polygon})
    if with_gradual_search and len(sites) == 1:
        partitions_to_process = gradual_search(polygon=polygon,
                                               sites=sites,
                                               edge=edge)
    else:
        partitions_to_process, graph, neighbor = nonconvex_divide(polygon,
                                                                  sites,
                                                                  graph=graph)
    partitions_union = unary_union([partition.polygon
                                    for partition in partitions_to_process])
    nodes_to_remove = [node for node in graph
                       if partitions_union.contains(node.buffer(-1e-9))]
    graph.remove_nodes_from(nodes_to_remove)
    if len(partitions_to_process) == 3:  # Case 1.3
        return prepend_partitions_inline(partitions_to_process,
                                         graph=graph,
                                         node_to_attach_to=neighbor)
    else:  # Cases 1.1 and 1.2
        return prepend_partitions(partitions_to_process,
                                  graph=graph,
                                  node_to_attach_to=neighbor)


def add_sites(polygon: Polygon,
              sites: SitesType) -> Polygon:
    """
    Adds sites to polygon vertices for further iteration over them
    in the nonconvex division part of the algorithm.
    """
    polygon_ring = polygon.exterior
    distance_along_vertices: Callable[[Point], float] = polygon_ring.project
    vertices = MultiPoint(polygon_ring.coords[:-1])
    sites_points = MultiPoint([*sites])
    vertices_and_sites = vertices.union(sites_points)
    sorted_points = sorted(vertices_and_sites.geoms,
                           key=distance_along_vertices)
    # workaround for a bug https://github.com/Toblerity/Shapely/issues/706
    return Polygon(LineString(sorted_points))


def order_by_edge(polygon: Polygon,
                  edge: LineString) -> Polygon:
    vertices = list(polygon.exterior.coords)
    edge_vertices = edge.coords
    end, start = sorted(map(vertices.index, edge_vertices))
    if end == 0 and start != 1:
        end, start = start, len(vertices) - 1
    reordered_vertices = vertices[start:] + vertices[1:end + 1]
    return Polygon(reordered_vertices)


def order_by_site(polygon: Polygon,
                  sites: SitesType) -> Polygon:
    """
    Reorders sites of the polygon
    so that the vertex corresponding to the first site
    would be the last vertex in the new ordering
    """
    vertices = list(polygon.exterior.coords[:-1])
    # not necessary should be the first site
    first_site = next(iter(sites))
    index = vertices.index(first_site.coords[0])
    reordered_vertices = vertices[index + 1:] + vertices[:index + 1]
    return Polygon(reordered_vertices)


def nonconvex_divide(polygon: Polygon,
                     sites: SitesType,
                     *,
                     graph: nx.Graph,
                     tolerance: float = 1e-4) -> Tuple[List[Partition],
                                                       nx.Graph,
                                                       Polygon]:
    """
    Splits a pred_poly based on the polygon to 2 or 3 parts
    for further division
    """
    if not polygon.exterior.is_ccw:
        raise ValueError("Polygon division is implemented only for polygons "
                         "oriented counter-clockwise")
    is_site = left.applier(contains, sites)
    vertices_points = LineString(polygon.exterior.coords)
    distance_along_vertices = vertices_points.project
    sites_points = MultiPoint(list(sites))
    vertices_and_sites = MultiPoint(vertices_points.coords).union(sites_points)
    vertices_and_sites = sorted(vertices_and_sites,
                                key=distance_along_vertices)
    vertices_and_sites.append(vertices_and_sites[0])
    first_site_index = next_index(vertices_and_sites, is_site)
    first_site_point = vertices_and_sites[first_site_index]
    endpoints = vertices_and_sites[first_site_index:-1]
    first_partition, first_line = get_first_partition_for_nonconvex(
        polygon,
        start_point=vertices_and_sites[0],
        endpoints=endpoints,
        sites=sites,
        graph=graph)
    area_requirement = sum(first_partition.sites.values())
    # Step Nº4
    if first_partition.polygon.area >= area_requirement:
        if first_line.boundary[1].equals(first_site_point):
            # probably could be simply vertices_and_sites
            # also, this is similar to `get_new_search_line`
            start_points = vertices_and_sites[:first_site_index]
            lines_points = zip(start_points, repeat(first_site_point))
            lines = list(map(LineString, lines_points))
            right_parts = map(plr, repeat(polygon), lines, repeat(graph))
            partitions = map(Partition,
                             repeat(first_partition.sites),
                             right_parts)
            partitions_and_lines = list(zip(partitions, lines))
            partition_has_enough_area = compose(has_more_than_enough_area,
                                                itemgetter(0))
            _, line = next(
                filterfalse(partition_has_enough_area, partitions_and_lines),
                partitions_and_lines[0])
            t1 = line.boundary[0]
            t2 = previous_side(t1, LinearRing(LineString(vertices_and_sites))
                               ).boundary[0]
            t3 = line.boundary[1]
            triangle = Polygon(LineString([t1, t2, t3]))
            if triangle.is_empty or not triangle.is_valid:
                triangle = Polygon()
            fixed_point = line.boundary[1]
            is_head_fixed = True
        else:
            line_end_index = vertices_and_sites.index(first_line.boundary[1])
            t1 = vertices_and_sites[line_end_index - 1]
            t2 = vertices_and_sites[line_end_index]
            t3 = vertices_and_sites[0]
            line = LineString([t3, t1])
            triangle = Polygon(LineString([t1, t2, t3]))
            if (triangle.is_empty or not triangle.is_valid
                    or triangle.area < 1e-15):
                triangle = Polygon()
            try:
                fixed_point = line.boundary[0]
            except IndexError:
                # line is actually in the form of LineString([a, a])
                fixed_point = Point(line.coords[0])
            is_head_fixed = False
        plr_1 = plr(polygon=polygon,
                    line=line,
                    graph=graph)
        pll_1 = pll(polygon=polygon,
                    plr_=plr_1,
                    graph=graph)
        if plr_1.area + triangle.area > area_requirement:
            t = bisection_search_w_point(fixed_point=fixed_point,
                                         search_line=LineString([t1, t2]),
                                         polygon=polygon,
                                         is_head_fixed=is_head_fixed,
                                         area_requirement=area_requirement,
                                         graph=graph,
                                         area_tolerance=tolerance)
            if t not in {t1, t2}:
                original_polygon = polygon
                polygon = insert_between(point=t,
                                         vertices=(t1, t2),
                                         polygon=polygon)
                graph = nx.relabel_nodes(graph, {original_polygon: polygon})
                graph = add_point_to_neighbor(point=t,
                                              vertices=(t1, t2),
                                              polygon=polygon,
                                              graph=graph)
                if (Point(t1).intersects(plr_1)
                        and Point(t2).intersects(plr_1)
                        and (LineString([t1, t2]) in segments(plr_1.exterior)
                             or LineString([t2, t1])
                             in segments(plr_1.exterior))):
                    plr_1 = insert_between(point=t,
                                           vertices=(t1, t2),
                                           polygon=plr_1)
                if (Point(t1).intersects(pll_1)
                        and Point(t2).intersects(pll_1)
                        and (LineString([t1, t2]) in segments(pll_1.exterior)
                             or LineString([t2, t1])
                             in segments(pll_1.exterior))):
                    pll_1 = insert_between(point=t,
                                           vertices=(t1, t2),
                                           polygon=pll_1)
            triangle = Polygon(LineString([t1, t, t3]))
            if triangle.is_empty or not triangle.is_valid:
                triangle = Polygon()
            if LineString([t1, t]).is_valid:
                pred_by_line = pred_poly_by_line(LineString([t1, t]),
                                                 polygon,
                                                 graph)
                a = plr_1.union(triangle).union(pred_by_line)
            else:
                a = plr_1
                pred_by_line = Polygon()
            b = pll_1.difference(triangle).difference(pred_by_line)
            # TODO: this looks like a dirty hack for the issue #13
            if isinstance(b, MultiPolygon):
                b = max(b, key=Polygon.area.fget)
            if not a.is_empty:
                a_partition = Partition(polygon=a,
                                        sites=first_partition.sites)
                b_partition = Partition(
                    polygon=b,
                    sites=difference_by_key(sites, first_partition.sites))
            else:
                a_partition = Partition(polygon=a,
                                        sites={})
                b_partition = Partition(polygon=b,
                                        sites=sites)
            return [a_partition, b_partition], graph, next_neighbor(polygon,
                                                                    graph)
        pred_by_line = pred_poly_by_line(LineString([t1, t2]), polygon, graph)
        if plr_1.area + pred_by_line.area < area_requirement:
            t = bisection_search_w_point(fixed_point=fixed_point,
                                         search_line=LineString([t1, t2]),
                                         polygon=polygon,
                                         is_head_fixed=is_head_fixed,
                                         area_requirement=area_requirement,
                                         graph=graph,
                                         area_tolerance=tolerance)
            if t not in {t1, t2}:
                original_polygon = polygon
                polygon = insert_between(point=t,
                                         vertices=(t1, t2),
                                         polygon=polygon)
                graph = nx.relabel_nodes(graph, {original_polygon: polygon})
                graph = add_point_to_neighbor(point=t,
                                              vertices=(t1, t2),
                                              polygon=polygon,
                                              graph=graph)
            triangle = Polygon(LineString([t1, t, t3]))
            if triangle.is_empty or not triangle.is_valid:
                triangle = Polygon()
            pred_by_line = pred_poly_by_line(LineString([t1, t]),
                                             polygon,
                                             graph)
            a = plr_1.union(triangle).union(pred_by_line)
            b = pll_1.difference(triangle).difference(pred_by_line)
            if not a.is_empty:
                a = Partition(polygon=a,
                              sites=first_partition.sites)
                b_sites = difference_by_key(sites, first_partition.sites)
                b = Partition(polygon=b,
                              sites=b_sites)
            else:
                a = Partition(polygon=a,
                              sites={})
                b = Partition(polygon=b,
                              sites=sites)
            return [a, b], graph, next_neighbor(polygon, graph)
        else:
            ps = midpoint(LineString([t1, t2]))
            triangle = Polygon(LineString([t1, ps, t3]))
            a = pred_poly_by_line(LineString([t1, t2]), polygon, graph)
            a = insert_between(Point(ps), (t1, t2), a)
            b = plr_1.union(triangle)
            c = pll_1.difference(triangle).difference(a)
            a = Partition(polygon=a, sites={})
            b = Partition(polygon=b, sites=first_partition.sites)
            c = Partition(polygon=c,
                          sites=difference_by_key(sites,
                                                  first_partition.sites))
            original_polygon = polygon
            polygon = insert_between(point=Point(ps),
                                     vertices=(t1, t2),
                                     polygon=polygon)
            graph = nx.relabel_nodes(graph, {original_polygon: polygon})
            graph = add_point_to_neighbor(point=Point(ps),
                                          vertices=(t1, t2),
                                          polygon=polygon,
                                          graph=graph)
            return [b, a, c], graph, next_neighbor(polygon, graph)
    else:
        t = midpoint(LineString([vertices_and_sites[-2],
                                 vertices_and_sites[0]]))
        line = LineString([t, first_site_point])
        original_polygon = polygon
        polygon = Polygon([*polygon.exterior.coords[:-1], t])
        graph = nx.relabel_nodes(graph, {original_polygon: polygon})
        graph = add_point_to_neighbor(point=Point(t),
                                      vertices=(vertices_and_sites[-2],
                                                vertices_and_sites[0]),
                                      polygon=polygon,
                                      graph=graph)
        plr_1 = plr(polygon=polygon,
                    line=line,
                    graph=graph)
        pll_1 = pll(polygon=polygon,
                    plr_=plr_1,
                    graph=graph)
        plr_1_sites = {first_site_point: sites[first_site_point]}
        pll_1_sites = difference_by_key(sites, plr_1_sites)
        plr_1_partition = Partition(polygon=plr_1,
                                    sites=plr_1_sites)
        pll_1_partition = Partition(polygon=pll_1,
                                    sites=pll_1_sites)
        return ([pll_1_partition, plr_1_partition],
                graph,
                next_neighbor(polygon, graph))


def get_first_partition_for_nonconvex(polygon: Polygon,
                                      start_point: Point,
                                      endpoints: List[Point],
                                      sites: SitesType,
                                      graph: nx.Graph
                                      ) -> Tuple[Partition, LineString]:
    """
    Similar to step Nº3 of the ConvexDivide procedure
    but we also have to take into account predecessors
    """
    lines = list(map(LineString, zip(repeat(start_point), endpoints)))
    plrs = list(map(plr, repeat(polygon), lines, repeat(graph)))
    right_sites_per_vertex = list(sites_per_vertex(endpoints, sites))
    partitions = list(map(Partition, right_sites_per_vertex, plrs))
    first_has_enough_area = compose(has_enough_area, itemgetter(0))
    return find_if_or_last(first_has_enough_area, zip(partitions, lines))


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


def has_enough_area(partition: Partition) -> bool:
    return partition.polygon.area >= cached_sum(*partition.sites.values())


def has_more_than_enough_area(partition: Partition) -> bool:
    return partition.polygon.area > cached_sum(*partition.sites.values())


def pll(polygon: Polygon,
        plr_: Polygon,
        graph: nx.Graph) -> Polygon:
    """
    According to the definition of P_L^l,
    it is CP_L^l plus PredPoly(CP, e) for each edge `e`
    to the left of or containing an endpoint of L.
    This, though, contradicts to what is shown on the Fig.3
    because P_L^l and P_L^r should be overlapping there.
    We assume that no overlapping should occur,
    hence the calculation here differs from the definition.
    """
    pred_poly_ = pred_poly(polygon, graph)
    return pred_poly_.difference(plr_)


def bisection_search_w_point(*,
                             fixed_point: Point,
                             search_line: LineString,
                             polygon: Polygon,
                             is_head_fixed: bool,
                             area_requirement: float,
                             area_tolerance: float,
                             graph: nx.Graph,
                             max_iterations_count: int = 10 ** 9) -> Point:
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
    :param graph: input graph
    :return: right and left parts of the input polygon
    """
    polygon_coordinates = list(polygon.exterior.coords)
    polygon_sides = segments(polygon.exterior)
    start_search_index, search_line = next_enumerate(polygon_sides,
                                                     search_line.equals)
    # checking shortcuts (both sides of the search line):
    endpoint = search_line.boundary[0]
    division_line = (LineString([endpoint, fixed_point]) if is_head_fixed
                     else LineString([fixed_point, endpoint]))
    plr_ = plr(polygon, division_line, graph)
    pll_ = pll(polygon, plr_, graph)
    if math.isclose(plr_.area,
                    area_requirement,
                    rel_tol=area_tolerance):
        return endpoint

    plr_0, pll_0, endpoint_0 = plr_, pll_, endpoint

    endpoint = search_line.boundary[1]
    division_line = (LineString([endpoint, fixed_point]) if is_head_fixed
                     else LineString([fixed_point, endpoint]))
    plr_ = plr(polygon, division_line, graph)
    pll_ = pll(polygon, plr_, graph)
    if math.isclose(plr_.area,
                    area_requirement,
                    rel_tol=area_tolerance):
        return endpoint

    plr_1, pll_1, endpoint_1 = plr_, pll_, endpoint

    # can't satisfy area requirement by current polygon:
    if plr_0.area < area_requirement > plr_1.area:
        return endpoint_0
    if plr_0.area > area_requirement < plr_1.area:
        return endpoint_1

    previous_area = 0

    # standard bisection search
    for _ in range(max_iterations_count):
        endpoint = midpoint(search_line)
        division_line = (LineString([endpoint, fixed_point]) if is_head_fixed
                         else LineString([fixed_point, endpoint]))
        # adding a "snap"-point on the search line - prevents precision errors
        polygon_coordinates.insert(start_search_index + 1, endpoint)
        polygon = Polygon(polygon_coordinates)
        plr_ = plr(polygon, division_line, graph)
        plr_area = plr_.area
        # this can happen when Plr is attached
        # with the last point of search_line
        if plr_area == previous_area:
            return search_line.boundary[1]
        previous_area = plr_area
        if math.isclose(plr_.area,
                        area_requirement,
                        rel_tol=area_tolerance):
            return Point(endpoint)
        # removing "snap"-point
        polygon_coordinates.pop(start_search_index + 1)
        if iff(plr_.area > area_requirement,
               is_head_fixed):
            search_line = LineString([endpoint, search_line.coords[1]])
        else:
            search_line = LineString([search_line.coords[0], endpoint])
    raise ValueError(f"Failed to find partition "
                     f"after {max_iterations_count} steps.")


def plr(polygon: Polygon,
        line: LineString,
        graph: nx.Graph) -> Polygon:
    """
    CP_L^r plus PredPoly(CP, e) for each edge `e`
    to the right of or containing an edpoint of L
    """
    parent_ = get_original_node(polygon, graph)
    right_part_ = right_part(polygon, line)
    if right_part_.is_empty:
        predecessors_and_polygon = right_part_
    else:
        right_edges = edges_on_the_right(right_part_, line)
        pred_polys_by_line = [pred_poly_by_line(line=edge,
                                                polygon=right_part_,
                                                graph=graph,
                                                parent_=parent_)
                              for edge in right_edges]
        pred_polys_by_line = [poly for poly in pred_polys_by_line
                              if not poly.is_empty]
        predecessors_and_polygon = unary_union([right_part_,
                                                *pred_polys_by_line])
    try:
        line_endpoint = line.boundary[1]
    except IndexError:  # line is actually in the form of LineString([a, a])
        line_endpoint = Point(line.coords[0])
    predecessors_ = set(predecessors(parent_, graph))
    predecessors_with_endpoint = polygons_with_point(predecessors_,
                                                     point=line_endpoint)
    pred_polys_from_endpoint = map(pred_poly,
                                   predecessors_with_endpoint,
                                   repeat(graph))
    resulting_polygon = unary_union([predecessors_and_polygon,
                                     *pred_polys_from_endpoint])
    return resulting_polygon if resulting_polygon.area > 1e-16 else Polygon()


def get_original_node(node: Polygon,
                      graph: nx.Graph) -> Polygon:
    """
    We assume that there are no overlapping nodes in the graph,
    hence any two nodes having same leftmost-lowest vertices,
    rightmost-highest vertices, and having those vertices
    along with centroids oriented the same way are the same nodes.
    """
    node_leftmost_lowest = leftmost_lowest(node)
    node_rightmost_highest = rightmost_highest(node)
    is_node_ring_ccw = LinearRing([node_leftmost_lowest,
                                   node_rightmost_highest,
                                   *node.centroid.coords]).is_ccw
    for candidate in graph:
        candidate_leftmost_lowest = leftmost_lowest(candidate)
        candidate_rightmost_highest = rightmost_highest(candidate)
        is_candidate_ring = LinearRing([candidate_leftmost_lowest,
                                        candidate_rightmost_highest,
                                        *candidate.centroid.coords]).is_ccw
        if (node_leftmost_lowest == candidate_leftmost_lowest
                and node_rightmost_highest == candidate_rightmost_highest
                and is_node_ring_ccw == is_candidate_ring):
            return candidate
    else:
        raise ValueError("Couldn't find necessary node in the graph")


def leftmost_lowest(polygon: Polygon) -> Tuple[float, float]:
    return min(polygon.exterior.coords)


def rightmost_highest(polygon: Polygon) -> Tuple[float, float]:
    return max(polygon.exterior.coords)


def edges_on_the_right(polygon: Polygon,
                       line: LineString) -> Iterator[LineString]:
    for segment in segments(polygon.exterior):
        if is_on_the_right(segment, line):
            yield segment


def pred_poly_by_line(line: LineString,
                      polygon: Polygon,
                      graph: nx.Graph,
                      parent_: Optional[Polygon] = None) -> Polygon:
    """
    According to the definition it is a collection of predecessors
    of the polygon that are reachable by crossing the specified edge.
    """
    graph = graph.copy()
    # the following is needed when we check only a part of a polygon
    # but could be probably rewritten using `get_original_node`
    if polygon not in graph:
        if parent_ is not None:
            parent_ = parent(parent_, graph)
        if parent_ is None:
            parent_ = parent(polygon, graph)
        if parent_ is not None:
            polygon = parent_
        else:
            # second-guessing that the polygon doesn't have predecessors
            return Polygon()

    all_predecessors = set(predecessors(node=polygon,
                                        graph=graph))
    neighbors = set(nx.neighbors(graph, polygon))
    predecessors_neighbors = neighbors & all_predecessors
    is_touching_polygon_by_line = partial(are_touching_by_line,
                                          polygon=polygon,
                                          graph=graph,
                                          line=line)
    line_touching_nodes = set(filter(is_touching_polygon_by_line,
                                     predecessors_neighbors))
    graph.remove_node(polygon)
    components = nx.connected_components(graph)
    components_with_neighbors = [component for component in components
                                 if any(node in component
                                        for node in line_touching_nodes)]
    components_nodes = flatten(components_with_neighbors)
    components_nodes = [node for node in components_nodes
                        if node in all_predecessors]
    return unary_union(components_nodes)


def parent(polygon: Polygon,
           graph: nx.Graph) -> Optional[Polygon]:
    """
    Finds a polygon in a graph that contains given polygon
    Note that this is prone to precision errors,
    and ideally should be replaced by `get_original_node`
    """
    return next(filter(polygon.within, graph.nodes), None)


def predecessors(node: Node,
                 graph: nx.Graph) -> Iterator[Node]:
    """
    For a graph (N1, N2, ..., Nm) and its node Nj
    returns all Nj's predecessor nodes (N1, N2, ..., Nj-1).
    Note that this definition is different
    from the definition of predecessors for directed graphs.
    """
    if node not in graph.nodes:
        raise ValueError(f"Can't return predecessors for a node "
                         f"that is not in the graph.\n"
                         f"Got the following node: {node.wkt}")
    not_a_node = partial(ne, node)
    yield from takewhile(not_a_node, graph.nodes)


def are_touching_by_line(candidate: Polygon,
                         polygon: Polygon,
                         graph: nx.Graph,
                         line: LineString) -> bool:
    touching_line = graph[polygon][candidate]['side']
    return are_touching(line, touching_line)


def polygons_with_point(polygons: Iterable[Polygon],
                        point: Point) -> Iterator[Polygon]:
    """with the point on their exteriors"""
    for polygon in polygons:
        exterior = polygon.exterior
        if exterior.contains(point):
            yield polygon


def pred_poly(polygon: Polygon,
              graph: nx.Graph) -> Polygon:
    return unary_union(pred_polys(polygon, graph=graph))


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


def prepend_partitions(partitions: Iterable[Partition],
                       graph: nx.Graph,
                       node_to_attach_to: Optional[Polygon]) -> nx.Graph:
    """Prepends partitions to the specified node in parallel"""
    new_graph = nx.Graph()
    empty_parts = [partition for partition in partitions
                   if partition.polygon.is_empty]
    not_empty_partitions = [partition for partition in partitions
                            if not partition.polygon.is_empty]
    new_nodes = [partition.polygon for partition in not_empty_partitions]

    new_graph.add_nodes_from(chain(new_nodes, graph.nodes))

    if node_to_attach_to is not None:
        new_edges = zip(new_nodes, repeat(node_to_attach_to))
        new_graph.add_edges_from(chain(new_edges, graph.edges))
    else:
        new_graph.add_edges_from(graph.edges)

    # copy node attributes:
    for node in graph:
        new_graph.nodes[node].update(graph.nodes[node])
    # copy edge attributes:
    for edge in graph.edges:
        new_graph.edges[edge]['side'] = graph.edges[edge]['side']
    # add new node attributes:
    for partition in not_empty_partitions:
        new_graph.nodes[partition.polygon].update(partition.sites)

    if node_to_attach_to is not None:
        # add new edge attributes:
        for partition in not_empty_partitions:
            new_edge = partition.polygon, node_to_attach_to
            touching_edges = touching_sides(*new_edge)
            if touching_edges is not None:
                new_graph.edges[new_edge]['side'] = touching_edges[0]

    if node_to_attach_to is None:
        node_to_attach_to = first(new_graph)
    # assign empty partitions' sites to their successor:
    for partition in empty_parts:
        new_graph.nodes[node_to_attach_to].update(partition.sites)
    return new_graph


def prepend_partitions_inline(partitions: Iterable[Partition],
                              graph: nx.Graph,
                              node_to_attach_to: Optional[Polygon]
                              ) -> nx.Graph:
    """Sequentially prepends partitions to the specified node"""
    new_graph = nx.Graph()
    empty_parts = [partition for partition in partitions
                   if partition.polygon.is_empty]
    new_parts = [partition for partition in partitions
                 if not partition.polygon.is_empty]
    new_nodes = [partition.polygon for partition in new_parts]

    new_graph.add_nodes_from(chain(new_nodes, graph.nodes))

    new_edges = (pairwise(new_nodes)
                 if node_to_attach_to is None
                 else pairwise(chain(new_nodes, [node_to_attach_to])))
    new_edges = list(new_edges)
    new_graph.add_edges_from(chain(new_edges, graph.edges))

    # copy node attributes:
    for node in graph:
        new_graph.nodes[node].update(graph.nodes[node])
    # copy edge attributes:
    for edge in graph.edges:
        new_graph.edges[edge]['side'] = graph.edges[edge]['side']
    # add new node attributes:
    for partition in new_parts:
        new_graph.nodes[partition.polygon].update(partition.sites)

    # add new edge attributes:
    for new_edge in new_edges:
        touching_edges = touching_sides(*new_edge)
        if touching_edges is not None:
            new_graph.edges[new_edge]['side'] = touching_edges[0]

    if node_to_attach_to is None:
        return new_graph

    # assign empty partitions' sites to their successor:
    for partition in empty_parts:
        new_graph.nodes[node_to_attach_to].update(partition.sites)
    return new_graph


def is_area_incomplete(polygon: Polygon,
                       sites: SitesType) -> bool:
    return polygon.area < sum(sites.values())


def to_ordered(graph: nx.Graph) -> nx.Graph:
    """Orders a graph as in OrderPieces"""
    ordered_nodes = nx.dfs_postorder_nodes(graph)
    reordered_graph = nx.Graph()
    reordered_graph.add_nodes_from(ordered_nodes)
    reordered_graph.add_edges_from(graph.edges)
    # copy edge attributes:
    for edge in graph.edges:
        reordered_graph.edges[edge]['side'] = graph.edges[edge]['side']
    return reordered_graph


def attach_sites(graph: nx.Graph,
                 sites: SitesType) -> nx.Graph:
    """
    Attaches sites to the nodes of the graph
    by choosing rarest site for each node.
    """
    graph = graph.copy()

    sites_counts = Counter()
    sites_per_polygon = {}
    for polygon in graph.nodes:
        polygon_sites = find_sites(polygon=polygon,
                                   sites=sites)
        sites_per_polygon[polygon] = polygon_sites
        sites_counts.update(polygon_sites.keys())

    for polygon, polygon_sites in sites_per_polygon.items():
        if not polygon_sites:
            continue

        sites_to_add = {}
        for point, requirement in polygon_sites.items():
            if sites_counts[point] == 1:
                sites_to_add[point] = requirement
                sites_counts.pop(point)
        if sites_to_add:
            graph.nodes[polygon].update(sites_to_add)
            continue

        sites_in_counter = [site for site in polygon_sites
                            if site in sites_counts]
        if not sites_in_counter:
            continue
        rarest_site_point = min(sites_in_counter, key=sites_counts.get)
        sites_counts.pop(rarest_site_point)
        for site in polygon_sites:
            if site in sites_counts:
                sites_counts[site] -= 1
        site = {rarest_site_point: polygon_sites[rarest_site_point]}
        graph.nodes[polygon].update(site)
    return graph


def find_sites(polygon: Polygon,
               sites: SitesType) -> SitesType:
    is_point_in_polygon = compose(polygon.intersects, itemgetter(0))
    sites_in_polygon = filter(is_point_in_polygon, sites.items())
    return dict(sites_in_polygon)


def divide_on_the_go(graph: nx.Graph,
                     requirements: List[float],
                     with_gradual_search: bool = False
                     ) -> Iterator[Tuple[Polygon, Tuple[Point, float]]]:
    """
    Splits a polygon defined by the graph
    by assigning sites (in the same order as given) on the go
    """
    graph = graph.copy()
    requirements = iter(requirements)

    area_incomplete_polygons = defaultdict(list)
    pseudo_sites_relations = {}

    assigned_sites_locations = set()

    graph_iterator = iter(graph)
    while True:
        if not graph:
            return
        try:
            polygon: Polygon = next(graph_iterator)
        except StopIteration:
            return
        sites: SitesType = graph.nodes[polygon]
        if not sites:
            neighbor = next_neighbor(polygon, graph=graph)
            if neighbor is None:
                site_point = find_free_point(polygon, assigned_sites_locations)
            else:
                edge = graph[polygon][neighbor]['side']
                furthest_segment = max(segments(polygon.exterior),
                                       key=edge.distance)
                site_point = Point(midpoint(furthest_segment))
            try:
                sites = {site_point: next(requirements)}
                assigned_sites_locations.add(site_point)
            except StopIteration:
                continue
        pred_polygons = pred_polys(polygon, graph=graph)
        pred_polygon = unary_union(pred_polygons)
        if is_area_complete(pred_polygon, sites=sites):
            if len(sites) == 1:
                graph.remove_nodes_from(pred_polygons)
                site_point, requirement = first(sites.items())
                if site_point not in pseudo_sites_relations:
                    yield (pred_polygon, *sites.items())
                else:
                    original_site = pseudo_sites_relations[site_point]
                    point, requirement = first(original_site.items())
                    # TODO: should it be pred_polygons also
                    all_related_polygons = [*area_incomplete_polygons[point],
                                            *pred_polygons]
                    resulting_polygon = unary_union(all_related_polygons)
                    yield resulting_polygon, (point, requirement)
            else:
                neighbor = next_neighbor(polygon, graph=graph)
                graph = update_graph(polygon=polygon,
                                     sites=sites,
                                     edge=None,
                                     neighbor=neighbor,
                                     graph=graph)
        elif is_area_incomplete(pred_polygon, sites=sites):
            neighbor = next_neighbor(polygon, graph=graph)
            edge = graph[polygon][neighbor]['side']
            if len(sites) == 1:
                site_point, requirement = first(sites.items())
                site_point, initial_requirement = first(
                    pseudo_sites_relations.get(site_point, sites).items())
                area_incomplete_polygons[site_point].append(pred_polygon)
                graph.remove_nodes_from(pred_polygons)
                pseudo_requirement = requirement - pred_polygon.area
                pseudo_site_point = Point(midpoint(edge))
                pseudo_sites_relations[pseudo_site_point] = (
                    {site_point: initial_requirement})
                graph.nodes[neighbor].update(
                    {pseudo_site_point: pseudo_requirement})
            else:
                graph = update_graph(polygon=polygon,
                                     sites=sites,
                                     edge=edge,
                                     neighbor=neighbor,
                                     graph=graph)
        else:
            neighbor = next_neighbor(polygon, graph=graph)
            edge = (None if neighbor is None
                    else graph[polygon][neighbor]['side'])
            graph = update_graph(polygon=polygon,
                                 sites=sites,
                                 edge=edge,
                                 neighbor=neighbor,
                                 graph=graph,
                                 with_gradual_search=with_gradual_search)
        graph_iterator = iter(graph)


def find_free_point(polygon: Polygon,
                    taken_points: Set[Point]) -> Point:
    vertices = map(Point, polygon.exterior.coords)
    for point in vertices:
        if point not in taken_points:
            return point
    raise ValueError("Couldn't find a free vertex for a site")


def gradual_search(polygon: Polygon,
                   sites: SitesType,
                   edge: LineString,
                   *,
                   tolerance: float = 1e-4,
                   max_iterations_count: int = 10 ** 8) -> List[Partition]:
    """
    Instead of rotating a line by vertices,
    it moves a line to/from the edge to the next neighbor.
    """
    if not polygon.exterior.is_ccw:
        raise ValueError("Polygon division is implemented only for polygons "
                         "oriented counter-clockwise")
    if len(sites) != 1:
        raise ValueError("This function is only for partitions with 1 site")
    site_point, requirement = first(sites.items())
    if requirement == 0 or requirement >= polygon.area:
        raise ValueError(f"Invalid requirement: {requirement} "
                         f"for an area {polygon.area}")
    edges_with_site = [segment for segment in segments(polygon.exterior)
                       if site_point.touches(segment)]
    if edge is None:
        furthest_edge = edges_with_site[0]
        edge = max(segments(polygon.exterior), key=furthest_edge.distance)
    else:
        furthest_edge = max(edges_with_site, key=edge.distance)
    get_rotating_line = left.applier(rotating_splitter,
                                     source_line=furthest_edge,
                                     target_line=LineString(edge.coords[::-1]),
                                     length=polygon.exterior.length)
    f0, f1 = 0, 1
    for _ in range(max_iterations_count):
        f = f0 + (f1 - f0) / 2
        splitter = get_rotating_line(f)
        right_polygon, left_polygon = right_left_parts(polygon, splitter)
        right_polygon = remove_duplicate_points(snap(right_polygon,
                                                     polygon.exterior.coords))
        left_polygon = remove_duplicate_points(snap(left_polygon,
                                                    polygon.exterior.coords))
        if math.isclose(right_polygon.area,
                        requirement,
                        rel_tol=tolerance):
            right_partition = Partition(polygon=right_polygon,
                                        sites=sites)
            left_partition = Partition(polygon=left_polygon,
                                       sites={})
            return [right_partition, left_partition]
        elif right_polygon.area > requirement:
            f1 = f
        else:
            f0 = f
    raise ValueError("Couldn't find appropriate partition of the polygon")


def snap(polygon: Polygon,
         snap_points: Iterable[Tuple[float, float]],
         min_dist: float = 1e-9) -> Polygon:
    """
    Used to remove points from newly created polygons
    that are too close to original vertices
    """
    if polygon.is_empty:
        return polygon
    coords_to_return = []
    for new_point in polygon.exterior.coords:
        for snap_point in snap_points:
            if (Point(new_point).distance(Point(snap_point)) < min_dist
                    and snap_point not in coords_to_return):
                coords_to_return.append(snap_point)
                break
        else:
            coords_to_return.append(new_point)
    return Polygon(coords_to_return)


def remove_duplicate_points(polygon: Polygon) -> Polygon:
    if polygon.is_empty:
        return polygon
    coords_to_return = []
    for coord in polygon.exterior.coords:
        if coord not in coords_to_return:
            coords_to_return.append(coord)
    return Polygon(coords_to_return)
