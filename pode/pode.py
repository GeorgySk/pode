from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from itertools import (chain,
                       combinations)
from numbers import Real
from typing import (DefaultDict,
                    Dict,
                    FrozenSet,
                    Iterable,
                    Iterator,
                    List,
                    NamedTuple,
                    Optional,
                    Tuple,
                    Union)

import networkx as nx
from gon.base import (Contour,
                      EMPTY,
                      Multipoint,
                      Multisegment,
                      Orientation,
                      Point,
                      Polygon,
                      Segment,
                      Triangulation)
from gon.hints import Maybe
from ground.base import get_context

from pode.hints import ConvexDivisorType
from pode.utils import (cut,
                        edges,
                        order_convex_contour_points,
                        orient,
                        rotate,
                        shrink_collinear_vertices,
                        splitter_point,
                        to_fractions,
                        unite)


class Requirement(NamedTuple):
    """Area requirement with an optional point (site)"""
    area: Real
    point: Optional[Point] = None


class PolygonsSet(FrozenSet[Polygon]):
    @cached_property
    def area(self) -> Fraction:
        return sum(polygon.area for polygon in self)


@dataclass(frozen=True)
class RootedPolygons:
    root: Maybe[Polygon]
    predecessors: PolygonsSet

    @cached_property
    def area(self) -> Fraction:
        root_area = 0 if self.root is EMPTY else self.root.area
        return root_area + self.predecessors.area


class Graph(nx.DiGraph):
    def __init__(self, **attr) -> None:
        super().__init__(**attr)

    @classmethod
    def from_undirected(cls, graph: nx.Graph) -> 'Graph':
        """
        Constructs an ordered and directed graph from an undirected
        graph.
        """
        ordered_graph = cls()
        ordered_nodes = list(nx.dfs_postorder_nodes(graph))
        ordered_graph.add_nodes_from(ordered_nodes)
        directed_edges = (sorted(edge, key=ordered_nodes.index)
                          for edge in graph.edges)
        ordered_graph.add_edges_from(directed_edges)
        for *edge, side in graph.edges.data('side'):
            if edge in ordered_graph.edges:
                ordered_graph.edges[edge]['side'] = side
            else:
                ordered_graph.edges[edge[::-1]]['side'] = side
        # there can't be more than one outcoming edges from a node
        for node in ordered_graph:
            if len(ordered_graph[node]) < 2:
                continue
            most_immediate_successor = cls.next_neighbor(ordered_graph,
                                                         node)
            bad_edges_nodes = (set(ordered_graph[node])
                               - {most_immediate_successor})
            for neighbor in bad_edges_nodes:
                ordered_graph.remove_edge(node, neighbor)
        for node in ordered_graph:
            ordered_graph.nodes[node].update({'requirements': frozenset()})
        return ordered_graph

    def next_neighbor(self, polygon: Polygon) -> Optional[Polygon]:
        """
        Most immediate successor which is a neighbor of a given node
        :param polygon: node of the graph
        :return: next neighbor node or `None` if not found
        """
        neighbors = (node for node in self.nodes if node in self[polygon])
        return next(neighbors, None)

    def pred_polys(self, polygon: Polygon) -> PolygonsSet:
        """
        PredPoly is defined in the paper as the current polygon plus
        all its ancestors. In the paper ancestors are called
        "predecessors".
        :param polygon: node of the graph
        :return: a set of polygons that are parts of PredPoly
        """
        return PolygonsSet({polygon, *nx.ancestors(self, polygon)})

    def _edges_with_neighbors(self, polygon: Polygon) -> List[Segment]:
        edges = []
        # neighbors() and successors() are the same in nx.DiGraph
        neighbors = chain(self.neighbors(polygon), self.predecessors(polygon))
        for neighbor in neighbors:
            try:
                edges.append(self[polygon][neighbor]['side'])
            except KeyError:
                edges.append(self[neighbor][polygon]['side'])
        return edges

    def neighbor_edges_vertices(self, polygon: Polygon) -> List[Point]:
        vertices = []
        for edge in self._edges_with_neighbors(polygon):
            vertices.extend((edge.start, edge.end))
        return vertices

    def _predecessors_by_edges(self,
                               polygon: Polygon) -> Dict[Segment, Polygon]:
        """
        Mapping of the most immediate ancestors by the edges they
        touch the given polygon
        """
        return {self[predecessor][polygon]['side']: predecessor
                for predecessor in self.predecessors(polygon)}

    def _pred_poly_by_edges(self,
                            polygon: Polygon) -> Dict[Segment, PolygonsSet]:
        """
        Mapping of edges to ancestors that can be reached from the
        given polygon
        """
        predecessors_by_edges = self._predecessors_by_edges(polygon)
        return {edge: self.pred_polys(predecessor)
                for edge, predecessor in predecessors_by_edges.items()}

    def pred_poly_by_line(self,
                          polygon: Polygon,
                          splitter: Segment) -> PolygonsSet:
        pred_poly_by_edges = self._pred_poly_by_edges(polygon)
        vertices_in_interval = cut(polygon.border,
                                   splitter.start,
                                   splitter.end)
        edges_in_interval = Multisegment([*map(Segment,
                                               vertices_in_interval[:-1],
                                               vertices_in_interval[1:])])
        return PolygonsSet({part
                            for edge, pred_poly in pred_poly_by_edges.items()
                            for part in pred_poly
                            if isinstance(edge & edges_in_interval, Segment)})

    def plr(self,
            polygon: Polygon,
            splitter: Segment) -> RootedPolygons:
        """
        Portion of the current polygon to the right of the splitter
        plus all immediate ancestors accessible through the polygon
        edges that lie on the right of the splitter
        """
        vertices_in_interval = cut(polygon.border,
                                   splitter.start,
                                   splitter.end)
        if len(vertices_in_interval) < 3:
            current_polygon_part = EMPTY
        else:
            contour = Contour(shrink_collinear_vertices(
                Contour(vertices_in_interval)))
            current_polygon_part = Polygon(contour)
        pred_poly_by_line = self.pred_poly_by_line(polygon, splitter=splitter)
        return RootedPolygons(root=current_polygon_part,
                              predecessors=pred_poly_by_line)

    def pll(self,
            polygon: Polygon,
            splitter: Segment) -> RootedPolygons:
        return self.plr(polygon, Segment(splitter.end, splitter.start))

    def prepend_two(self,
                    parts: Tuple[nx.DiGraph, nx.DiGraph],
                    target: Optional[Polygon]) -> 'Graph':
        new_graph = Graph()
        new_nodes = [node for part in parts for node in part]
        new_edges_and_sides = [(edge, part.edges[edge]['side'])
                               for part in parts for edge in part.edges]
        new_graph.add_nodes_from(chain(new_nodes, self.nodes))
        for edge, side in new_edges_and_sides:
            new_graph.add_edge(*edge, side=side)
        for *edge, side in self.edges.data('side'):
            new_graph.add_edge(*edge, side=side)
        if target is not None:
            for node in new_nodes:
                intersection = node & target
                if isinstance(intersection, Segment):
                    new_graph.add_edge(node, target, side=intersection)
        # Copying attributes containing requirements per node
        for graph in (*parts, self):
            for node in graph:
                new_graph.nodes[node].update(graph.nodes[node])
        return new_graph

    def prepend_three(self,
                      parts: Tuple[nx.DiGraph, nx.DiGraph, nx.DiGraph],
                      target: Optional[Polygon]) -> 'Graph':
        new_graph = Graph()
        new_nodes = [node for part in parts for node in part]
        new_edges_and_sides = [(edge, part.edges[edge]['side'])
                               for part in parts for edge in part.edges]
        new_graph.add_nodes_from(chain(new_nodes, self.nodes))
        first_connector_node = list(parts[0])[-1]
        new_edges_and_sides_between_parts = []
        for node in parts[1]:
            intersection = node & first_connector_node
            if isinstance(intersection, Segment):
                new_edges_and_sides_between_parts.append(
                    ((first_connector_node, node), intersection))
        second_connector_node = list(parts[2])[-1]
        for node in parts[1]:
            intersection = node & second_connector_node
            if isinstance(intersection, Segment):
                new_edges_and_sides_between_parts.append(
                    ((node, second_connector_node), intersection))
        for edge, side in chain(new_edges_and_sides,
                                new_edges_and_sides_between_parts):
            new_graph.add_edge(*edge, side=side)
        for *edge, side in self.edges.data('side'):
            new_graph.add_edge(*edge, side=side)
        if target is not None:
            intersection = second_connector_node & target
            if isinstance(intersection, Segment):
                new_graph.add_edge(second_connector_node,
                                   target,
                                   side=intersection)
        # Copying attributes containing requirements per node
        for graph in (*parts, self):
            for node in graph:
                new_graph.nodes[node].update(graph.nodes[node])
        return new_graph

    def subgraph(self, polygons: Iterable[Polygon]) -> 'Graph':
        """
        Same as Graph.subgraph from NetworkX but preserves the order
        of the nodes as in the given iterable.
        """
        new_graph = Graph()
        ordered_polygons = sorted(polygons, key=list(self).index)
        new_graph.add_nodes_from(ordered_polygons)
        polygons = set(ordered_polygons)
        for *edge, side in self.edges.data('side'):
            if all(polygon in polygons for polygon in edge):
                new_graph.add_edge(*edge, side=side)
        # Copying attributes containing requirements per node
        for node in new_graph:
            new_graph.nodes[node].update(self.nodes[node])
        return new_graph


def divide(polygon: Polygon,
           requirements: List[Requirement],
           convex_divisor: ConvexDivisorType
           = Triangulation.constrained_delaunay
           ) -> List[Polygon]:
    """
    Divides given polygon for the given list of `Requirement` objects
    :param polygon: input polygon
    :param requirements: area requirements with optional points
    :param convex_divisor: function to split input polygon to convex
    parts
    :return: a list of parts of the polygon in the order corresponding
    to `requirements`
    """
    validate_requirements(requirements, polygon)
    if len(requirements) == 1:
        return [polygon]
    polygon = to_fractions(polygon)
    requirements = [requirement._replace(point=to_fractions(requirement.point))
                    if requirement.point is not None
                    else requirement
                    for requirement in requirements]
    areas = [requirement.area for requirement in requirements]
    normalized_areas = normalize_areas(areas, polygon.area)
    requirements = [requirement._replace(area=area)
                    for requirement, area in zip(requirements,
                                                 normalized_areas)]
    resulting_parts: List[Optional[Polygon]] = [None] * len(requirements)
    reference_requirements = requirements.copy()
    pseudo_to_original_requirements: Dict[Requirement, Requirement] = {}
    incomplete_parts: DefaultDict[Requirement,
                                  List[Polygon]] = defaultdict(list)
    site_points = [requirement.point for requirement in requirements
                   if requirement.point is not None]
    graph = to_graph(polygon,
                     extra_points=site_points,
                     convex_divisor=convex_divisor)
    graph = Graph.from_undirected(graph)
    remaining_nodes = iter(graph.nodes)
    while graph:
        current_polygon: Polygon = orient(next(remaining_nodes))
        pred_polys = graph.pred_polys(current_polygon)
        current_sites, requirements = assign_requirements(
            polygon=current_polygon,
            requirements=requirements,
            graph=graph)
        if not current_sites:
            continue
        required_area = sum(requirement.area for requirement in current_sites)
        if len(current_sites) == 1 and pred_polys.area < required_area:
            neighbor = graph.next_neighbor(current_polygon)
            edge = graph[current_polygon][neighbor]['side']
            requirement = list(current_sites)[0]
            pseudorequirement = Requirement(requirement.area - pred_polys.area,
                                            point=edge.centroid)
            pred_poly = unite(*pred_polys)
            original_requirement = pseudo_to_original_requirements.get(
                requirement, requirement)
            incomplete_parts[original_requirement].append(pred_poly)
            graph.remove_nodes_from(pred_polys)
            pseudo_to_original_requirements[
                pseudorequirement] = original_requirement
            graph.nodes[neighbor]['requirements'] |= {pseudorequirement}
        elif len(current_sites) == 1 and pred_polys.area == required_area:
            requirement = list(current_sites)[0]
            graph.remove_nodes_from(pred_polys)
            if requirement not in pseudo_to_original_requirements:
                resulting_part = unite(*pred_polys)
            else:
                requirement = pseudo_to_original_requirements[requirement]
                resulting_part = unite(*pred_polys,
                                       *incomplete_parts[requirement])
            for index, (reference_requirement, part) in enumerate(
                    zip(reference_requirements, resulting_parts)):
                if requirement == reference_requirement and part is None:
                    resulting_parts[index] = resulting_part
                    break
            else:
                for index, (reference_requirement, part) in enumerate(
                        zip(reference_requirements, resulting_parts)):
                    if (requirement.area == reference_requirement.area
                            and part is None):
                        resulting_parts[index] = resulting_part
                        break
        else:
            neighbor = graph.next_neighbor(current_polygon)
            extra_points = graph.neighbor_edges_vertices(current_polygon)
            if neighbor is None:
                vertices = list({*current_polygon.border.vertices,
                                 *(site.point for site in current_sites),
                                 *extra_points})
                if current_sites:
                    vertices = order_by_sites(vertices,
                                              list(current_sites)[0].point)
            else:
                edge = graph[current_polygon][neighbor]['side']
                vertices = list({*current_polygon.border.vertices,
                                 *extra_points,
                                 *(site.point for site in current_sites),
                                 edge.start,
                                 edge.end})
                vertices = order_by_edge(vertices, edge)
            parts = split(polygon=current_polygon,
                          vertices=vertices,
                          sites=current_sites,
                          graph=graph)
            graph.remove_nodes_from(pred_polys)
            graph = (graph.prepend_two(parts, neighbor) if len(parts) == 2
                     else graph.prepend_three(parts, neighbor))
        remaining_nodes = iter(graph.nodes)
    return resulting_parts


def validate_requirements(requirements: List[Requirement],
                          polygon: Polygon) -> None:
    if sum(requirement.area for requirement in requirements) != 1:
        raise ValueError("Area requirements should sum up to 1.")
    sites = [requirement.point for requirement in requirements
             if requirement.point is not None]
    sites_set = set(sites)
    if any(site not in polygon for site in sites_set):
        raise ValueError("Not all the sites lie in the polygon.")
    if len(sites_set) != len(sites):
        raise ValueError("Sites cannot share their locations.")


def assign_requirements(*,
                        polygon: Polygon,
                        requirements: List[Requirement],
                        graph: Graph) -> Tuple[FrozenSet[Requirement],
                                               List[Requirement]]:
    requirements_with_points = {requirement for requirement in requirements
                                if requirement.point is not None}
    preassigned_requirements_with_points = {
        site for node in graph for site in graph.nodes[node]['requirements']}
    if (not requirements_with_points
            and not preassigned_requirements_with_points):
        bare_requirements = [requirement for requirement in requirements
                             if requirement not in requirements_with_points]
        if not bare_requirements:
            return frozenset({}), requirements
        *requirements, requirement = requirements
        point = polygon.border.vertices[0]
        requirement = Requirement(requirement.area, point=point)
        return frozenset({requirement}), requirements
    current_preassigned_requirements = graph.nodes[polygon]['requirements']
    ancestors = nx.ancestors(graph, polygon)
    remaining_nodes = set(graph.nodes) - {polygon, *ancestors}
    if not remaining_nodes:
        leftover_requirements = [
            requirement for requirement in requirements
            if requirement not in requirements_with_points]
        return (current_preassigned_requirements | requirements_with_points,
                leftover_requirements)
    current_requirements = {requirement
                            for requirement in requirements_with_points
                            if requirement.point in polygon}
    polygon_requirements_only = {requirement
                                 for requirement in current_requirements
                                 if all(requirement.point not in node
                                        for node in remaining_nodes)}
    if polygon_requirements_only or current_preassigned_requirements:
        requirements_to_return = frozenset(polygon_requirements_only
                                           | current_preassigned_requirements)
        leftover_requirements = [
            requirement for requirement in requirements
            if requirement not in requirements_to_return]
        return requirements_to_return, leftover_requirements
    if not current_requirements:
        return frozenset({}), requirements
    current_requirement = current_requirements.pop()
    leftover_requirements = [requirement for requirement in requirements
                             if requirement != current_requirement]
    return frozenset([current_requirement]), leftover_requirements


def split(*,
          polygon: Polygon,
          vertices: List[Point],
          sites: FrozenSet[Requirement],
          graph: Graph
          ) -> Union[Tuple[nx.DiGraph, nx.DiGraph],
                     Tuple[nx.DiGraph, nx.DiGraph, nx.DiGraph]]:
    """Splits a PredPoly to 2 or 3 parts for further division"""
    if polygon.border.orientation is not Orientation.COUNTERCLOCKWISE:
        raise ValueError("Polygon division is implemented only for polygons "
                         "oriented counter-clockwise")
    sites_locations = set(site.point for site in sites)
    first_site_index, first_site_point = next(
        ((index, vertex) for index, vertex in enumerate(vertices[1:], start=1)
         if vertex in sites_locations), (0, vertices[0]))
    first_head_index = max(1, first_site_index)
    plrs = [graph.plr(polygon, Segment(vertices[0], vertex))
            for vertex in vertices[first_head_index:]]
    head_indices = range(first_head_index, len(vertices))
    heads = vertices[first_head_index:]
    if len(sites) == 1:
        sites_per_vertex = [sites] * len(vertices)
        requirements = [list(sites)[0].area] * len(vertices)
    else:
        sites_per_vertex = list(to_requirements_per_vertex(heads, sites))
        requirements = [sum(site.area for site in sites_)
                        for sites_ in sites_per_vertex]
    for (plr, head_index,
         requirement, right_sites) in zip(plrs, head_indices,
                                          requirements, sites_per_vertex):
        if plr.area >= requirement:
            break

    if plr.area >= requirement:
        if head_index == first_site_index:
            origins_indices = range(first_site_index - 1, -1, -1)
            origins = vertices[first_site_index - 1::-1]
            plrs = [graph.plr(polygon, Segment(origin, first_site_point))
                    for origin in origins]
            for plr, origin_index in zip(plrs, origins_indices):
                if plr.area >= requirement:
                    break
            pivot_index = head_index
            low_area_index = origin_index + 1
            high_area_index = origin_index
            splitter = Segment(vertices[low_area_index], vertices[head_index])
        else:
            pivot_index = 0
            low_area_index = head_index - 1
            high_area_index = head_index
            splitter = Segment(vertices[pivot_index], vertices[low_area_index])
        pivot_point = vertices[pivot_index]
        low_area_point = vertices[low_area_index]
        high_area_point = vertices[high_area_index]
        triangle = Polygon(Contour((pivot_point,
                                    low_area_point,
                                    high_area_point)))
        plr_1 = graph.plr(polygon, splitter)
        pll_1 = graph.pll(polygon, splitter)
        edge = (Segment(low_area_point, high_area_point)
                if pivot_index == 0
                else Segment(high_area_point, low_area_point))
        if plr_1.area + triangle.area > requirement:
            triangle_requirement = requirement - plr_1.area
            t = splitter_point(requirement=triangle_requirement,
                               pivot=pivot_point,
                               low_area_point=low_area_point,
                               high_area_point=high_area_point)
            triangle = Polygon(Contour([low_area_point, t, pivot_point]))
            a = to_subgraph(predecessors=plr_1.predecessors,
                            old_root=polygon,
                            new_root=plr_1.root | triangle,
                            sites=right_sites,
                            graph=graph)
            b = to_subgraph(predecessors=pll_1.predecessors,
                            old_root=polygon,
                            new_root=pll_1.root - triangle,
                            sites=sites - right_sites,
                            graph=graph)
            return a, b
        pred_by_line = graph.pred_poly_by_line(polygon, edge)
        plr_and_pred_by_line_area = plr_1.area + pred_by_line.area
        if plr_and_pred_by_line_area < requirement:
            requirement = requirement - plr_and_pred_by_line_area
            t = splitter_point(requirement=requirement,
                               pivot=pivot_point,
                               low_area_point=low_area_point,
                               high_area_point=high_area_point)
            triangle = Polygon(Contour([low_area_point, t, pivot_point]))
            edge = (Segment(low_area_point, t) if pivot_index == 0
                    else Segment(t, low_area_point))
            pred_by_line = graph.pred_poly_by_line(polygon, edge)
            a = to_subgraph(predecessors=plr_1.predecessors | pred_by_line,
                            old_root=polygon,
                            new_root=plr_1.root | triangle,
                            sites=right_sites,
                            graph=graph)
            b = to_subgraph(predecessors=pll_1.predecessors - pred_by_line,
                            old_root=polygon,
                            new_root=pll_1.root - triangle,
                            sites=sites - right_sites,
                            graph=graph)
            return a, b
        else:
            ps = Multipoint([low_area_point, high_area_point]).centroid
            triangle = Polygon(Contour([low_area_point, ps, pivot_point]))
            edge = (Segment(low_area_point, high_area_point)
                    if pivot_index == 0
                    else Segment(high_area_point, low_area_point))
            pred_by_line = graph.pred_poly_by_line(polygon, edge)
            a = graph.subgraph(pred_by_line)
            b = to_subgraph(predecessors=plr_1.predecessors,
                            old_root=polygon,
                            new_root=plr_1.root | triangle,
                            sites=right_sites,
                            graph=graph)
            c = to_subgraph(predecessors=pll_1.predecessors - pred_by_line,
                            old_root=polygon,
                            new_root=pll_1.root - triangle,
                            sites=sites - right_sites,
                            graph=graph)
            return b, a, c
    else:
        t = Multipoint([vertices[-1], vertices[0]]).centroid
        splitter = Segment(t, first_site_point)
        plr_1 = graph.plr(polygon, splitter)
        pll_1 = graph.pll(polygon, splitter)
        first_site_set = sites_per_vertex[0]
        a = to_subgraph(predecessors=plr_1.predecessors,
                        old_root=polygon,
                        new_root=plr_1.root,
                        sites=first_site_set,
                        graph=graph)
        b = to_subgraph(predecessors=pll_1.predecessors,
                        old_root=polygon,
                        new_root=pll_1.root,
                        sites=sites - first_site_set,
                        graph=graph)
        return b, a


def to_requirements_per_vertex(vertices: List[Point],
                               sites: FrozenSet[Requirement]
                               ) -> Iterator[FrozenSet[Requirement]]:
    """
    When rotating a line head over `domain_vertices`
    we need to get on each step a list of all previously encountered
    sites, so that later we can extract corresponding area requirement.
    """
    if not sites:
        return ValueError(f"No sites given")
    if vertices[0] not in {site.point for site in sites}:
        raise ValueError(f"The first vertex should be a site")
    sites_per_locations: Dict[Point, Requirement] = {site.point: site
                                                     for site in sites}
    first_site = sites_per_locations[vertices[0]]
    yield frozenset({first_site})
    accumulated_sites = frozenset()
    for vertex in vertices[:-1]:
        site = sites_per_locations.get(vertex)
        if site is not None:
            accumulated_sites |= {site}
        yield accumulated_sites


def normalize_areas(requirements: List[Real],
                    polygon_area: Fraction) -> List[Fraction]:
    *requirements, last_requirement = requirements
    requirements = [Fraction(requirement) * polygon_area
                    for requirement in requirements]
    last_requirement = polygon_area - sum(requirements)
    requirements.append(last_requirement)
    return requirements


def to_graph(polygon: Polygon,
             extra_points: List[Point],
             *,
             convex_divisor: ConvexDivisorType) -> nx.Graph:
    """
    Converts polygon to a region-adjacency graph by dividing it to 
    parts using `convex_divisor` function. Resulting parts become 
    nodes connected when they touch each other.
    :param polygon: input polygon that will be split
    :param extra_points: list of points which will be used in
    splitting the polygon to convex parts
    :param convex_divisor: function to split the polygon into convex
    parts
    :return: graph with parts of the polygon as nodes;
    edges will keep `side` attributes with the touching segments.
    """
    graph = nx.Graph()
    polygon_border = polygon.border
    holes = polygon.holes
    site_points = extra_points
    polygon_points = {*polygon_border.vertices,
                      *chain.from_iterable(hole.vertices
                                           for hole in holes)}
    extra_points = list(set(site_points) - polygon_points)
    parts = convex_divisor(polygon,
                           extra_points=extra_points,
                           context=get_context())
    if isinstance(parts, Triangulation):
        parts = parts.triangles()
    parts = list(map(Polygon, parts))
    if len(parts) == 1:
        graph.add_nodes_from(parts)
    else:
        if convex_divisor is Triangulation.constrained_delaunay:
            parts_per_sides = defaultdict(set)
            for part in parts:
                for side in edges(part.border):
                    parts_per_sides[side].add(part)
            for side, parts in parts_per_sides.items():
                if len(parts) == 2:
                    graph.add_edge(*parts, side=side)
        else:
            pairs: Iterator[Tuple[Polygon, Polygon]] = combinations(parts, 2)
            for part, other in pairs:
                intersection = part & other
                if isinstance(intersection, Segment):
                    graph.add_edge(part, other, side=intersection)
    return graph


def order_by_sites(vertices: List[Point],
                   site_location: Point) -> List[Point]:
    """
    Orders vertices of the convex polygon and the sites in a
    counterclockwise manner so that the last vertex would be a site.
    :param vertices: convex polygon's vertices and sites
    :param site_location: site that will be the last vertex
    :return: ordered union of polygon vertices and sites
    """
    ordered_vertices = order_convex_contour_points(vertices)
    site_index = next(index for index, point in enumerate(ordered_vertices)
                      if point == site_location)
    return rotate(ordered_vertices, site_index + 1)


def order_by_edge(vertices: List[Point],
                  edge: Segment) -> List[Point]:
    """
    Orders vertices of the convex polygon and the sites in a
    counterclockwise manner so that the last edge would be the edge to
    the next neighbor.
    :param vertices: vertices of the convex polygon and the sites
    located on the boundary of the polygon
    :param edge: edge to the next neighbor
    :return: ordered union of polygon vertices and sites
    """
    ordered_points = order_convex_contour_points(vertices)
    edge_start_index = ordered_points.index(edge.start)
    edge_end_index = ordered_points.index(edge.end)
    last_index = (min(edge_start_index, edge_end_index)
                  if abs(edge_end_index - edge_start_index) == 1
                  else max(edge_start_index, edge_end_index))
    return rotate(ordered_points, last_index + 1)


def to_subgraph(*,
                predecessors: PolygonsSet,
                old_root: Polygon,
                new_root: Polygon,
                sites: FrozenSet[Requirement],
                graph: Graph) -> Graph:
    graph = graph.subgraph([old_root, *predecessors])
    for edge in graph.edges:
        if old_root in edge:
            neighbor = edge[1] if edge[0] == old_root else edge[0]
            graph.edges[edge]['side'] = new_root & neighbor
    graph = nx.relabel_nodes(graph, {old_root: new_root})
    graph.nodes[new_root]['requirements'] = sites
    return graph
