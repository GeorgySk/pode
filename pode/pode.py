from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from itertools import (chain,
                       combinations)
from math import atan2
from numbers import Real
from typing import (DefaultDict,
                    Dict,
                    FrozenSet,
                    Iterable,
                    Iterator,
                    List,
                    Optional,
                    Tuple,
                    Union,
                    cast)

import networkx as nx
from gon.angular import Orientation
from gon.compound import Shaped
from gon.degenerate import (EMPTY,
                            Empty)
from gon.discrete import Multipoint
from gon.linear import (Contour,
                        Multisegment,
                        Segment)
from gon.primitive import Point
from gon.shaped import Polygon
from sect.triangulation import constrained_delaunay_triangles

from pode.hints import ConvexDivisorType
from pode.utils import (centroid,
                        cut,
                        edges,
                        midpoint,
                        orient,
                        shrink_collinear_vertices,
                        splitter_point,
                        to_fractions,
                        union)


@dataclass(frozen=True)
class Site:
    """Point with an area requirement"""
    location: Point
    requirement: Real


class PolygonsSet(FrozenSet[Polygon]):
    @cached_property
    def area(self) -> Fraction:
        return sum(polygon.area for polygon in self)


@dataclass(frozen=True)
class RootedPolygons:
    root: Union[Polygon, Empty]
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
            if len(ordered_graph[node]) in {0, 1}:
                continue
            most_immediate_successor = cls.next_neighbor(ordered_graph,
                                                         node)
            bad_edges_nodes = (set(ordered_graph[node])
                               - {most_immediate_successor})
            for neighbor in bad_edges_nodes:
                ordered_graph.remove_edge(node, neighbor)
        for node in ordered_graph:
            ordered_graph.nodes[node].update({'sites': frozenset()})
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
        edges_in_interval = Multisegment(*map(Segment,
                                              vertices_in_interval[:-1],
                                              vertices_in_interval[1:]))
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
        # Copying attributes containing sites per node
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
        # Copying attributes containing sites per node
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
        # Copying attributes containing sites per node
        for node in new_graph:
            new_graph.nodes[node].update(self.nodes[node])
        return new_graph


def divide_by_sites(
        polygon: Polygon,
        sites: List[Site],
        *,
        convex_divisor: ConvexDivisorType = constrained_delaunay_triangles
        ) -> List[Tuple[Site, Shaped]]:
    """
    Divides given polygon for the given sites
    :param polygon: input polygon
    :param sites: input sites
    :param convex_divisor: function to split input polygon to convex
    parts
    :return: mapping of input sites to resulting parts of input polygon
    """
    if any(site.location not in polygon for site in sites):
        raise ValueError("Not all the sites lie in the polygon.")
    if sum(site.requirement for site in sites) != 1:
        raise ValueError("Area requirements should sum up to 1.")
    if len(set(site.location for site in sites)) != len(sites):
        raise ValueError("Sites cannot share their locations.")
    if len(sites) == 1:
        return [(sites[0], polygon)]
    if any(site.requirement < 0.01 for site in sites):
        raise ValueError("Sites' requirements are currently limited to be "
                         "greater than 0.01")
    polygon = to_fractions(polygon)
    sites = frozenset(normalize_sites(sites, polygon_area=polygon.area))
    division = []
    pseudosites_to_sites: Dict[Site, Site] = {}
    area_incomplete_polygons: DefaultDict[Site,
                                          List[Polygon]] = defaultdict(list)
    graph = to_graph(polygon,
                     sites_locations=[site.location for site in sites],
                     convex_divisor=convex_divisor)
    graph = Graph.from_undirected(graph)
    remaining_nodes = iter(graph.nodes)
    while graph:
        current_polygon: Polygon = orient(next(remaining_nodes))
        pred_polys = graph.pred_polys(current_polygon)
        current_sites = get_current_sites(current_polygon,
                                          pred_polys=pred_polys,
                                          sites=sites,
                                          graph=graph)
        if not current_sites:
            continue
        sites -= current_sites
        requirement = sum(site.requirement for site in current_sites)
        if len(current_sites) == 1 and pred_polys.area < requirement:
            neighbor = graph.next_neighbor(current_polygon)
            edge = graph[current_polygon][neighbor]['side']
            site = list(current_sites)[0]
            site = pseudosites_to_sites.get(site, site)
            pseudosite = Site(location=midpoint(edge.start, edge.end),
                              requirement=requirement - pred_polys.area)
            pred_poly = union(*pred_polys)
            area_incomplete_polygons[site].append(pred_poly)
            graph.remove_nodes_from(pred_polys)
            pseudosites_to_sites[pseudosite] = site
            graph.nodes[neighbor]['sites'] |= {pseudosite}
        elif len(current_sites) == 1 and pred_polys.area == requirement:
            site = list(current_sites)[0]
            graph.remove_nodes_from(pred_polys)
            if site not in pseudosites_to_sites:
                division.append((site, union(*pred_polys)))
            else:
                original_site = pseudosites_to_sites[site]
                area = union(*pred_polys,
                             *area_incomplete_polygons[original_site])
                division.append((original_site, area))
        else:
            neighbor = graph.next_neighbor(current_polygon)
            if neighbor is None:
                sites_locations = [site.location for site in current_sites]
                extra_points = graph.neighbor_edges_vertices(current_polygon)
                vertices = Multipoint(*{*current_polygon.border.vertices,
                                        *sites_locations,
                                        *extra_points})
                vertices = order_by_sites(vertices, sites_locations[0])
            else:
                edge = graph[current_polygon][neighbor]['side']
                extra_points = graph.neighbor_edges_vertices(current_polygon)
                vertices = Multipoint(*{*current_polygon.border.vertices,
                                        *extra_points,
                                        *(site.location
                                          for site in current_sites)})
                vertices = order_by_edge(vertices, edge)
            parts = nonconvex_divide(polygon=current_polygon,
                                     vertices=vertices,
                                     sites=current_sites,
                                     graph=graph)
            graph.remove_nodes_from(pred_polys)
            graph = (graph.prepend_two(parts, neighbor) if len(parts) == 2
                     else graph.prepend_three(parts, neighbor))
        remaining_nodes = iter(graph.nodes)
    return division


def divide_by_requirements(
        polygon: Polygon,
        requirements: List[Real],
        *,
        convex_divisor: ConvexDivisorType = constrained_delaunay_triangles
        ) -> List[Shaped]:
    """
    Divides given polygon for the given requirements
    :param polygon: input polygon
    :param requirements: input requirements
    :param convex_divisor: function to split input polygon to convex
    parts
    :return: mapping of sites to resulting parts of input polygon
    """
    if sum(requirement for requirement in requirements) != 1:
        raise ValueError("Area requirements should sum up to 1.")
    if len(requirements) == 1:
        return [polygon]
    if any(requirement < 0.01 for requirement in requirements):
        raise ValueError("Requirements are currently limited to be "
                         "greater than 0.01")
    polygon = to_fractions(polygon)
    requirements = normalize_requirements(requirements,
                                          polygon_area=polygon.area)
    division = []
    pseudosites_to_sites: Dict[Site, Site] = {}
    area_incomplete_polygons: DefaultDict[Site,
                                          List[Polygon]] = defaultdict(list)
    graph = to_graph(polygon,
                     sites_locations=[],
                     convex_divisor=convex_divisor)
    graph = Graph.from_undirected(graph)
    remaining_nodes = iter(graph.nodes)
    while graph:
        current_polygon: Polygon = orient(next(remaining_nodes))
        current_sites = graph.nodes[current_polygon]['sites']
        if not current_sites and requirements:
            *requirements, site_requirement = requirements
            current_sites = frozenset({
                Site(location=current_polygon.border.vertices[0],
                     requirement=site_requirement)})
        if not current_sites:
            continue
        pred_polys = graph.pred_polys(current_polygon)
        requirement = sum(site.requirement for site in current_sites)
        if len(current_sites) == 1 and pred_polys.area < requirement:
            neighbor = graph.next_neighbor(current_polygon)
            edge = graph[current_polygon][neighbor]['side']
            site = list(current_sites)[0]
            site = pseudosites_to_sites.get(site, site)
            pseudosite = Site(location=midpoint(edge.start, edge.end),
                              requirement=requirement - pred_polys.area)
            pred_poly = union(*pred_polys)
            area_incomplete_polygons[site].append(pred_poly)
            graph.remove_nodes_from(pred_polys)
            pseudosites_to_sites[pseudosite] = site
            graph.nodes[neighbor]['sites'] |= {pseudosite}
        elif len(current_sites) == 1 and pred_polys.area == requirement:
            site = list(current_sites)[0]
            graph.remove_nodes_from(pred_polys)
            if site not in pseudosites_to_sites:
                division.append(union(*pred_polys))
            else:
                original_site = pseudosites_to_sites[site]
                division.append(
                    union(*pred_polys,
                          *area_incomplete_polygons[original_site]))
        else:
            neighbor = graph.next_neighbor(current_polygon)
            if neighbor is not None:
                edge = graph[current_polygon][neighbor]['side']
                extra_points = graph.neighbor_edges_vertices(current_polygon)
                vertices = Multipoint(*{*current_polygon.border.vertices,
                                        *extra_points,
                                        *(site.location
                                          for site in current_sites)})
                vertices = order_by_edge(vertices, edge)
            else:
                sites_locations = [site.location for site in current_sites]
                extra_points = graph.neighbor_edges_vertices(current_polygon)
                vertices = Multipoint(*{*current_polygon.border.vertices,
                                        *sites_locations,
                                        *extra_points})
                vertices = order_by_sites(vertices, sites_locations[0])
            parts = nonconvex_divide(polygon=current_polygon,
                                     vertices=vertices,
                                     sites=current_sites,
                                     graph=graph)
            graph.remove_nodes_from(pred_polys)
            graph = (graph.prepend_two(parts, neighbor) if len(parts) == 2
                     else graph.prepend_three(parts, neighbor))
        remaining_nodes = iter(graph.nodes)
    return division


def normalize_sites(sites: List[Site],
                    *,
                    polygon_area: Fraction) -> List[Site]:
    locations = (to_fractions(site.location) for site in sites)
    requirements = [site.requirement for site in sites]
    requirements = normalize_requirements(requirements,
                                          polygon_area=polygon_area)
    sites = [Site(location=location,
                  requirement=requirement)
             for location, requirement in zip(locations, requirements)]
    return sites


def normalize_requirements(requirements: List[Real],
                           *,
                           polygon_area: Fraction) -> List[Fraction]:
    *requirements, last_requirement = requirements
    requirements = [Fraction(requirement) * polygon_area
                    for requirement in requirements]
    last_requirement = polygon_area - sum(requirements)
    requirements.append(last_requirement)
    return requirements


def to_graph(polygon: Polygon,
             sites_locations: List[Point],
             *,
             convex_divisor: ConvexDivisorType) -> nx.Graph:
    """
    Converts polygon to a graph by dividing it to parts using
    `convex_divisor` function. Resulting parts become nodes connected
    when they touch each other.
    :param polygon: input polygon that will be split
    :param sites_locations: list of sites the points of which will be
    used in splitting the polygon to convex parts
    :param convex_divisor: function to split the polygon into convex
    parts
    :return: graph with parts of the polygon as nodes;
    edges will keep `side` attributes with the touching segments.
    """
    graph = nx.Graph()
    polygon_border = polygon.border.raw()
    holes = list(map(Contour.raw, polygon.holes))
    site_points = list(map(Point.raw, sites_locations))
    polygon_points = {*polygon_border, *chain.from_iterable(holes)}
    extra_points = list(set(site_points) - polygon_points)
    parts = convex_divisor(polygon_border,
                           holes,
                           extra_points=extra_points,
                           extra_constraints=())
    parts = [Polygon.from_raw((list(part), [])) for part in parts]
    if len(parts) == 1:
        graph.add_nodes_from(parts)
    else:
        if convex_divisor is constrained_delaunay_triangles:
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


def order_by_sites(points: Multipoint,
                   site_location: Point) -> List[Point]:
    """
    Orders vertices of the convex polygon and the sites in a
    counterclockwise manner so that the last vertex would be a site.
    :param points: convex polygon's vertices and sites
    :param site_location: site that will be the last vertex
    :return: ordered union of polygon vertices and sites
    """
    centroid_ = centroid(points)

    def angle(point: Point) -> float:
        return atan2(point.y - centroid_.y, point.x - centroid_.x)

    ordered_points = sorted(points.points, key=angle)
    site_index = next(index for index in range(len(ordered_points))
                      if ordered_points[index] == site_location)
    return ordered_points[site_index + 1:] + ordered_points[:site_index + 1]


def order_by_edge(vertices: Multipoint,
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
    points = cast(Multipoint, vertices | Multipoint(edge.start, edge.end))
    centroid_ = centroid(points)

    def angle(point: Point) -> float:
        return atan2(point.y - centroid_.y, point.x - centroid_.x)

    ordered_points = sorted(points.points, key=angle)
    edge_start_index = ordered_points.index(edge.start)
    edge_end_index = ordered_points.index(edge.end)
    last_index = (min(edge_start_index, edge_end_index)
                  if abs(edge_end_index - edge_start_index) == 1
                  else max(edge_start_index, edge_end_index))
    return ordered_points[last_index + 1:] + ordered_points[:last_index + 1]


def nonconvex_divide(*,
                     polygon: Polygon,
                     vertices: List[Point],
                     sites: FrozenSet[Site],
                     graph: Graph
                     ) -> Union[Tuple[nx.DiGraph, nx.DiGraph],
                                Tuple[nx.DiGraph, nx.DiGraph, nx.DiGraph]]:
    """Splits a PredPoly to 2 or 3 parts for further division"""
    if polygon.border.orientation is not Orientation.COUNTERCLOCKWISE:
        raise ValueError("Polygon division is implemented only for polygons "
                         "oriented counter-clockwise")
    sites_locations = set(site.location for site in sites)
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
        requirements = [list(sites)[0].requirement] * len(vertices)
    else:
        sites_per_vertex = list(to_sites_per_vertex(heads, sites))
        requirements = [sum(site.requirement for site in sites_)
                        for sites_ in sites_per_vertex]
        # requirements = list(requirements_per_vertex(heads, sites))
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
            ps = midpoint(low_area_point, high_area_point)
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
        t = midpoint(vertices[-1], vertices[0])
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


def to_sites_per_vertex(vertices: List[Point],
                        sites: FrozenSet[Site]) -> Iterator[FrozenSet[Site]]:
    """
    When rotating a line head over `domain_vertices`
    we need to get on each step a list of all previously encountered
    sites, so that later we can extract corresponding area requirement.
    """
    if not sites:
        return ValueError(f"No sites given")
    if vertices[0] not in {site.location for site in sites}:
        raise ValueError(f"The first vertex should be a site")
    sites_per_locations: Dict[Point, Site] = {site.location: site
                                              for site in sites}
    first_site = sites_per_locations[vertices[0]]
    yield frozenset({first_site})
    accumulated_sites = frozenset()
    for vertex in vertices[:-1]:
        site = sites_per_locations.get(vertex)
        if site is not None:
            accumulated_sites |= {site}
        yield accumulated_sites


def get_current_sites(polygon: Polygon,
                      pred_polys: PolygonsSet,
                      sites: FrozenSet[Site],
                      graph: Graph) -> FrozenSet[Site]:
    preassigned_sites = frozenset({site for node in graph
                                   for site in graph.nodes[node]['sites']})
    if not sites and not preassigned_sites:
        raise ValueError("Got no sites to assign")
    current_preassigned_sites: FrozenSet[Site] = graph.nodes[polygon]['sites']
    all_nodes = frozenset(graph.nodes)
    remaining_nodes = all_nodes - pred_polys
    if not remaining_nodes:
        return current_preassigned_sites | sites
    current_sites = frozenset({site for site in sites
                               if site.location in polygon})
    shared_sites = {site for site in current_sites
                    if any(site.location in node for node in remaining_nodes)}
    polygon_sites_only = current_sites - shared_sites
    if polygon_sites_only or current_preassigned_sites:
        return polygon_sites_only | current_preassigned_sites
    if len(current_sites) in {0, 1}:
        return current_sites
    sites_with_given_requirement = frozenset({
        site for site in sites if pred_polys.area == site.requirement})
    if len(sites_with_given_requirement) == 1:
        return sites_with_given_requirement
    return frozenset({list(current_sites)[0]})


def to_subgraph(*,
                predecessors: PolygonsSet,
                old_root: Polygon,
                new_root: Polygon,
                sites: FrozenSet[Site],
                graph: Graph) -> Graph:
    graph = graph.subgraph([old_root, *predecessors])
    for edge in graph.edges:
        if old_root in edge:
            neighbor = edge[1] if edge[0] == old_root else edge[0]
            graph.edges[edge]['side'] = new_root & neighbor
    graph = nx.relabel_nodes(graph, {old_root: new_root})
    graph.nodes[new_root]['sites'] = sites
    return graph
