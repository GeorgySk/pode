from typing import (Callable,
                    TypeVar)

import networkx as nx
from hypothesis.strategies import (SearchStrategy,
                                   builds,
                                   composite)
from sect.triangulation import constrained_delaunay_triangles

from pode.pode import (Graph,
                       to_graph)
from tests.strategies.geometry.composite import polygons_and_sites

T = TypeVar('T')


@composite
def _unordered_graphs(draw: Callable[[SearchStrategy[T]], T]) -> nx.Graph:
    polygon, sites = draw(polygons_and_sites)
    return to_graph(polygon,
                    [site.location for site in sites],
                    convex_divisor=constrained_delaunay_triangles)


unordered_graphs = _unordered_graphs()
graphs = builds(Graph.from_undirected, unordered_graphs)
