from typing import (List,
                    Tuple)

import networkx as nx
from gon.primitive import Point
from gon.shaped import Polygon
from hypothesis import strategies as st

from pode.hints import ConvexDivisorType
from pode.pode import (Graph,
                       to_graph)
from tests.strategies.geometry.base import convex_divisors
from tests.strategies.geometry.composite import fraction_polygons_and_points


def _to_graph(polygon_and_points: Tuple[Polygon, List[Point]],
              convex_divisor: ConvexDivisorType) -> nx.Graph:
    polygon, points = polygon_and_points
    return to_graph(polygon, points, convex_divisor=convex_divisor)


unordered_graphs = st.builds(_to_graph,
                             fraction_polygons_and_points,
                             convex_divisors)
graphs = st.builds(Graph.from_undirected, unordered_graphs)
