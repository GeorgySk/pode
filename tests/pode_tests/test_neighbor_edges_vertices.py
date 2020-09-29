from hypothesis import given

from pode.pode import Graph
from tests.strategies.geometry.base import nonnegative_integers
from tests.strategies.graphs import graphs


@given(graph=graphs,
       index=nonnegative_integers)
def test_points(graph: Graph,
                index: int) -> None:
    nodes = list(graph)
    node = nodes[index % len(nodes)]
    points = graph.neighbor_edges_vertices(node)
    assert all(point in node for point in points)
