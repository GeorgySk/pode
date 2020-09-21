from hypothesis import given
from hypothesis.strategies import integers

from pode.pode import Graph
from tests.strategies.graphs import graphs


@given(graph=graphs,
       index=integers(min_value=0))
def test_points(graph: Graph,
                index: int) -> None:
    nodes = list(graph)
    node = nodes[index % len(nodes)]
    points = graph.neighbor_edges_vertices(node)
    assert all(point in node for point in points)
