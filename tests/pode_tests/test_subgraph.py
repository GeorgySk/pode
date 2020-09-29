from hypothesis import given, strategies as st

from pode.pode import Graph
from tests.strategies.graphs import graphs


@given(graph=graphs,
       index=st.integers(min_value=0),
       other_index=st.integers(min_value=0))
def test_subset(graph: Graph,
                index: int,
                other_index: int) -> None:
    nodes = list(graph)
    index = index % len(nodes)
    other_index = other_index % len(nodes)
    subset = (nodes[index:other_index] if index < other_index
              else nodes[other_index:index])
    subgraph = graph.subgraph(subset)
    edges = set(graph.edges)
    assert all(node in graph for node in subgraph)
    assert all(edge in edges for edge in subgraph.edges)
