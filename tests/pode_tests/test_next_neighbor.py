from hypothesis import (assume,
                        given,
                        note,
                        strategies as st)

from pode.pode import Graph
from tests.strategies.graphs import graphs


@given(graphs)
def test_last_node(graph: Graph) -> None:
    note(f'Graph nodes: {list(graph)}')
    note(f'Graph edges: {graph.edges}')
    last_node = list(graph)[-1]
    note(f'Last node: {last_node}')
    assert graph.next_neighbor(last_node) is None


@given(graphs)
def test_first_node(graph: Graph) -> None:
    assume(len(graph) > 1)
    note(f'Graph nodes: {list(graph)}')
    note(f'Graph edges: {graph.edges}')
    first_node = next(iter(graph))
    note(f'First node: {first_node}')
    assert graph.next_neighbor(first_node) in graph


@given(graph=graphs,
       index=st.integers(min_value=0))
def test_order(graph: Graph,
               index: int) -> None:
    assume(len(graph) > 1)
    note(f'Graph nodes: {list(graph)}')
    note(f'Graph edges: {graph.edges}')
    nodes = list(graph)
    node_index = index % (len(nodes) - 1)
    node = nodes[node_index]
    note(f'Chosen node: {node}')
    assert nodes.index(graph.next_neighbor(node)) > node_index
