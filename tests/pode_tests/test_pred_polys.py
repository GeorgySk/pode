from hypothesis import (given,
                        note)

from pode.pode import Graph
from tests.strategies.geometry.base import nonnegative_integers
from tests.strategies.graphs import graphs


@given(graphs)
def test_first_node(graph: Graph) -> None:
    note(f'Graph nodes: {list(graph)}')
    note(f'Graph edges: {graph.edges}')
    first_node = next(iter(graph))
    note(f'First node: {first_node}')
    pred_poly = graph.pred_polys(first_node)
    assert len(pred_poly) == 1
    assert list(pred_poly)[0] == first_node


@given(graph=graphs,
       index=nonnegative_integers)
def test_node_presence(graph: Graph,
                       index: int) -> None:
    nodes = list(graph)
    note(f'Graph nodes: {list(graph)}')
    note(f'Graph edges: {graph.edges}')
    node = nodes[index % len(graph)]
    note(f'Mode: {node}')
    pred_poly = graph.pred_polys(node)
    assert node in pred_poly


@given(graph=graphs,
       index=nonnegative_integers)
def test_predecessors(graph: Graph,
                      index: int) -> None:
    nodes = list(graph)
    note(f'Graph nodes: {list(graph)}')
    note(f'Graph edges: {graph.edges}')
    node_index = index % len(graph)
    node = nodes[node_index]
    note(f'Mode: {node}')
    pred_poly = graph.pred_polys(node)
    assert all(nodes.index(polygon) <= node_index for polygon in pred_poly)
