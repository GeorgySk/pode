from typing import (Optional,
                    Set,
                    Tuple)

import networkx as nx
from hypothesis.strategies import (builds,
                                   integers,
                                   just,
                                   sets)

from tests.configs import MAX_ITERABLES_SIZE
from tests.strategies.common import fractions

RNG_SEED = 0

nodes_indices = integers(min_value=0,
                         max_value=MAX_ITERABLES_SIZE)
positive_nodes_counts = integers(min_value=1,
                                 max_value=MAX_ITERABLES_SIZE)
limited_integers_gt_one = integers(min_value=2,
                                   max_value=MAX_ITERABLES_SIZE)

single_node_graphs = builds(nx.path_graph, n=just(1))


def path_graph_and_node(n1: int,
                        n2: int) -> Tuple[nx.Graph, int]:
    node_index, graph_size = sorted([n1, n2])
    graph = nx.path_graph(graph_size + 1)
    node = node_index  # nodes are equal to their indices in path graphs
    return graph, node


def circle_graph_and_node(n1: int,
                          n2: int) -> Tuple[nx.Graph, int]:
    node_index, graph_size = sorted([n1, n2])
    graph = nx.circulant_graph(graph_size + 1, [1])
    node = node_index  # nodes are equal to their indices in path graphs
    return graph, node


path_graphs_and_random_nodes = builds(path_graph_and_node,
                                      nodes_indices,
                                      nodes_indices)
circle_graphs_and_random_nodes = builds(circle_graph_and_node,
                                        nodes_indices,
                                        nodes_indices)

empty_graphs = builds(nx.Graph)


def barabasi_albert_graph(m_and_n: Set[int],
                          *,
                          seed: Optional[int] = RNG_SEED) -> nx.Graph:
    m, n = sorted(m_and_n)
    return nx.barabasi_albert_graph(n, m, seed=seed)


def connected_watts_strogatz_graph(n_and_k: Set[int],
                                   p: float,
                                   *,
                                   seed: Optional[int] = RNG_SEED) -> nx.Graph:
    k, n = sorted(n_and_k)
    return nx.connected_watts_strogatz_graph(n, k, p, seed=seed)


def powerlaw_cluster_graph(n: int,
                           m: int,
                           p: float,
                           *,
                           seed: Optional[int] = RNG_SEED) -> nx.Graph:
    m, n = sorted([n, m])
    return nx.powerlaw_cluster_graph(n, m, p, seed=seed)


def regular_graph(parameters: Set[int],
                  *,
                  seed: Optional[int] = RNG_SEED) -> nx.Graph:
    d, n = sorted(parameters)
    return nx.random_regular_graph(d, n, seed=seed)


barabasi_albert_graphs = builds(barabasi_albert_graph,
                                m_and_n=sets(positive_nodes_counts,
                                             min_size=2,
                                             max_size=2))
connected_watts_strogatz_graphs = builds(connected_watts_strogatz_graph,
                                         n_and_k=sets(limited_integers_gt_one,
                                                      min_size=2,
                                                      max_size=2),
                                         p=fractions)
powerlaw_cluster_graphs = builds(powerlaw_cluster_graph,
                                 n=positive_nodes_counts,
                                 m=positive_nodes_counts,
                                 p=fractions)
complete_graphs = builds(nx.complete_graph, positive_nodes_counts)
path_graphs = builds(nx.path_graph, positive_nodes_counts)
lobsters = builds(nx.random_lobster,
                  n=positive_nodes_counts,
                  p1=fractions,
                  p2=fractions)
connected_graphs = (barabasi_albert_graphs
                    | complete_graphs
                    | connected_watts_strogatz_graphs
                    | path_graphs
                    | powerlaw_cluster_graphs
                    | lobsters)
connected_graphs_with_edges = connected_graphs.filter(nx.Graph.size)
