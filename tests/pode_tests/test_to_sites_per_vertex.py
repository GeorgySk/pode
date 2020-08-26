from typing import (List,
                    Set,
                    Tuple)

from gon.primitive import Point
from hypothesis import given

from pode.pode import (Site,
                       to_sites_per_vertex)
from tests.strategies.geometry.composite import vertices_and_sites


@given(vertices_and_sites)
def test_max_sum(vertices_and_sites_: Tuple[List[Point], Set[Site]]) -> None:
    vertices, sites = vertices_and_sites_
    accumulated_sites_per_vertex = list(to_sites_per_vertex(vertices,
                                                            frozenset(sites)))
    full_requirement = sum(site.requirement for site in sites)
    assert all(sum(site.requirement
                   for site in current_sites) <= full_requirement
               for current_sites in accumulated_sites_per_vertex)
