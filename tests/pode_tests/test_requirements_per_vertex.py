from typing import (List,
                    Set,
                    Tuple)

from gon.primitive import Point
from hypothesis import given

from pode.pode import (Site,
                       requirements_per_vertex)
from tests.strategies.geometry.composite import vertices_and_sites


@given(vertices_and_sites)
def test_max_sum(vertices_and_sites_: Tuple[List[Point], Set[Site]]) -> None:
    vertices, sites = vertices_and_sites_
    requirements = list(requirements_per_vertex(vertices, sites))
    full_requirement = sum(site.requirement for site in sites)
    assert all(requirement <= full_requirement for requirement in requirements)
