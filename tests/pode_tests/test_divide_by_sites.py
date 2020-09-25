from typing import (List,
                    Tuple)

from gon.shaped import Polygon
from hypothesis import given

from pode.hints import ConvexDivisorType
from pode.pode import (Site,
                       divide_by_sites)
from tests.strategies.geometry.base import convex_divisors
from tests.strategies.geometry.composite import polygons_and_sites
from utils import to_fractions


@given(polygon_and_sites=polygons_and_sites,
       convex_divisor=convex_divisors)
def test_partitions(polygon_and_sites: Tuple[Polygon, List[Site]],
                    convex_divisor: ConvexDivisorType) -> None:
    polygon, sites = polygon_and_sites
    division = divide_by_sites(*polygon_and_sites,
                               convex_divisor=convex_divisor)
    assert len(division) == len(sites)
    if len(sites) > 1:
        assert to_fractions(polygon).area == sum(part.area
                                                 for _, part in division)
    else:
        assert polygon.area == sum(part.area for _, part in division)
