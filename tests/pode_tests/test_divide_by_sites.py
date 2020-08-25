from fractions import Fraction
from typing import (List,
                    Tuple)

from gon.shaped import Polygon
from hypothesis import given

from pode.hints import ConvexDivisor
from pode.pode import (Site,
                       divide_by_sites)
from tests.strategies.geometry.base import convex_divisors
from tests.strategies.geometry.composite import polygons_and_sites


@given(polygon_and_sites=polygons_and_sites,
       convex_divisor=convex_divisors)
def test_partitions(polygon_and_sites: Tuple[Polygon, List[Site]],
                    convex_divisor: ConvexDivisor) -> None:
    polygon, sites = polygon_and_sites
    division = divide_by_sites(*polygon_and_sites,
                               convex_divisor=convex_divisor)
    assert len(division) == len(sites)
    assert Fraction(polygon.area) == sum(part.area for _, part in division)
