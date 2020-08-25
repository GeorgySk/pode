from fractions import Fraction
from numbers import Real
from typing import List

from gon.shaped import Polygon
from hypothesis import given
from hypothesis.strategies import integers

from pode.hints import ConvexDivisor
from pode.pode import divide_by_requirements
from tests.pode_tests.config import (MAX_SITES_COUNT,
                                     MIN_REQUIREMENT,
                                     MIN_SITES_COUNT)
from tests.strategies.geometry.base import (convex_divisors,
                                            polygons)
from tests.strategies.sites import requirements


@given(polygon=polygons,
       requirements_=integers(min_value=MIN_SITES_COUNT,
                              max_value=MAX_SITES_COUNT).flatmap(
           lambda n: requirements(sum_=1,
                                  min_value=MIN_REQUIREMENT,
                                  size=n)),
       convex_divisor=convex_divisors)
def test_partitions(polygon: Polygon,
                    requirements_: List[Real],
                    convex_divisor: ConvexDivisor) -> None:
    division = divide_by_requirements(polygon, requirements_,
                                      convex_divisor=convex_divisor)
    assert len(division) == len(requirements_)
    assert Fraction(polygon.area) == sum(part.area for part in division)
