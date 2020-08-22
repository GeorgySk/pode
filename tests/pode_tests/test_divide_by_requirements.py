from fractions import Fraction
from numbers import Real
from typing import List

from gon.shaped import Polygon
from hypothesis import (Verbosity,
                        given,
                        settings)
from hypothesis.strategies import integers

from pode.pode import divide_by_requirements
from tests.pode_tests.config import (MAX_SITES_COUNT,
                                     MIN_REQUIREMENT,
                                     MIN_SITES_COUNT)
from tests.strategies.geometry.base import polygons
from tests.strategies.sites import requirements


@given(polygon=polygons,
       requirements_=integers(min_value=MIN_SITES_COUNT,
                              max_value=MAX_SITES_COUNT).flatmap(
           lambda n: requirements(sum_=1,
                                  min_value=MIN_REQUIREMENT,
                                  size=n)))
@settings(verbosity=Verbosity.verbose)
def test_area(polygon: Polygon,
              requirements_: List[Real]) -> None:
    division = divide_by_requirements(polygon, requirements_)
    assert Fraction(polygon.area) == sum(part.area for _, part in division)
