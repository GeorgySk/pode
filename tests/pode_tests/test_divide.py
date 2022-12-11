from typing import (List,
                    Tuple)

from gon.base import Polygon
from hypothesis import given

from pode.hints import ConvexDivisorType
from pode.pode import (Requirement,
                       divide)
from pode.utils import to_fractions
from tests.strategies.geometry.base import convex_divisors
from tests.strategies.geometry.composite import polygons_and_requirements


@given(polygon_and_requirements=polygons_and_requirements,
       convex_divisor=convex_divisors)
def test_partitions(polygon_and_requirements: Tuple[Polygon,
                                                    List[Requirement]],
                    convex_divisor: ConvexDivisorType) -> None:
    polygon, requirements = polygon_and_requirements
    division = divide(polygon, requirements, convex_divisor)
    assert len(division) == len(requirements)
    if len(requirements) > 1:
        assert to_fractions(polygon).area == sum(part.area
                                                 for part in division)
    else:
        assert polygon.area == sum(part.area for part in division)
