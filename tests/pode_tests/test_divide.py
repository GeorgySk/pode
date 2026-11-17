from gon.base import (Contour,
                      Point,
                      Polygon,
                      Triangulation)
from hypothesis import (example,
                        given)

from pode.hints import ConvexDivisorType
from pode.pode import (Requirement,
                       divide)
from pode.utils import to_fractions
from tests.strategies.geometry.base import convex_divisors
from tests.strategies.geometry.composite import polygons_and_requirements


@given(polygon_and_requirements=polygons_and_requirements,
       convex_divisor=convex_divisors)
# Requirements with equal areas: the free one got anchored at the site of
# the other and was mistaken for it, so their parts overlapped
@example(polygon_and_requirements=(
             Polygon(Contour([Point(-265828656001435, -999999999999999),
                              Point(-83, -384962981531728),
                              Point(17070, -21154),
                              Point(17070, 120),
                              Point(1, 1282),
                              Point(-83, 281475010265157)])),
             [Requirement(area=1e-15, point=Point(0, 0)),
              Requirement(area=1e-15, point=None),
              Requirement(area=0.999999999999998, point=None)]),
         convex_divisor=Triangulation.constrained_delaunay)
def test_partitions(polygon_and_requirements: tuple[Polygon,
                                                    list[Requirement]],
                    convex_divisor: ConvexDivisorType) -> None:
    polygon, requirements = polygon_and_requirements
    division = divide(polygon, requirements, convex_divisor)
    assert len(division) == len(requirements)
    if len(requirements) > 1:
        assert to_fractions(polygon).area == sum(part.area
                                                 for part in division)
    else:
        assert polygon.area == sum(part.area for part in division)
