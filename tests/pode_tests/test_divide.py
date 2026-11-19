from itertools import combinations

from gon.base import (Contour,
                      Point,
                      Polygon,
                      Relation,
                      Triangulation)
from hypothesis import (example,
                        given)

from pode.hints import ConvexDivisorType
from pode.pode import (Requirement,
                       divide,
                       normalize_areas)
from pode.utils import to_fractions
from tests.strategies.geometry.base import convex_divisors
from tests.strategies.geometry.composite import polygons_and_requirements


@given(polygon_and_requirements=polygons_and_requirements,
       convex_divisor=convex_divisors)
def test_parts_count(polygon_and_requirements: tuple[Polygon,
                                                     list[Requirement]],
                     convex_divisor: ConvexDivisorType) -> None:
    polygon, requirements = polygon_and_requirements
    division = divide(polygon, requirements, convex_divisor)
    assert len(division) == len(requirements)


@given(polygon_and_requirements=polygons_and_requirements,
       convex_divisor=convex_divisors)
def test_parts_types(polygon_and_requirements: tuple[Polygon,
                                                     list[Requirement]],
                     convex_divisor: ConvexDivisorType) -> None:
    polygon, requirements = polygon_and_requirements
    division = divide(polygon, requirements, convex_divisor)
    assert all(isinstance(part, Polygon) for part in division)


@given(polygon_and_requirements=polygons_and_requirements,
       convex_divisor=convex_divisors)
def test_areas(polygon_and_requirements: tuple[Polygon, list[Requirement]],
               convex_divisor: ConvexDivisorType) -> None:
    polygon, requirements = polygon_and_requirements
    division = divide(polygon, requirements, convex_divisor)
    expected_areas = normalize_areas(
        [requirement.area for requirement in requirements],
        to_fractions(polygon).area)
    assert [to_fractions(part).area for part in division] == expected_areas


@given(polygon_and_requirements=polygons_and_requirements,
       convex_divisor=convex_divisors)
def test_sites(polygon_and_requirements: tuple[Polygon, list[Requirement]],
               convex_divisor: ConvexDivisorType) -> None:
    polygon, requirements = polygon_and_requirements
    division = divide(polygon, requirements, convex_divisor)
    assert all(to_fractions(requirement.point) in to_fractions(part)
               for requirement, part in zip(requirements, division,
                                            strict=True)
               if requirement.point is not None)


@given(polygon_and_requirements=polygons_and_requirements,
       convex_divisor=convex_divisors)
def test_containment(polygon_and_requirements: tuple[Polygon,
                                                     list[Requirement]],
                     convex_divisor: ConvexDivisorType) -> None:
    polygon, requirements = polygon_and_requirements
    division = divide(polygon, requirements, convex_divisor)
    polygon = to_fractions(polygon)
    assert all(polygon.relate(to_fractions(part)) in (Relation.EQUAL,
                                                      Relation.ENCLOSED,
                                                      Relation.WITHIN)
               for part in division)


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
def test_disjointness(polygon_and_requirements: tuple[Polygon,
                                                      list[Requirement]],
                      convex_divisor: ConvexDivisorType) -> None:
    polygon, requirements = polygon_and_requirements
    division = divide(polygon, requirements, convex_divisor)
    assert all(part.relate(other) in (Relation.DISJOINT, Relation.TOUCH)
               for part, other in combinations(division, 2))
