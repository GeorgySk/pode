from fractions import Fraction
from typing import (List,
                    Tuple)

from gon.shaped import Polygon
from hypothesis import given

from pode.pode import (Site,
                       divide)
from tests.strategies.geometry.composite import polygons_and_sites


@given(polygons_and_sites)
def test_area(polygon_and_sites: Tuple[Polygon, List[Site]]) -> None:
    polygon, sites = polygon_and_sites
    division = divide(*polygon_and_sites)
    assert Fraction(polygon.area) == sum(part.area
                                         for part in division.values())
