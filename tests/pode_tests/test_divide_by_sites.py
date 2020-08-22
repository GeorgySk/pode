from fractions import Fraction
from typing import (List,
                    Tuple)

from gon.shaped import Polygon
from hypothesis import (Verbosity,
                        given,
                        settings)

from pode.pode import (Site,
                       divide_by_sites)
from tests.strategies.geometry.composite import polygons_and_sites


@given(polygons_and_sites)
@settings(verbosity=Verbosity.verbose)
def test_area(polygon_and_sites: Tuple[Polygon, List[Site]]) -> None:
    polygon, sites = polygon_and_sites
    division = divide_by_sites(*polygon_and_sites)
    assert Fraction(polygon.area) == sum(part.area for _, part in division)
