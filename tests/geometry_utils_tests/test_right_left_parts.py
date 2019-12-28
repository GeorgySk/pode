from math import isclose

from hypothesis import (given,
                        note)
from shapely.geometry import (LineString,
                              Polygon)

from pode.geometry_utils import right_left_parts
from tests.strategies import (convex_polygons,
                              segments)


@given(polygon=convex_polygons,
       segment=segments)
def test_area(polygon: Polygon,
              segment: LineString) -> None:
    note(f"Polygon: {polygon.wkt}\n"
         f"LineString: {segment.wkt}")
    part, other_part = right_left_parts(polygon, segment)
    assert isclose(polygon.area,
                   part.area + other_part.area,
                   rel_tol=1e-08)  # smaller tolerance can result in errors
