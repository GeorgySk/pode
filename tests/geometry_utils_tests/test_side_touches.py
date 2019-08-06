from typing import Tuple

from hypothesis import (given,
                        note)
from shapely import wkt
from shapely.geometry import Polygon

from pode.geometry_utils import side_touches
from tests.strategies import (disjoint_polygons_pairs,
                              polygons)


@given(polygons)
def test_equal(polygon: Polygon) -> None:
    note(f'Polygon: {wkt.dumps(polygon, trim=False)}')
    assert side_touches(polygon, polygon)


@given(disjoint_polygons_pairs)
def test_disjoint(polygon_pair: Tuple[Polygon, Polygon]) -> None:
    note(f'Polygon 1: {wkt.dumps(polygon_pair[0], trim=False)}\n'
         f'Polygon 2: {wkt.dumps(polygon_pair[1], trim=False)}')
    assert not side_touches(*polygon_pair)
