from functools import partial
from typing import (List,
                    Iterable)

from hypothesis import (given,
                        note)
from shapely import wkt
from shapely.geometry import Polygon

from pode.geometry_utils import neighbors
from tests.strategies import (disjoint_polygons_lists,
                              same_polygons_iterators)

no_trim_wkt = partial(wkt.dumps, trim=False)


@given(disjoint_polygons_lists)
def test_disjoint(polygons: List[Polygon]) -> None:
    note(f'Polygons: {list(map(no_trim_wkt, polygons))}')
    assert not list(neighbors(polygons))


@given(same_polygons_iterators)
def test_coinciding(polygons: Iterable[Polygon]) -> None:
    polygons = list(polygons)
    note(f'Polygons: {list(map(no_trim_wkt, polygons))}')
    polygons_count = len(polygons)
    neighbors_count = len(list(neighbors(polygons)))
    assert neighbors_count == (polygons_count - 1) * polygons_count // 2
