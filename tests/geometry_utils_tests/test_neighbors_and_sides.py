from typing import (List,
                    Iterable)

from hypothesis import (given,
                        note)
from shapely.geometry import Polygon

from pode.geometry_utils import neighbors_and_sides
from tests.strategies import (disjoint_polygons_lists,
                              same_polygons_iterators)
from tests.utils import no_trim_wkt


@given(disjoint_polygons_lists)
def test_disjoint(polygons: List[Polygon]) -> None:
    note(f'Polygons: {list(map(no_trim_wkt, polygons))}')
    assert not list(neighbors_and_sides(polygons))


@given(same_polygons_iterators)
def test_coinciding(polygons: Iterable[Polygon]) -> None:
    polygons = list(polygons)
    note(f'Polygons: {list(map(no_trim_wkt, polygons))}')
    polygons_count = len(polygons)
    neighbors_count = len(list(neighbors_and_sides(polygons)))
    assert neighbors_count == (polygons_count - 1) * polygons_count // 2
