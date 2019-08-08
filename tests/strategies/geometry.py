from functools import partial
from itertools import repeat
from typing import (List,
                    Sequence,
                    Tuple,
                    TypeVar)

from hypothesis.strategies import (SearchStrategy,
                                   builds,
                                   floats,
                                   integers,
                                   lists,
                                   sampled_from,
                                   sets,
                                   tuples)
from shapely.affinity import rotate
from shapely.geometry import (LineString,
                              Point,
                              Polygon,
                              box)

from tests.configs import (ABS_TOL,
                           MAX_ITERABLES_SIZE)
from tests.utils import (form_object_with_area,
                         points_are_sparse,
                         polygons_are_sparse)

T = TypeVar('T')

iterable_sizes = integers(min_value=0,
                          max_value=MAX_ITERABLES_SIZE)
to_finite_floats = partial(floats,
                           allow_infinity=False,
                           allow_nan=False,
                           width=16)
finite_floats = to_finite_floats()
nonnegative_floats = to_finite_floats(min_value=0)
positive_floats = to_finite_floats(min_value=ABS_TOL,
                                   exclude_min=True)
coordinates = slope_intercepts = tuples(finite_floats, finite_floats)
angles = to_finite_floats(min_value=-360,
                          max_value=360)
points = builds(Point, finite_floats, finite_floats)
segments = builds(LineString, lists(coordinates,
                                    min_size=2,
                                    max_size=2,
                                    unique=True))
# too small segments conflict with precision errors
segments = segments.filter(ABS_TOL.__lt__)

fractions = to_finite_floats(min_value=0,
                             max_value=1)


def points_by_distances(line: LineString,
                        distances: Sequence[float]) -> List[Point]:
    return [line.interpolate(distance, normalized=True)
            for distance in distances]


def to_aligned_points(count: int) -> SearchStrategy[List[Point]]:
    sorted_fractions = sets(fractions,
                            min_size=count,
                            max_size=count).map(sorted)
    return builds(points_by_distances,
                  segments,
                  sorted_fractions)


def to_containing_lines(points: List[Point]) -> Tuple[LineString, LineString]:
    long_line = LineString([points[0], points[3]])
    short_line = LineString([points[1], points[2]])
    return long_line, short_line


lines_one_in_another = builds(to_containing_lines, to_aligned_points(4))
lines_one_in_another = lines_one_in_another.filter(points_are_sparse)


def to_overlapping_lines(points: List[Point]) -> Tuple[LineString, LineString]:
    line = LineString([points[0], points[2]])
    other_line = LineString([points[1], points[3]])
    return line, other_line


overlapping_lines = builds(to_overlapping_lines, to_aligned_points(4))
overlapping_lines = overlapping_lines.filter(points_are_sparse)


def to_touching_lines(points: List[Point]) -> Tuple[LineString, LineString]:
    line = LineString([points[0], points[1]])
    other_line = LineString([points[1], points[2]])
    return line, other_line


touching_lines = builds(to_touching_lines, to_aligned_points(3))
touching_lines = touching_lines.filter(points_are_sparse)


def to_disjoint_lines(points: List[Point]) -> Tuple[LineString, LineString]:
    line = LineString(points[:2])
    other_line = LineString(points[2:])
    return line, other_line


disjoint_lines = builds(to_disjoint_lines, to_aligned_points(4))
disjoint_lines = disjoint_lines.filter(points_are_sparse)

offset_sides = sampled_from(['left', 'right'])

three_unique_coordinates_lists = lists(coordinates,
                                       min_size=3,
                                       max_size=3,
                                       unique=True)
triangles = (three_unique_coordinates_lists
             .filter(form_object_with_area)
             .map(Polygon))

circles = builds(Point.buffer, points, positive_floats)

straight_rectangles = (builds(box, *repeat(finite_floats, 4))
                       .filter(form_object_with_area))
rectangles = builds(rotate, straight_rectangles, angles)

polygons = triangles | circles | rectangles
disjoint_polygons_pairs = (tuples(polygons, polygons)
                           .filter(polygons_are_sparse))
disjoint_polygons_lists = (lists(polygons, max_size=MAX_ITERABLES_SIZE)
                           .filter(polygons_are_sparse))
same_polygons_iterators = builds(repeat, polygons, iterable_sizes)
