from functools import partial
from typing import (List,
                    Sequence,
                    Tuple,
                    TypeVar)

from hypothesis.strategies import (SearchStrategy,
                                   builds,
                                   floats,
                                   lists,
                                   sampled_from,
                                   sets,
                                   tuples)
from shapely.geometry import (LineString,
                              Point)

T = TypeVar('T')

to_finite_floats = partial(floats,
                           allow_infinity=False,
                           allow_nan=False,
                           width=16)
finite_floats = to_finite_floats()
nonnegative_floats = to_finite_floats(min_value=0)
coordinates = slope_intercepts = tuples(finite_floats, finite_floats)
segments = builds(LineString, lists(coordinates,
                                    min_size=2,
                                    max_size=2,
                                    unique=True))

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


def to_overlapping_lines(points: List[Point]) -> Tuple[LineString, LineString]:
    line = LineString([points[0], points[2]])
    other_line = LineString([points[1], points[3]])
    return line, other_line


overlapping_lines = builds(to_overlapping_lines, to_aligned_points(4))


def to_touching_lines(points: List[Point]) -> Tuple[LineString, LineString]:
    line = LineString([points[0], points[1]])
    other_line = LineString([points[1], points[2]])
    return line, other_line


touching_lines = builds(to_touching_lines, to_aligned_points(3))


def to_disjoint_lines(points: List[Point]) -> Tuple[LineString, LineString]:
    line = LineString(points[:2])
    other_line = LineString(points[2:])
    return line, other_line


disjoint_lines = builds(to_disjoint_lines, to_aligned_points(4))

offset_sides = sampled_from(['left', 'right'])
