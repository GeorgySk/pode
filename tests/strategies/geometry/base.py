import math
from collections.abc import Callable
from fractions import Fraction
from math import floor
from typing import TypeVar

from gon.base import (Multipoint,
                      Polygon,
                      Triangulation)
from hypothesis import strategies as st
from hypothesis_geometry import planar

from tests.pode_tests.config import (MAX_CONTOUR_SIZE,
                                     MAX_COORDINATE,
                                     MAX_HOLES_SIZE,
                                     MIN_CONTOUR_SIZE,
                                     MIN_COORDINATE,
                                     MIN_HOLES_SIZE)
from pode.hints import Scalar
from pode.utils import joined_constrained_delaunay_triangles

T = TypeVar("T")

TRIANGULAR_CONTOUR_SIZE = 3
RECTANGLE_CONTOUR_SIZE = 4
MIN_PARTITION_SIZE = 1

nonnegative_integers = st.integers(min_value=0)
fractions = st.fractions(MIN_COORDINATE, MAX_COORDINATE)
fractions_contours = planar.contours(fractions)
fraction_triangles = planar.polygons(fractions,
                                     max_size=TRIANGULAR_CONTOUR_SIZE)
convex_divisors = st.sampled_from([Triangulation.constrained_delaunay,
                                   joined_constrained_delaunay_triangles])

# each factory takes bounds of its own type, which a `dict` type can't express
coordinates_strategies_factories: dict[
    type[Scalar], Callable[..., st.SearchStrategy[Scalar]]
] = {int: st.integers,
     Fraction: st.fractions,
     float: st.floats}
coordinates_strategies = st.sampled_from([
    factory(MIN_COORDINATE, MAX_COORDINATE)
    for factory in coordinates_strategies_factories.values()])


def coordinates_to_polygons(coordinates: st.SearchStrategy[Scalar]
                            ) -> st.SearchStrategy[Polygon]:
    return planar.polygons(
        coordinates,
        min_size=MIN_CONTOUR_SIZE,
        max_size=MAX_CONTOUR_SIZE,
        min_holes_size=MIN_HOLES_SIZE,
        max_holes_size=MAX_HOLES_SIZE)


def coordinates_to_multipoints(coordinates: st.SearchStrategy[Scalar]
                               ) -> st.SearchStrategy[Multipoint]:
    return planar.multipoints(coordinates, min_size=1)


polygons = coordinates_strategies.flatmap(coordinates_to_polygons)
fractions_polygons = coordinates_to_polygons(fractions)
multipoints = coordinates_strategies.flatmap(coordinates_to_multipoints)


def requirements(sum_: Scalar,  # noqa: C901
                 *,
                 min_value: Scalar = 0,
                 size: int = MIN_PARTITION_SIZE,
                 base: st.SearchStrategy[Scalar] | None = None
                 ) -> st.SearchStrategy[list[Scalar]]:
    if base is None:
        base = st.integers()
    if size < MIN_PARTITION_SIZE:
        raise ValueError("`size` should not be less "
                         f"than {MIN_PARTITION_SIZE}.")
    if not (0 <= min_value <= sum_):
        raise ValueError(f"`min_value` should be in [0, {sum_}] interval.")
    if min_value:
        max_size_approximation = sum_ / min_value
        if math.isfinite(max_size_approximation):
            max_size = floor(max_size_approximation)
            if max_size < size:
                raise ValueError(
                    f"`size` should not be greater than {max_size}.")

    def to_proportions(numbers: list[Scalar]) -> list[Scalar]:
        # `max(number, -number)` is `abs(number)`, but unlike `abs` mypy can
        # infer its type for a union of number types.
        return [2 * max(number, -number) / (1 + number * number)
                for number in numbers]

    def to_partition(proportions: list[Scalar]) -> list[Scalar]:
        factor = sum_ / sum(proportions)
        return [proportion * factor for proportion in proportions]

    def bound_minimum(partition: list[Scalar]) -> list[Scalar]:
        minimum = min(partition)
        if minimum >= min_value:
            return partition
        partition_size = len(partition)
        denominator = sum_ - partition_size * minimum
        slope = sum_ - partition_size * min_value
        intercept = sum_ * (min_value - minimum)
        return [max((part * slope + intercept) / denominator, min_value)
                for part in partition]

    def normalize(partition: list[Scalar]) -> list[Scalar]:
        partition_sum = sum(partition)
        if partition_sum < sum_:
            arg_min = min(range(len(partition)),
                          key=partition.__getitem__)
            partition[arg_min] += sum_ - partition_sum
        elif partition_sum > sum_:
            arg_max = max(range(len(partition)),
                          key=partition.__getitem__)
            partition[arg_max] -= partition_sum - sum_
        return partition

    def is_valid(partition: list[Scalar]) -> bool:
        return sum(partition) == sum_

    return (st.lists(base,
                     min_size=size,
                     max_size=size)
            .filter(any)
            .map(to_proportions)
            .map(to_partition)
            .map(bound_minimum)
            .map(normalize)
            .filter(is_valid))
