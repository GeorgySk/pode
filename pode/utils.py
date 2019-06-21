from functools import lru_cache
from itertools import (chain,
                       repeat)
from operator import (itemgetter,
                      mul)
from typing import (Callable,
                    Dict,
                    Iterable,
                    Iterator,
                    Optional,
                    Sequence,
                    Tuple,
                    TypeVar)

from lz.functional import compose
from lz.iterating import pairwise
from shapely.geometry import (LinearRing,
                              LineString,
                              Polygon)
from shapely.ops import split

T = TypeVar('T')


def segments(ring: LinearRing) -> Iterator[LineString]:
    """Yields consecutive lines from the given ring"""
    pairs = pairwise(ring.coords)
    yield from map(LineString, pairs)


def next_enumerate(iterable: Iterable[T],
                   predicate: Callable[[T], bool]) -> Tuple[int, T]:
    """
    Returns index and value of the first element
    satisfying a predicate
    """
    apply_on_value = compose(predicate, itemgetter(1))
    return next(filter(apply_on_value, enumerate(iterable)))


def next_index(iterable: Iterable[T],
               predicate: Callable[[T], bool]) -> int:
    """Returns index of the first element satisfying a predicate"""
    return next(index for index, value in enumerate(iterable)
                if predicate(value))


def last_index(iterable: Sequence[T],
               predicate: Callable[[T], bool]) -> int:
    """Returns index of a last element satisfying a predicate"""
    return len(iterable) - next_index(reversed(iterable), predicate)


def find_if_or_last(predicate: Callable[[T], bool],
                    iterable: Iterable[T]) -> Optional[T]:
    """
    Returns the first element of `iterable`
    satisfying a condition defined by `predicate`.
    If no such an element was found, the last element is returned.
    If the `iterable` was empty, `None` is returned.
    """
    element = None
    for element in iterable:
        if predicate(element):
            return element
    else:
        return element


def difference_by_key(mapping: Dict, other_mapping: Dict) -> Dict:
    """
    Returns dict of key-value pairs from mapping
    which keys are not encountered in other_mapping
    """
    keys_difference = set(mapping) - set(other_mapping)
    return dict(zip(keys_difference, map(mapping.get, keys_difference)))


def scalar_multiplication(vector: Iterable[float],
                          scalar: float) -> Iterator[float]:
    """scalar_multiplication([1 2 3], 2) -> [2 4 6]"""
    yield from map(mul, vector, repeat(scalar))


def right_left_parts(polygon: Polygon,
                     line: LineString) -> Tuple[Polygon, Polygon]:
    """Splits polygon by a line and returns two parts: right and left"""
    part, other_part = safe_split(polygon, line)
    if is_on_the_left(part, line):
        return other_part, part
    return part, other_part


right_part: Callable[[Polygon, LineString], Polygon] = compose(
    itemgetter(0), right_left_parts)
right_part.__doc__ = "Splits polygon by a line and returns the right part"


def is_on_the_left(polygon: Polygon,
                   line: LineString) -> bool:
    """
    Determines if the polygon is on the left side of the line
    according to:
    https://stackoverflow.com/questions/50393718/determine-the-left-and-right-side-of-a-split-shapely-geometry
    Doesn't work for 3D geometries:
    https://github.com/Toblerity/Shapely/issues/709
    """
    ring = LinearRing(chain(line.coords, polygon.centroid.coords))
    return ring.is_ccw


def safe_split(polygon: Polygon,
               line: LineString) -> Tuple[Polygon, Polygon]:
    parts = tuple(split(polygon, line))
    if len(parts) == 1:
        return parts[0], Polygon()
    return parts


def iff(statement: bool,
        other_statement: bool) -> bool:
    """
    'If and only if' - biconditional logical connective.
    Returns `True` if both statements are `True` or `False`.
    """
    return not (statement ^ other_statement)


def midpoint(line: LineString) -> Tuple[float, float]:
    """Returns coordinates of the middle of the input line"""
    return line.interpolate(0.5, normalized=True).coords[0]


@lru_cache(maxsize=None)
def cached_sum(*values: Tuple[float]) -> float:
    """
    Calculates sum of input arguments
    and remembers results for the same input
    """
    return sum(values)
