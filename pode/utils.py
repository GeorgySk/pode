from functools import lru_cache
from itertools import repeat
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

from lz.functional import (compose,
                           pack)

T = TypeVar('T')
Predicate = Callable[[T], bool]


def next_enumerate(iterable: Iterable[T],
                   predicate: Predicate) -> Tuple[int, T]:
    """
    Returns index and value of the first element
    satisfying a predicate
    """
    apply_on_value = compose(predicate, itemgetter(1))
    return next(filter(apply_on_value, enumerate(iterable)))


def next_index(iterable: Iterable[T],
               predicate: Predicate) -> int:
    """Returns index of the first element satisfying a predicate"""
    return next(index for index, value in enumerate(iterable)
                if predicate(value))


def last_index(iterable: Sequence[T],
               predicate: Predicate) -> int:
    """Returns index of a last element satisfying a predicate"""
    return len(iterable) - next_index(reversed(iterable), predicate) - 1


def find_if_or_last(predicate: Predicate,
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


def iff(statement: bool,
        other_statement: bool) -> bool:
    """
    'If and only if' - biconditional logical connective.
    Returns `True` if both statements are `True` or `False`.
    """
    return not (statement ^ other_statement)


@lru_cache(maxsize=None)
def cached_sum(*values: Tuple[float]) -> float:
    """
    Calculates sum of input arguments
    and remembers results for the same input
    """
    return sum(values)


def starfilter(predicate: Predicate,
               iterable: Iterable[T]) -> Iterator[T]:
    """
    Can be used when elements of an iterable are packed in tuples
    but the predicate function expects separate arguments.
    Similar to `itertools.starmap`.
    """
    return filter(pack(predicate), iterable)
