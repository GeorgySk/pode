from collections.abc import Sequence
from typing import (TYPE_CHECKING,
                    Any,
                    TypeVar)

if TYPE_CHECKING:
    from _typeshed import SupportsDunderGT

T = TypeVar("T", bound="SupportsDunderGT[Any]")


def cyclic_equivalence(u: Sequence[T],
                       v: Sequence[T]) -> bool:
    """Taken from https://stackoverflow.com/a/31001317/7851470."""
    n, i, j = len(u), 0, 0
    if n != len(v):
        return False
    while i < n and j < n:
        k = 1
        while k <= n and u[(i + k) % n] == v[(j + k) % n]:
            k += 1
        if k > n:
            return True
        if u[(i + k) % n] > v[(j + k) % n]:
            i += k
        else:
            j += k
    return False
