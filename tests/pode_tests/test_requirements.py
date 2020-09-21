from math import floor
from numbers import Real

from hypothesis import (given,
                        strategies)
from hypothesis.strategies import DataObject

from tests.strategies.sites import (MIN_PARTITION_SIZE,
                                    requirements)


@given(strategies.data(), strategies.floats(0, 100))
def test_to_partitions(data: DataObject, sum_: Real) -> None:
    min_value = data.draw(strategies.floats(0, sum_))
    size = data.draw(strategies.integers(MIN_PARTITION_SIZE,
                                         min(floor(sum_ / min_value), 100)
                                         if min_value
                                         else 100))
    strategy = requirements(sum_,
                            min_value=min_value,
                            size=size)

    partition = data.draw(strategy)

    assert sum(partition) == sum_
    assert len(partition) == size
    assert all(part >= min_value for part in partition)
