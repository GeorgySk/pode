from numbers import Real
from typing import (List,
                    Protocol,
                    Sequence,
                    Tuple)

PointType = Tuple[Real, Real]
ContourType = Sequence[PointType]
SegmentsType = Sequence[Tuple[PointType, PointType]]
ConvexPartsType = List[Sequence[PointType]]


class ConvexDivisor(Protocol):
    def __call__(self,
                 border: ContourType,
                 holes: Sequence[ContourType] = (),
                 *,
                 extra_points: ContourType = (),
                 extra_constraints: SegmentsType = ()) -> ConvexPartsType:
        ...
