from numbers import Real
from typing import (List,
                    Protocol,
                    Sequence,
                    Tuple)

PointType = Tuple[Real, Real]
ContourType = Sequence[PointType]
SegmentType = Tuple[PointType, PointType]
ConvexPartsType = List[Sequence[PointType]]
PolygonType = Tuple[ContourType, Sequence[ContourType]]


class ConvexDivisor(Protocol):
    def __call__(self,
                 border: ContourType,
                 holes: Sequence[ContourType] = (),
                 *,
                 extra_points: ContourType = (),
                 extra_constraints: Sequence[SegmentType] = ()
                 ) -> ConvexPartsType:
        ...
