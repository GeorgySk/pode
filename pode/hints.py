from typing import (List,
                    Protocol,
                    Sequence)

from gon.base import (Contour,
                      Point,
                      Polygon,
                      Segment)
from ground.base import Context


class ConvexDivisorType(Protocol):
    def __call__(self,
                 polygon: Polygon,
                 *,
                 extra_points: Sequence[Point] = (),
                 extra_constraints: Sequence[Segment] = (),
                 context: Context) -> List[Contour]:
        ...
