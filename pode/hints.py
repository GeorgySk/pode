"""Type hints shared across the package."""

from collections.abc import Sequence
from fractions import Fraction
from typing import (Protocol,
                    TypeAlias)

from gon.base import (Contour,
                      Point,
                      Polygon,
                      Segment)
from ground.base import Context

#: Real number accepted as an area requirement. ``int`` is included
#: implicitly, as type checkers accept it wherever ``float`` is expected.
#: We don't use ``numbers.Real``, as type checkers don't support
#: arithmetic on it.
Scalar: TypeAlias = Fraction | float


class ConvexDivisorType(Protocol):
    """Function that divides a polygon into convex parts."""

    def __call__(self,
                 polygon: Polygon,
                 *,
                 extra_points: Sequence[Point] = (),
                 extra_constraints: Sequence[Segment] = (),
                 context: Context) -> list[Contour]:
        """
        Divides the polygon into convex parts.

        :param polygon: polygon to divide
        :param extra_points: points that must become vertices of the
                             parts
        :param extra_constraints: segments that must lie on the borders
                                  of the parts
        :param context: geometric context
        :return: borders of the convex parts
        """
