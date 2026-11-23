"""Type hints shared across the package."""

from collections.abc import (Iterable,
                             Sequence)
from fractions import Fraction
from typing import (Protocol,
                    TypeAlias)

from gon.base import (Contour,
                      Point,
                      Polygon,
                      Triangulation)
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
                 /,
                 *,
                 extra_points: Sequence[Point],
                 context: Context) -> Iterable[Contour] | Triangulation:
        """
        Divides the polygon into convex parts.

        :param polygon: polygon to divide
        :param extra_points: points that must become vertices of the
                             parts
        :param context: geometric context
        :return: borders of the convex parts, or a triangulation whose
                 triangles are the parts
        """
