"""
Implementation the polygon decomposition algorithm by Hert, S. and
Lumelsky, V., 1998.
"""

__author__ = "Georgy Skorobogatov"
__email__ = "skorobogatovgeorgy+github@gmail.com"
__version__ = "0.4.7-alpha"

from gon.base import (Contour,
                      Point,
                      Polygon)

from .pode import Requirement, divide
from .utils import joined_constrained_delaunay_triangles

__all__ = [
    "Contour",
    "Point",
    "Polygon",
    "Requirement",
    "divide",
    "joined_constrained_delaunay_triangles",
]
