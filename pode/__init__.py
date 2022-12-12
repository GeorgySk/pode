"""Polygon decomposition"""
__version__ = '0.2.0'

from .pode import (divide,
                   Contour,
                   Point,
                   Polygon,
                   Requirement)
from .utils import joined_constrained_delaunay_triangles
