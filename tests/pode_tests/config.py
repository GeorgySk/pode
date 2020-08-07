from fractions import Fraction
from functools import partial

from hypothesis.strategies import (floats,
                                   fractions,
                                   integers)


MAX_COORDINATE_EXPONENT = 15
MAX_COORDINATE = 10 ** MAX_COORDINATE_EXPONENT
MIN_COORDINATE = -MAX_COORDINATE
MIN_CONTOUR_SIZE = 3
MAX_CONTOUR_SIZE = 10
MAX_HOLES_SIZE = 5
MIN_REQUIREMENT = 0.1
MIN_SITES_COUNT = 1
MAX_SITES_COUNT = 5
coordinates_strategies_factories = {int: integers}
