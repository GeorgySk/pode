import math
from functools import partial

from tests.configs import ABS_TOL

is_close = partial(math.isclose,
                   abs_tol=ABS_TOL)
