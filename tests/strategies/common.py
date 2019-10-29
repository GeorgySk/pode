from functools import partial

from hypothesis.strategies import floats

to_finite_floats = partial(floats,
                           allow_infinity=False,
                           allow_nan=False,
                           width=16)
fractions = to_finite_floats(min_value=0,
                             max_value=1)
