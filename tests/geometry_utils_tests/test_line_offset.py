from hypothesis import (given,
                        note)
from shapely import wkt
from shapely.geometry import LineString

from pode.geometry_utils import line_offset
from tests.strategies import (finite_floats,
                              nonnegative_floats,
                              offset_sides,
                              segments)
from tests.utils import is_close


@given(segments)
def test_zero_offset(line: LineString) -> None:
    note(f'Line: {wkt.dumps(line, trim=False)}')
    left_line = line_offset(line,
                            distance=0,
                            side='left')
    right_line = line_offset(line,
                             distance=0,
                             side='right')
    assert left_line.almost_equals(line)
    assert right_line.almost_equals(line)


@given(line=segments,
       distance=finite_floats,
       side=offset_sides)
def test_lengths(line: LineString,
                 distance: float,
                 side: str) -> None:
    note(f'Line: {wkt.dumps(line, trim=False)}')
    new_line = line_offset(line,
                           distance=distance,
                           side=side)

    assert is_close(line.length,
                    new_line.length)


@given(line=segments,
       distance=nonnegative_floats,
       side=offset_sides)
def test_boundaries(line: LineString,
                    distance: float,
                    side: str) -> None:
    note(f'Line: {wkt.dumps(line, trim=False)}')
    new_line = line_offset(line,
                           distance=distance,
                           side=side)
    if side == 'right' and distance > 0:
        new_line = LineString(new_line.coords[::-1])
    line_start, line_end = line.boundary
    new_line_start, new_line_end = new_line.boundary
    assert is_close(new_line.project(line_start), 0)
    assert is_close(line.project(new_line_start), 0)
    assert is_close(new_line.project(line_end), new_line.length)
    assert is_close(line.project(new_line_end), line.length)
