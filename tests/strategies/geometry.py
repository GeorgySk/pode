from itertools import (product,
                       repeat,
                       starmap)
from math import (cos,
                  radians,
                  sin)
from typing import (List,
                    Sequence,
                    Set,
                    Tuple,
                    TypeVar)

from hypothesis.strategies import (SearchStrategy,
                                   builds,
                                   integers,
                                   lists,
                                   sampled_from,
                                   sets,
                                   tuples)
from lz.iterating import interleave
from shapely.affinity import rotate
from shapely.geometry import (LineString,
                              MultiPolygon,
                              Point,
                              Polygon,
                              box)

from tests.configs import (ABS_TOL,
                           MAX_ITERABLES_SIZE)
from tests.strategies.common import (fractions,
                                     to_finite_floats)
from tests.utils import (are_sparse,
                         form_object_with_area,
                         has_no_close_points,
                         have_no_close_points)

MAX_GRID_SIDE_SIZE = 5

T = TypeVar('T')

iterable_sizes = integers(min_value=0,
                          max_value=MAX_ITERABLES_SIZE)
grid_side_sizes = integers(min_value=0,
                           max_value=MAX_GRID_SIDE_SIZE)
polygon_side_counts = integers(min_value=3,
                               max_value=MAX_ITERABLES_SIZE)
finite_floats = to_finite_floats()
positive_floats = to_finite_floats(min_value=ABS_TOL,
                                   exclude_min=True)
coordinates = tuples(finite_floats, finite_floats)
angles = to_finite_floats(min_value=-360,
                          max_value=360)
points = builds(Point, finite_floats, finite_floats)
segments = builds(LineString, lists(coordinates,
                                    min_size=2,
                                    max_size=2,
                                    unique=True))
# too small segments conflict with precision errors
segments = segments.filter(has_no_close_points)


def points_by_distances(line: LineString,
                        distances: Sequence[float]) -> List[Point]:
    return [line.interpolate(distance, normalized=True)
            for distance in distances]


def to_aligned_points(count: int) -> SearchStrategy[List[Point]]:
    sorted_fractions = sets(fractions,
                            min_size=count,
                            max_size=count).map(sorted)
    return builds(points_by_distances,
                  segments,
                  sorted_fractions)


def to_containing_lines(points: List[Point]) -> Tuple[LineString, LineString]:
    long_line = LineString([points[0], points[3]])
    short_line = LineString([points[1], points[2]])
    return long_line, short_line


lines_one_in_another = builds(to_containing_lines, to_aligned_points(4))
lines_one_in_another = lines_one_in_another.filter(have_no_close_points)


def to_overlapping_lines(points: List[Point]) -> Tuple[LineString, LineString]:
    line = LineString([points[0], points[2]])
    other_line = LineString([points[1], points[3]])
    return line, other_line


overlapping_lines = builds(to_overlapping_lines, to_aligned_points(4))
overlapping_lines = overlapping_lines.filter(have_no_close_points)


def to_touching_lines(points: List[Point]) -> Tuple[LineString, LineString]:
    line = LineString([points[0], points[1]])
    other_line = LineString([points[1], points[2]])
    return line, other_line


touching_lines = builds(to_touching_lines, to_aligned_points(3))
touching_lines = touching_lines.filter(have_no_close_points)


def to_disjoint_lines(points: List[Point]) -> Tuple[LineString, LineString]:
    line = LineString(points[:2])
    other_line = LineString(points[2:])
    return line, other_line


disjoint_lines = builds(to_disjoint_lines, to_aligned_points(4))
disjoint_lines = disjoint_lines.filter(have_no_close_points)

three_unique_coordinates_lists = lists(coordinates,
                                       min_size=3,
                                       max_size=3,
                                       unique=True)
triangles = (three_unique_coordinates_lists
             .filter(form_object_with_area)
             .map(Polygon))

circles = builds(Point.buffer, points, positive_floats)

straight_rectangles = (builds(box, *repeat(finite_floats, 4))
                       .filter(form_object_with_area))
rectangles = builds(rotate, straight_rectangles, angles)

convex_polygons = triangles | circles | rectangles


def l_shaped(xs: Set[float],
             ys: Set[float],
             angle: float):
    x0, x1, x2 = sorted(xs)
    y0, y1, y2 = sorted(ys)
    vertical = box(x0, y0, x1, y2)
    horizontal = box(x1, y0, x2, y1)
    polygon = vertical.union(horizontal)
    return rotate(polygon, angle)


l_shaped_polygons = builds(l_shaped,
                           xs=sets(finite_floats, min_size=3, max_size=3),
                           ys=sets(finite_floats, min_size=3, max_size=3),
                           angle=angles)


def window(xs: Set[float],
           ys: Set[float],
           angle: float):
    x0, x1, x2, x3 = sorted(xs)
    y0, y1, y2, y3 = sorted(ys)
    vertical = box(x0, y0, x3, y3)
    horizontal = box(x1, y1, x2, y2)
    polygon = vertical.difference(horizontal)
    return rotate(polygon, angle)


window_polygons = builds(window,
                         xs=sets(finite_floats, min_size=4, max_size=4),
                         ys=sets(finite_floats, min_size=4, max_size=4),
                         angle=angles)


def star(rs: Set[float],
         n: int):
    r0, r1 = rs
    delta_angle = 360 / n
    angles0 = [delta_angle * i for i in range(n)]
    angles1 = [angle + delta_angle / 2 for angle in angles0]
    angles0 = map(radians, angles0)
    angles1 = map(radians, angles1)
    coords0 = ((r0 * cos(angle), r0 * sin(angle)) for angle in angles0)
    coords1 = ((r1 * cos(angle), r1 * sin(angle)) for angle in angles1)
    coords = interleave([coords0, coords1])
    return Polygon(coords)


star_polygons = builds(star,
                       rs=sets(positive_floats, min_size=2, max_size=2),
                       n=polygon_side_counts)

nonconvex_polygons = l_shaped_polygons | window_polygons | star_polygons
nonconvex_polygons = nonconvex_polygons.filter(has_no_close_points)

polygons = convex_polygons | nonconvex_polygons

disjoint_polygons_lists = (lists(polygons,
                                 max_size=MAX_ITERABLES_SIZE)
                           .filter(are_sparse))
same_polygons_iterators = builds(repeat, polygons, iterable_sizes)


def polygons_grid(cell_size: Tuple[float, float],
                  grid_shape: Tuple[int, int],
                  origin: Tuple[float, float],
                  angle: float) -> List[Polygon]:
    """
    A simple case of connected polygons that form a grid.
    Used to run simple representative tests.
    """
    dx, dy = cell_size
    ox, oy = origin

    def cell_coords(i: int, j: int) -> List[Tuple[float, float]]:
        left_x = ox + i * dx
        right_x = left_x + dx
        low_y = oy + j * dy
        top_y = low_y + dy
        return [(left_x, low_y), (right_x, low_y),
                (right_x, top_y), (left_x, top_y)]

    nx, ny = grid_shape
    indices = product(range(nx), range(ny))
    cells_coords = starmap(cell_coords, indices)
    cells = MultiPolygon(map(Polygon, cells_coords))
    return [] if cells.is_empty else list(rotate(cells, angle).geoms)


def polygons_grid_and_dimension(cell_size: Tuple[float, float],
                                grid_shape: Tuple[int, int],
                                origin: Tuple[float, float],
                                angle: float
                                ) -> Tuple[List[Polygon], Tuple[int, int]]:
    grid = polygons_grid(cell_size=cell_size,
                         grid_shape=grid_shape,
                         origin=origin,
                         angle=angle)
    return grid, grid_shape


polygons_grids_and_dimensions = builds(
    polygons_grid_and_dimension,
    cell_size=tuples(positive_floats, positive_floats),
    grid_shape=tuples(grid_side_sizes, grid_side_sizes),
    origin=tuples(finite_floats, finite_floats),
    angle=angles)
