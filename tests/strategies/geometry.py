from itertools import (product,
                       repeat,
                       starmap)
from typing import (List,
                    Sequence,
                    Set,
                    Tuple,
                    TypeVar)

from hypothesis.strategies import (SearchStrategy,
                                   builds,
                                   integers,
                                   lists,
                                   sets,
                                   tuples)
from hypothesis_geometry import planar
from shapely.affinity import rotate
from shapely.geometry import (GeometryCollection,
                              LineString,
                              LinearRing,
                              MultiLineString,
                              MultiPoint,
                              MultiPolygon,
                              Point,
                              Polygon,
                              box)

from tests.configs import (ABS_TOL,
                           MAX_ITERABLES_SIZE)
from tests.strategies.common import (fractions,
                                     to_finite_floats)
from tests.utils import (are_sparse,
                         has_no_close_points,
                         have_no_close_points)

MAX_GRID_SIDE_SIZE = 5

T = TypeVar('T')

iterable_sizes = integers(min_value=0,
                          max_value=MAX_ITERABLES_SIZE)
grid_side_sizes = integers(min_value=0,
                           max_value=MAX_GRID_SIDE_SIZE)
finite_floats = to_finite_floats()
positive_floats = to_finite_floats(min_value=ABS_TOL,
                                   exclude_min=True)

angles = to_finite_floats(min_value=-360,
                          max_value=360)
segments = builds(LineString, planar.segments(finite_floats))

empty_linestrings = builds(LineString)
non_segments = builds(LineString,
                      planar.contours(finite_floats,
                                      min_size=3,
                                      max_size=MAX_ITERABLES_SIZE))


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

empty_linear_rings = builds(LinearRing)
nonempty_linear_rings = builds(LinearRing,
                               planar.contours(finite_floats,
                                               max_size=MAX_ITERABLES_SIZE))

convex_polygons = builds(Polygon,
                         planar.convex_contours(finite_floats,
                                                max_size=MAX_ITERABLES_SIZE))


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

nonconvex_polygons = builds(
    Polygon, planar.concave_contours(finite_floats,
                                     max_size=MAX_ITERABLES_SIZE))
nonconvex_polygons |= window_polygons
nonconvex_polygons = nonconvex_polygons.filter(has_no_close_points)

empty_polygons = builds(Polygon)
nonempty_polygons = convex_polygons | nonconvex_polygons
polygons = empty_polygons | nonempty_polygons

polygons_lists = lists(polygons, max_size=MAX_ITERABLES_SIZE)
nonempty_polygons_lists = lists(polygons,
                                min_size=1,
                                max_size=MAX_ITERABLES_SIZE)
disjoint_polygons_lists = polygons_lists.filter(are_sparse)
same_nonempty_polygons_iterators = builds(repeat,
                                          nonempty_polygons,
                                          iterable_sizes)


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


def dont_coincide_in_centroid(polygon_and_line: Tuple[Polygon, LineString]
                              ) -> bool:
    polygon, line = polygon_and_line
    return not line.intersects(polygon.centroid)


polygons_and_segments = tuples(nonempty_polygons, segments)
polygons_and_segments_not_passing_through_centroids = (
    polygons_and_segments.filter(dont_coincide_in_centroid))
