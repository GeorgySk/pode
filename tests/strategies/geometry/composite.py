from fractions import Fraction
from itertools import zip_longest
from numbers import Real
from typing import (Callable,
                    Iterable,
                    List,
                    Set,
                    Tuple,
                    TypeVar)

from gon.discrete import Multipoint
from gon.linear import (Contour,
                        Segment)
from gon.primitive import Point
from gon.shaped import Polygon
from hypothesis import strategies as st
from hypothesis_geometry import planar

from pode.pode import Requirement
from pode.utils import edges
from tests.pode_tests.config import (MAX_COORDINATE,
                                     MAX_SITES_COUNT,
                                     MIN_COORDINATE,
                                     MIN_REQUIREMENT,
                                     MIN_SITES_COUNT)
from tests.strategies.geometry.base import (coordinates_strategies_factories,
                                            fraction_coordinates,
                                            multipoints,
                                            polygons)
from tests.strategies.sites import requirements

T = TypeVar('T')


@st.composite
def _contours_and_points(draw: Callable[[st.SearchStrategy[T]], T]
                         ) -> Tuple[Contour, Point, Point]:
    fraction_contours = planar.contours(st.fractions(MIN_COORDINATE,
                                                     MAX_COORDINATE))
    contour: Contour = draw(st.builds(Contour.from_raw, fraction_contours))
    segments = list(edges(contour))
    segments_and_fractions = st.tuples(st.sampled_from(segments),
                                       st.builds(Fraction,
                                                 st.floats(0, 1,
                                                           exclude_max=True)))
    segments_and_fractions_pair = draw(st.sets(segments_and_fractions,
                                               min_size=2,
                                               max_size=2))
    points = [
        Point(fraction * (segment.end.x - segment.start.x) + segment.start.x,
              fraction * (segment.end.y - segment.start.y) + segment.start.y)
        for segment, fraction in segments_and_fractions_pair]
    return contour, *points


@st.composite
def _multipoints_and_segments(draw: Callable[[st.SearchStrategy[T]], T]
                              ) -> Tuple[Multipoint, Segment]:
    fraction_contours = planar.contours(st.fractions(MIN_COORDINATE,
                                                     MAX_COORDINATE))
    contour: Contour = draw(st.builds(Contour.from_raw, fraction_contours))
    segments = list(edges(contour))
    multipoint = Multipoint(*contour.vertices)
    edge = draw(st.sampled_from(segments))
    return multipoint, edge


@st.composite
def _polygons_and_sites(draw: Callable[[st.SearchStrategy[T]], T]
                        ) -> Tuple[Polygon, List[Requirement]]:
    polygon: Polygon = draw(polygons)
    x_min, x_max, y_min, y_max = _bounding_box(polygon.border.vertices)
    coordinates = coordinates_strategies_factories[type(x_min)]
    bounded_points = st.builds(Point,
                               coordinates(x_min, x_max),
                               coordinates(y_min, y_max))
    points_in_polygon = bounded_points.filter(polygon.__contains__)
    points = draw(st.lists(points_in_polygon,
                           min_size=MIN_SITES_COUNT,
                           max_size=MAX_SITES_COUNT,
                           unique=True))
    requirements_ = draw(requirements(sum_=1,
                                      min_value=MIN_REQUIREMENT,
                                      size=len(points)))
    sites = [Requirement(requirement, point=point)
             for point, requirement in zip(points, requirements_)]
    return polygon, sites


@st.composite
def _polygons_and_requirements(draw: Callable[[st.SearchStrategy[T]], T]
                               ) -> Tuple[Polygon, List[Requirement]]:
    polygon: Polygon = draw(polygons)
    requirements_count = draw(st.integers(MIN_SITES_COUNT, MAX_SITES_COUNT))
    areas = draw(requirements(sum_=1,
                              min_value=MIN_REQUIREMENT,
                              size=requirements_count))
    x_min, x_max, y_min, y_max = _bounding_box(polygon.border.vertices)
    coordinates = coordinates_strategies_factories[type(x_min)]
    bounded_points = st.builds(Point,
                               coordinates(x_min, x_max),
                               coordinates(y_min, y_max))
    points_in_polygon = bounded_points.filter(polygon.__contains__)
    points = draw(st.lists(points_in_polygon,
                           min_size=0,
                           max_size=requirements_count,
                           unique=True))
    requirements_ = [Requirement(area, point=point)
                     for area, point in zip_longest(areas, points)]
    return polygon, requirements_


@st.composite
def _vertices_and_sites(draw: Callable[[st.SearchStrategy[T]], T]
                        ) -> Tuple[List[Point], Set[Requirement]]:
    multipoint: Multipoint = draw(multipoints)
    slice_ = draw(st.integers(min_value=0,
                              max_value=len(multipoint.points) - 1))
    # TODO: this "subset" strategy could be taken out
    vertices_subset = [multipoint.points[0],
                       *draw(st.permutations(multipoint.points[1:]))[:slice_]]
    requirements_ = draw(requirements(sum_=1,
                                      size=slice_ + 1,
                                      min_value=MIN_REQUIREMENT))
    sites = {Requirement(requirement, point=vertex)
             for vertex, requirement in zip(vertices_subset, requirements_)}
    return multipoint.points, sites


def _bounding_box(points: Iterable[Point]) -> Tuple[Real, Real, Real, Real]:
    """Builds bounding box from points."""
    points = ((point.x, point.y) for point in points)
    x_min, y_min = x_max, y_max = next(points)
    for x, y in points:
        x_min, x_max = min(x_min, x), max(x_max, x)
        y_min, y_max = min(y_min, y), max(y_max, y)
    return x_min, x_max, y_min, y_max


@st.composite
def _convex_contour_points(draw: Callable[[st.SearchStrategy[T]], T]
                           ) -> List[Point]:
    contour: Contour = draw(
        st.builds(Contour.from_raw,
                  planar.convex_contours(fraction_coordinates)))
    fractions_lists = draw(st.lists(
        st.lists(st.fractions(min_value=0,
                              max_value=1)
                 .filter(lambda x: x not in {0, 1}),
                 unique=True).map(sorted),
        min_size=len(contour.vertices),
        max_size=len(contour.vertices)))
    points = []
    points_pairs = zip(contour.vertices,
                       contour.vertices[1:] + contour.vertices[:1])

    for (point, next_point), fractions_list in zip(points_pairs,
                                                   fractions_lists):
        points.append(point)
        new_points = [interpolate(point, next_point, fraction)
                      for fraction in fractions_list]
        points.extend(new_points)
    return points


def interpolate(start: Point,
                end: Point,
                fraction: Fraction) -> Point:
    return Point(start.x + (end.x - start.x) * fraction,
                 start.y + (end.y - start.y) * fraction)


contours_and_points = _contours_and_points()
multipoints_and_segments = _multipoints_and_segments()
polygons_and_sites = _polygons_and_sites()
vertices_and_sites = _vertices_and_sites()
convex_contour_points = _convex_contour_points()
polygons_and_requirements = _polygons_and_requirements()
