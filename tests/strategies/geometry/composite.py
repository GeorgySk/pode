from fractions import Fraction
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

from pode.pode import Site
from pode.utils import edges
from tests.pode_tests.config import (MAX_COORDINATE,
                                     MAX_SITES_COUNT,
                                     MIN_COORDINATE,
                                     MIN_REQUIREMENT,
                                     MIN_SITES_COUNT)
from tests.strategies.geometry.base import (coordinates_strategies_factories,
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
                        ) -> Tuple[Polygon, List[Site]]:
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
    sites = [Site(location=point,
                  requirement=requirement)
             for point, requirement in zip(points, requirements_)]
    return polygon, sites


@st.composite
def _vertices_and_sites(draw: Callable[[st.SearchStrategy[T]], T]
                        ) -> Tuple[List[Point], Set[Site]]:
    multipoint: Multipoint = draw(multipoints)
    slice_ = draw(st.integers(min_value=0,
                              max_value=len(multipoint.points) - 1))
    # TODO: this "subset" strategy could be taken out
    vertices_subset = [multipoint.points[0],
                       *draw(st.permutations(multipoint.points[1:]))[:slice_]]
    requirements_ = draw(requirements(sum_=1,
                                      size=slice_ + 1))
    sites = {Site(location=vertex,
                  requirement=requirement)
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


contours_and_points = _contours_and_points()
multipoints_and_segments = _multipoints_and_segments()
polygons_and_sites = _polygons_and_sites()
vertices_and_sites = _vertices_and_sites()