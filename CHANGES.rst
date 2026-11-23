Changelog
=========

All notable changes to this project are documented in this file.
The format is based on `Keep a Changelog <https://keepachangelog.com/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/>`_.

Unreleased
----------

Changed
~~~~~~~

- ``ConvexDivisorType`` now matches how ``divide`` calls the divisor:
  the polygon is positional-only, ``extra_points`` is required, the unused
  ``extra_constraints`` parameter is removed, and the return type is
  ``Iterable[Contour] | Triangulation``.

Fixed
~~~~~

- The fast adjacency-graph construction for the default convex divisor
  (``Triangulation.constrained_delaunay``) was never used, because the
  divisor was compared by identity. Building the graph for large polygons
  is now much faster.

0.4.13 (2026-11-21)
------------------------------

Added
~~~~~

- Property-based tests that check the output of ``divide``: the number and
  types of the parts, their areas, that each anchored part contains its
  site, that the parts lie within the polygon, and that they don't overlap.

0.4.12 (2026-11-18)
------------------------------

Fixed
~~~~~

- The adjacency graph test no longer fails for convex divisors that join
  triangles into larger convex parts (such as
  ``joined_constrained_delaunay_triangles``), which can return a convex
  polygon as a single part.

0.4.11 (2026-11-18)
------------------------------

Fixed
~~~~~

- ``divide`` no longer returns overlapping parts when several requirements
  have the same area and site (or no site).

0.4.10 (2026-11-17)
------------------------------

Added
~~~~~

- Pushing a release tag also creates a GitHub release, with the version's
  section of this changelog as its notes.

0.4.9 (2026-11-17)
------------------------------

Fixed
~~~~~

- ``divide`` no longer restarts from the first part of the polygon after
  each split, and no longer tries to process parts that were already
  removed.

0.4.8 (2026-11-16)
------------------------------

Fixed
~~~~~

- The documentation badge in the README now points to the right Read the
  Docs project.

0.4.7 (2026-11-14)
------------------------------

Added
~~~~~

- Type hints for the whole package, checked by mypy in strict mode, and a
  ``py.typed`` marker, so type checkers use them.
- ``__all__`` in the ``pode`` package.

Changed
~~~~~~~

- Python 3.11 or later is required (3.14.1 is excluded, as networkx
  excludes it).
- networkx 3.6.1 or later is required, instead of 2.x.
- ``Contour``, ``Point`` and ``Polygon`` are re-exported from ``gon.base``
  directly.
- The project is generated from a Copier template; linting and formatting
  use Ruff.
