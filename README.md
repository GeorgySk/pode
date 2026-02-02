pode
===========


[![](https://github.com/GeorgySk/pode/workflows/CI/badge.svg)](https://github.com/GeorgySk/pode/actions/workflows/ci.yml "Github Actions")
[![](https://codecov.io/gh/GeorgySk/gon/branch/master/graph/badge.svg)](https://codecov.io/gh/GeorgySk/pode "Codecov")
[![](https://readthedocs.org/projects/pode/badge/?version=latest)](https://pode.readthedocs.io/en/latest "Documentation")
[![](https://img.shields.io/github/license/GeorgySk/pode.svg)](https://github.com/GeorgySk/pode/blob/master/LICENSE "License")
[![](https://badge.fury.io/py/pode.svg)](https://badge.fury.io/py/pode "PyPI")

Summary
-------

Pode is a Python library that implements an algorithm of 
[Hert, S. and Lumelsky, V., 1998](https://www.worldscientific.com/doi/abs/10.1142/S0218195998000230)
for a polygon decomposition into separate parts depending on the area 
requirements.

Main features are
- all calculations are robust for floating point numbers
& precise for integral numbers (like `int`);
- support for partition of convex/nonconvex polygons with/without holes;
- support for anchored partition, free partition, and a mixture of both 
where anchored polygon partition requires the resulting polygon parts 
to contain specified points called "anchors" or "sites", and free 
partition does not have any constraints on the resulting geometries; 
- most of the code is covered with property-based tests.
---

In what follows, `python` is an alias for `python3.8` or any later
version.

Installation
------------

Install the latest `pip` & `setuptools` packages versions:
  ```bash
  python -m pip install --upgrade pip setuptools
  ```

### User

Download and install the latest stable version from PyPI repository:
  ```bash
  python -m pip install --upgrade pode
  ```

### Developer

Download the latest version from `GitHub` repository
```bash
git clone https://github.com/GeorgySk/pode.git
cd pode
```

Install using Poetry:
  ```bash
  poetry install
  ```

Usage
-----
```python
>>> from pode import divide, Contour, Point, Polygon, Requirement
>>> contour = Contour([Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)])
>>> polygon = Polygon(contour)
>>> requirements = [Requirement(0.5), Requirement(0.5, point=Point(1, 1))]
>>> parts = divide(polygon, requirements)
>>> parts[0].area == parts[1].area < polygon.area
True
>>> Point(1, 1) in parts[1]
True

```
Currently, the algorithm uses constrained Delaunay triangulation to form convex 
parts which are used internally for a convex-polygon partition.
This triangulation, however, affects the resulting partition. The resulting 
parts can have zigzag pattern separating different parts. In this way, the 
solution can be far from being optimal in sense of the number of sides of the
resulting polygons. Alternatively, we implemented a helper function that would 
join neighboring triangles if they form a larger convex part. It can be 
imported as `from pode import joined_constrained_delaunay_triangles` and used
as `divide(polygon, requirements, joined_constrained_delaunay_triangles)`. But, 
in the future, a better algorithm should be implemented, like 
[Greene, D.H., 1983](https://www.goodreads.com/book/show/477772.Advances_in_Computing_Research_Volume_1) 
or [Chazelle, B. and Dobkin, D., 1979](https://dl.acm.org/doi/abs/10.1145/800135.804396).


Development
-----------

### Bumping version

#### Preparation

Install
[bump-my-version](https://github.com/callowayproject/bump-my-version?tab=readme-ov-file#installation).

#### Pre-release

Choose which version number category to bump following [semver
specification](http://semver.org/).

Test bumping version
```bash
bump-my-version bump --dry-run --verbose $CATEGORY
```

where `$CATEGORY` is the target version number category name, possible
values are `patch`/`minor`/`major`.

Bump version
```bash
bump-my-version bump --verbose $CATEGORY
```

This will set version to `major.minor.patch-alpha`. 

#### Release

Test bumping version
```bash
bump-my-version bump --dry-run --verbose release
```

Bump version
```bash
bump-my-version bump --verbose release
```

This will set version to `major.minor.patch`.

#### Notes

To avoid inconsistency between branches and pull requests,
bumping version should be merged into `master` branch 
as separate pull request.

### Running tests

Plain:
  ```bash
  pytest
  ```

Inside `Docker` container:
  ```bash
  docker compose run --entrypoint pytest pode
  ```
