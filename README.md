pode
===

[![](https://github.com/GeorgySk/pode/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/GeorgySk/pode/actions/workflows/ci.yml?query=branch%3Amain "GitHub Actions")
[![](https://codecov.io/gh/GeorgySk/pode/branch/main/graph/badge.svg)](https://codecov.io/gh/GeorgySk/pode "Codecov")
[![](https://readthedocs.org/projects/pode/badge/?version=latest)](https://pode.readthedocs.io/en/latest "Documentation")
[![](https://img.shields.io/github/license/GeorgySk/pode.svg)](https://github.com/GeorgySk/pode/blob/main/LICENSE "License")
[![](https://badge.fury.io/py/pode.svg)](https://badge.fury.io/py/pode "PyPI")
[![](https://img.shields.io/pypi/pyversions/pode.svg)](https://pypi.org/project/pode "Supported Python versions")
[![](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/GeorgySk/copier-pypackage "Generated from the copier-pypackage template")

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

In what follows, `python` is an alias for `python3.11`
or any later version.

Installation
------------

Install the latest `pip` & `setuptools` packages versions
```bash
python -m pip install --upgrade pip setuptools
```

### User

Download and install the latest stable version from `PyPI` repository
```bash
python -m pip install --upgrade pode
```

### Developer

Download the latest version from `GitHub` repository
```bash
git clone https://github.com/GeorgySk/pode.git
cd pode
```

Install
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
[bump-my-version](https://github.com/callowayproject/bump-my-version/tree/master?tab=readme-ov-file#installation).

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

### Publishing

Pushing a final release tag `vX.Y.Z` runs the CI publish jobs, which upload
the package to PyPI and push
the `prod` Docker image to the GitHub container registry (GHCR).
Pre-release tags such as `vX.Y.Z-alpha` are not published.

`bump-my-version bump release` commits and tags the release, so push both:
```bash
git push --follow-tags
```

One-time setup:

- GitHub: on PyPI, add a
  [Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
  for this repository with workflow `ci.yml` and environment `pypi`.
- GitHub: add a `CODECOV_TOKEN` repository secret from
  [Codecov](https://codecov.io/).

### Linting, formatting and type checking

[Ruff](https://docs.astral.sh/ruff/) is used for both linting and formatting
(installed with the `dev` dependency group).

Check code
```bash
ruff check .
```

Fix automatically fixable issues
```bash
ruff check --fix .
```

Format code
```bash
ruff format .
```

[mypy](https://mypy.readthedocs.io/) checks types in strict mode
(configured in `pyproject.toml`)
```bash
mypy .
```

The package ships a `py.typed` marker, so projects that use it get its type
hints too.

To run these checks (Ruff and mypy) automatically on each commit,
install the [pre-commit](https://pre-commit.com/) hooks
(once, after the repository has been initialized with `git init`)
```bash
pre-commit install
```

### Running tests

Install the project with the `dev` dependency group (pytest, pytest-cov,
Hypothesis)
```bash
poetry install --with dev
```

Run the tests
```bash
poetry run pytest
```

Run the tests with a coverage report, as CI does
```bash
poetry run pytest --cov --cov-report=term
```

Inside the dev container, dependencies are already installed into the
container's Python, so `poetry run` can be dropped: `pytest`.

Inside a `Docker` container (the default `IMAGE_TYPE=dev` image includes the
`dev` group)
```bash
docker compose run --rm --entrypoint pytest pode
```

### Updating from the template

This project was generated from the
[copier-pypackage](https://github.com/GeorgySk/copier-pypackage) template.
To pull in a newer release of the template (CI, tooling and configuration
fixes), commit your changes, then run
```bash
copier update --defaults
```
Copier merges the template's changes into the project and keeps yours.
Review them with `git diff`, resolve any conflict markers and commit.
`CHANGES.rst` and `pode/__init__.py` are never touched by
updates, and `.copier-answers.yml` must not be edited by hand.

The `Template update` workflow (`.github/workflows/template-update.yml`) does
this every Monday and opens a pull request when there is a new template
release (run it by hand from the Actions tab). Conflicting changes are
committed with conflict markers in a draft pull request: resolve them on its
branch. There is one pull request per release, so closing one skips that
release. The template must be at a URL GitHub can clone (the `_src_path` in
`.copier-answers.yml`).

With the default `GITHUB_TOKEN`, enable "Allow GitHub Actions to create and
approve pull requests" (Settings → Actions → General). Even then the pull
request doesn't run CI, and updates that change `.github/workflows` can't be
pushed. To avoid both, add a `TEMPLATE_UPDATE_TOKEN` repository secret: a
fine-grained personal access token for this repository with read and write
access to Contents, Pull requests and Workflows.
