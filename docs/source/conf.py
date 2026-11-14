# Configuration file for the Sphinx documentation builder.

import sys
import tomllib
from datetime import date
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

# tells autodoc where to find the code
sys.path.insert(0, str(project_root))

with (project_root / "pyproject.toml").open("rb") as file:
    project_data = tomllib.load(file)["project"]

# -- Project information

project = project_data["name"]
author = ", ".join(
    author_["name"] for author_ in project_data.get("authors", [])
)
copyright = f"{date.today().year}, {author}"

# release is the full version, version the short X.Y
release = project_data["version"]
version = ".".join(release.split(".")[:2])

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
