"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Project information -----------------------------------------------------
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""

from __future__ import annotations

import sys
from pathlib import Path

from docutils import nodes as docutils_nodes

import symmeplot

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


project = "symmeplot"
copyright = "2024, Timo Stienstra"
author = "Timo Stienstra"
release = symmeplot.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "autodocsumm",
    "jupyter_sphinx",
]


def setup(app: object) -> None:
    """Do setup for Sphinx app to register custom roles."""

    def mpltype_role(
        _name: str,
        _rawtext: str,
        text: str,
        _lineno: int,
        _inliner: object,
        options: dict[str, object] | None = None,
        _content: list[str] | None = None,
    ) -> tuple[list[object], list[object]]:
        """Handle :mpltype: role."""
        if options is None:
            options = {}
        node = docutils_nodes.literal(text, text, **options)
        return [node], []

    app.add_role("mpltype", mpltype_role)


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "sympy": ("https://docs.sympy.org/dev/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

autodoc_default_options = {"autosummary": True}

napoleon_numpy_docstring = True
napoleon_custom_sections = [
    ("Other Attributes", "Attributes"),
    ("Explanation", "notes_style"),
]
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "Point": "sympy.physics.vector.point.Point",
    "Vector": "sympy.physics.vector.vector.Vector",
    "ReferenceFrame": "sympy.physics.vector.frame.ReferenceFrame",
    "Particle": "sympy.physics.mechanics.particle.Particle",
    "RigidBody": "sympy.physics.mechanics.rigidbody.RigidBody",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
