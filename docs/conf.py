# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))


project = "symmeplot"
copyright = "2024, Timo Stienstra"
author = "Timo Stienstra"
release = "0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "autodocsumm",
    "jupyter_sphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "sympy": ("https://docs.sympy.org/dev/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

autodoc_default_options = {"autosummary": True}

napoleon_numpy_docstring = True
napoleon_custom_sections = [("Other Attributes", "Attributes")]
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
