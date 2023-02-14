# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('..'))
package_path = os.path.abspath('..')
os.environ['PYTHONPATH'] = ''.join((package_path, os.environ.get('PYTHONPATH', '')))

project = 'symmeplot'
copyright = '2023, TJStienstra'
author = 'TJStienstra'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'jupyter_sphinx',
]

templates_path = ['_templates']
exclude_patterns = []

intersphinx_mapping = {
    'sympy': ('https://docs.sympy.org/dev/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

napoleon_numpy_docstring = True
napoleon_custom_sections = [('Other Attributes', 'Attributes')]
napoleon_preprocess_types = True
napoleon_type_aliases = {
    'ArtistBase': 'symmeplot.plot_artists.ArtistBase',
    'Line3D': 'symmeplot.plot_artists.Line3D',
    'Vector3D': 'symmeplot.plot_artists.Vector3D',
    'Circle3D': 'symmeplot.plot_artists.Circle3D',
    'PlotBase': 'symmeplot.plot_base.PlotBase',
    'PlotPoint': 'symmeplot.plot_objects.PlotPoint',
    'PlotLine': 'symmeplot.plot_objects.PlotLine',
    'PlotVector': 'symmeplot.plot_objects.PlotVector',
    'PlotFrame': 'symmeplot.plot_objects.PlotFrame',
    'PlotBody': 'symmeplot.plot_objects.PlotBody',
    'SymMePlotter': 'symmeplot.plotter.SymMePlotter',
    'Point': 'sympy.physics.vector.point.Point',
    'Vector': 'sympy.physics.vector.vector.Vector',
    'ReferenceFrame': 'sympy.physics.vector.frame.ReferenceFrame',
    'Particle': 'sympy.physics.mechanics.particle.Particle',
    'RigidBody': 'sympy.physics.mechanics.rigidbody.RigidBody',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
