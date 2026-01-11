Installation
============

Installation Users
------------------

SymMePlot is available on both PyPI and Conda-Forge. To install the latest release from
PyPI, run: ::

    pip install symmeplot

To use the `matplotlib`_ backend, you’ll need to install `matplotlib`_ as well: ::

    pip install matplotlib

To use the `pyqtgraph`_ backend, you’ll need to install `pyqtgraph`_ with some optional
dependencies as well: ::

    pip install pyqtgraph pyopengl pyqt6

Installation Developers
-----------------------
SymMePlot uses `uv`_ as package manager. To install SymMePlot after installing
`uv`_ and cloning the repository, run: ::

    uv venv
    uv pip install -e .

SymMePlot offers dependency groups to assist developers:

- ``lint``: packages required for linting.
- ``test``: packages required for testing.
- ``docs``: packages required for building the documentation.

To install optional dependencies from a specific group, run: ::

    uv pip install -e .[<group>]

The backends are optional dependencies. These can be installed using: ::

    uv pip install -e .[mpl_backend]
    uv pip install -e .[pg_backend]

To install everything at once, run: ::

    uv sync --all-extras

To quickly check code for linting errors, it is recommended to set up ``pre-commit``
hooks by executing: ::

    pre-commit install

You can install ``pre-commit`` by running: ::

    uv pip install pre-commit

.. _uv: https://docs.astral.sh/uv/
.. _matplotlib: https://matplotlib.org
.. _pyqtgraph: https://www.pyqtgraph.org
