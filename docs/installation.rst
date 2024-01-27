Installation
============

Installation Users
------------------

SymMePlot is not yet available on PyPI. Therefore, you’ll need to install the
development version from GitHub using: ::

    pip install git+https://github.com/TJStienstra/symmeplot.git

To use the `matplotlib`_ backend, you’ll need to install `matplotlib`_ as well: ::

    pip install matplotlib

To use the `pyqtgraph`_ backend, you’ll need to install `pyqtgraph`_ with some optional
dependencies as well: ::

    pip install pyqtgraph pyopengl pyqt6

Installation Developers
-----------------------
SymMePlot uses `poetry`_ as package manager. To install SymMePlot after installing
`poetry`_ and cloning the repository, run: ::

    poetry install

SymMePlot offers dependency groups to assist developers:

- ``lint``: packages required for linting.
- ``test``: packages required for testing.
- ``docs``: packages required for building the documentation.

To install optional dependencies from a specific group, run: ::

    poetry install --with <group>

The backends are optional dependencies. These can be installed using: ::

    poetry install --extras mpl_backend
    poetry install --extras pg_backend

To install everything at once, run: ::

    poetry install --with lint,test,docs --all-extras

To quickly check code for linting errors, it is recommended to set up ``pre-commit``
hooks by executing: ::

    pre-commit install

.. _poetry: https://python-poetry.org
.. _matplotlib: https://matplotlib.org
.. _pyqtgraph: https://www.pyqtgraph.org
