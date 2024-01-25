Installation
============

Installation Users
------------------

SymMePlot is not yet available on PyPI. Therefore, you’ll need to install the
development version from GitHub using: ::

    pip install git+https://github.com/TJStienstra/symmeplot.git

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

To install everything at once, run: ::

    poetry install --with lint,test,docs

To quickly check code for linting errors, it is recommended to set up ``pre-commit``
hooks by executing: ::

    pip install pre-commit
    pre-commit install

.. _poetry: https://python-poetry.org
