How the documentation is organised
==================================

The documentation of ``bfsccylinder`` is written in four cumulative levels,
each with its own place, so that a reader, or a tool, can go from what a
function does to why it does it without the levels repeating each other.

.. list-table::
    :header-rows: 1

    * - Level
      - Where
      - Content
    * - 1, functional
      - docstrings of the modules, classes, functions and methods, except
        their ``Notes`` and ``References`` sections; comments in the code;
        docstrings and comments of the tests; the ``README.md``
      - what each resource does: a one-line summary, an extended summary
        (what, not why), ``Parameters`` with type, unit, shape and
        default, ``Returns``, ``Raises`` and ``See Also``, and short usage
        ``Examples`` for the main entry points (the element classes, the
        assembly functions and the integration points)
    * - 2, theory
      - the ``Notes`` and ``References`` sections of the docstrings, of the
        library and of the tests
      - the theory that each resource implements or checks, local to it:
        the equations in the notation of the code, the assumptions and
        limits of validity, and the published sources
    * - 3, overview
      - the pages of the :ref:`user guide <user-guide>`, from
        :doc:`kinematics` to :doc:`conventions`
      - how the pieces of level 2 fit together, like the chapters of a
        book: the kinematics, the constitutive relations, the element, its
        matrices, the follower pressure, the assembly of a model and the
        analyses, with links to the API instead of repeating level 2
    * - 4, worked examples
      - the notebooks in ``notebooks/`` at the root of the repository,
        rendered in the sections "Worked examples" and "Tutorials"
      - complete analyses with stored outputs, each step explained, the
        results printed and plotted, and compared in tables with their
        reference values

The API pages render levels 1 and 2 together. To find the theory of a
resource, read the ``Notes`` of its docstring; to find how a function fits
in an analysis, read the pages of level 3, starting from :doc:`assembling`
and :doc:`analyses`.


The tests
---------

The tests in ``tests/`` belong to levels 1 and 2 like the library: the
docstring of each test says what it verifies and against what, naming the
source, and its ``Notes`` and ``References`` give the theory and the full
references. A reference value is taken from the literature wherever
possible, e.g. the buckling load of the cylinder Z33 of Geier et al. (2002);
a value stored from a run of the element is called a regression value, and
the ``Notes`` then compare it with the closest published result. The
modules of ``tests/`` that are not tests, such as ``simo1990_cylinder.py``,
hold models shared by the tests and the notebooks.
