Tutorials
=========

The tutorials are Jupyter notebooks, stored in ``doc/source/tutorials/`` with
their outputs, which compare models of stringer-stiffened cylinders with
published tests: the skin is modelled with the elements of bfsccylinder and
each blade stringer with the BFSC plate element of `bfscplate2d
<https://github.com/saullocastro/bfscplate2d>`_. Both notebooks build their
models with the module ``stiffened_cylinder.py`` of the same folder, below.
Running them requires bfscplate2d 0.5.0 or newer, with the shape functions
named ``Su``, ..., ``Sw_y``, `composites
<https://github.com/saullocastro/composites>`_, `structsolve
<https://saullocastro.github.io/structsolve/>`_ and Matplotlib, besides
bfsccylinder.

.. toctree::
    :maxdepth: 1

    tutorials/stiffened_cylinder_frequencies.ipynb
    tutorials/stiffened_cylinder_buckling.ipynb
    tutorials/simo1990_cylinder_postbuckling.ipynb


Model of the stiffened cylinders
--------------------------------

The skin and the stringers are assembled as in :doc:`assembling`, with the
degrees of freedom of the stringer nodes along the skin eliminated by the
constraint basis `\boldsymbol{Z}` described in the module docstring:

.. literalinclude:: tutorials/stiffened_cylinder.py


Model of the post-buckling of a cylinder
----------------------------------------

The tutorial :doc:`tutorials/simo1990_cylinder_postbuckling` traces with
the Riks method of structsolve 0.10.0 or newer the post-buckling path of
the axially compressed cylinder of Simo, Fox and Rifai (1990), an
``L/10 x 15 deg`` section meshed with the Donnell-type element, with an
initial geometric imperfection added exactly. It requires structsolve,
composites and Matplotlib besides bfsccylinder, and builds its model with
the module ``simo1990_cylinder.py`` of the same folder, also used by
``tests/test_simo1990_cylinder.py``:

.. literalinclude:: tutorials/simo1990_cylinder.py
