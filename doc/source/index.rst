BFSC cylinder finite element - bfsccylinder
===========================================

The ``bfsccylinder`` module implements the single-curvature BFSC (SC-BFSC)
finite element, a four-node element of the developed mid-surface of a
cylindrical shell, for linear and nonlinear analyses, with Donnell-type
kinematics, :class:`bfsccylinder.bfsccylinder.BFSCCylinder`, and
Sanders-type kinematics,
:class:`bfsccylinder.bfsccylinder_sanders.BFSCCylinderSanders`. The module
gives the elements and the functions that compute the contributions of one
element to the matrices and vectors of a model, written in sparse arrays;
the model, its boundary conditions and the solution of its equations are
written by the user, with NumPy and SciPy. With ``bfsccylinder`` you can
compute:

* The linear constitutive stiffness matrix `\boldsymbol{K}_{C0}`, for linear static
  analyses, `\boldsymbol{K}_{C0}\boldsymbol{u} = \boldsymbol{F}_{ext}`

* The geometric stiffness matrix `\boldsymbol{K}_G`, for linear buckling analyses,
  `(\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G)\boldsymbol{u} = \boldsymbol{0}`, of the membrane stress
  resultants of a static solution or of a constant membrane stress state

* The mass matrix `\boldsymbol{M}`, for natural frequency analyses,
  `(\boldsymbol{K}_{C0} - \omega^2 \boldsymbol{M})\boldsymbol{u} = \boldsymbol{0}`

* The internal force vector `\boldsymbol{F}_{int}` and the tangent stiffness matrix
  `\boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL} + \boldsymbol{K}_G`, its exact Jacobian, for geometrically
  nonlinear static analyses

* The force vector and the load stiffness matrix of a follower pressure,
  normal to the deformed shell


Code repository
---------------

https://github.com/saullocastro/bfsccylinder


User guide
----------

.. toctree::
    :maxdepth: 1

    element.rst
    kinematics.rst
    assembling.rst
    analyses.rst
    follower_pressure.rst


Usage examples
--------------

.. toctree::
    :maxdepth: 1

    ex_linear_static.rst
    ex_linear_buckling.rst
    ex_natural_frequency.rst
    ex_nonlinear_static.rst
    ex_structsolve_nonlinear.rst
    ex_follower_pressure.rst
    ex_kinematics.rst


Tutorials
---------

.. toctree::
    :maxdepth: 1

    tutorials.rst


bfsccylinder API
----------------

.. toctree::
    :maxdepth: 2

    api.rst


Installing bfsccylinder
-----------------------

.. toctree::
    :maxdepth: 1

    install.rst


Changelog
---------

.. toctree::
    :maxdepth: 1

    changelog.rst


License
-------

.. literalinclude:: ../../LICENSE
    :encoding: latin-1


Citing this library
-------------------

Saullo G. P. Castro (2026). Implementation of the BFSC cylinder finite element (Version 0.7.0). https://github.com/saullocastro/bfsccylinder.

In BibTeX:

.. code-block:: bibtex

    @misc{castro_bfsccylinder,
      author  = {Castro, Saullo G. P.},
      title   = {Implementation of the BFSC cylinder finite element},
      year    = {2026},
      version = {0.7.0},
      url     = {https://github.com/saullocastro/bfsccylinder}
    }

The repository also has a ``CITATION.cff`` file with the same data.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
