Github Actions status:
[![pytest](https://github.com/saullocastro/bfsccylinder/actions/workflows/pytest.yml/badge.svg)](https://github.com/saullocastro/bfsccylinder/actions/workflows/pytest.yml)
[![Deploy](https://github.com/saullocastro/bfsccylinder/actions/workflows/pythonpublish.yml/badge.svg)](https://github.com/saullocastro/bfsccylinder/actions/workflows/pythonpublish.yml)

Coverage status:
[![codecov](https://github.com/saullocastro/bfsccylinder/actions/workflows/coverage.yml/badge.svg)](https://github.com/saullocastro/bfsccylinder/actions/workflows/coverage.yml)
[![Codecov Status](https://codecov.io/gh/saullocastro/bfsccylinder/branch/main/graph/badge.svg?token=KD9D8G8D2P)](https://codecov.io/gh/saullocastro/bfsccylinder)

Implementation of the BFSC cylinder finite element
==================================================

- Single-curvature BFSC (SC-BFSC) element: a four-node element of the
  developed mid-surface of a cylindrical shell
- Donnell-type kinematics, `BFSCCylinder`, and Sanders-Koiter type
  kinematics, `BFSCCylinderSanders`
- Linear statics: [KC0]{u} = {f}
- Linear buckling: ([KC0] + lambda[KG]){u} = 0, with [KG] of the membrane
  stress resultants of a static solution or of a constant membrane stress
  state
- Natural frequencies: ([KC0] - omega^2[M]){u} = 0
- Geometrically nonlinear statics: internal force vector and tangent
  stiffness matrix [KC0] + [KCNL] + [KG], its exact Jacobian
- Follower pressure, normal to the deformed shell: force vector and
  unsymmetric load stiffness matrix
- The contributions of each element are written in sparse arrays; the model,
  its boundary conditions and the solution of its equations are written by
  the user, with NumPy and SciPy

The ABD matrices of the laminates are calculated with my other repository
[composites](https://github.com/saullocastro/composites), the tests and
usage examples solve the nonlinear analyses with
[structsolve](https://github.com/saullocastro/structsolve), and the
tutorials model the stringers of stiffened cylinders with
[bfscplate2d](https://github.com/saullocastro/bfscplate2d).


Citing this library
===================

Saullo G. P. Castro (2026). Implementation of the BFSC cylinder finite element (Version 0.7.0). https://github.com/saullocastro/bfsccylinder.


Documentation
=============

The documentation is available on: https://saullocastro.github.io/bfsccylinder.


History
=======

See [CHANGELOG.md](CHANGELOG.md) for the details of each version.

* version 0.7.0 (2026-10-26)
    - Follower pressure, for both kinematics: `update_fext_follower` and
      `update_KCfollower`, the force vector and the unsymmetric load
      stiffness matrix of a pressure normal to the deformed mid-surface
    - BUG: `update_M` of the Sanders-type element uses the rotation
      `-w,y + v/R` in the rotary inertia
    - BUG: the constructors of `BFSCCylinder` and `BFSCCylinderSanders` set
      `init_k_KCNL`, which a second call of a constructor left at its
      previous value
    - Deterministic and verified natural frequency tests
    - Sphinx documentation, with a user guide, usage examples, the solution
      of the analyses with `structsolve.Analysis` and tutorials of
      stringer-stiffened cylinders
    - The tests solve the nonlinear analyses with `structsolve.Analysis`,
      requiring `structsolve>=0.9.3`
    - Metadata in `pyproject.toml`; Python 3.10 to 3.14
* version 0.6.1 (2026-09-25)
    - Cython compiler flags
* version 0.6.0 (2026-09-11)
    - Fixed the tangent stiffness matrix, now the derivative of the internal
      force vector, for both kinematics
    - Fixed `update_KC0` of the Sanders-type element, which missed a part of
      the twisting curvature
* version 0.5.1 (2026-08-27), including 0.5.0
    - Fixed the bending part of the Sanders-type equations
    - Shape functions renamed from `N` to `S`
    - Python 3.14 support
* versions 0.4.1 to 0.4.8 (2024-09-25 to 2025-11-25)
    - Fixed the compatibility with integer numbers; updated continuous
      integration and deployment
* version 0.4.0 (2023-04-12)
    - Removed the dependency on the NumPy C API
* version 0.3.12 (2021-12-04), including 0.3.11
    - Fixed the nonlinear terms, where `v` was missing, and `fint`, which
      included terms of `KG`
* version 0.3.0 (2021-11-03)
    - Sanders-type kinematics: `BFSCCylinderSanders`, in
      `bfsccylinder.sanders`
* version 0.2.3 (2021-10-04), including 0.2.2
    - 3-Clause BSD license
* version 0.2.0 (2021-05-26)
    - Geometrically nonlinear analyses: `update_KCNL` and `update_fint`
* version 0.1.1 (2020-08-22)
    - First tagged version: `BFSCCylinder`, with `update_KC0`, `update_KG`,
      `update_KG_constant_stress` and `update_M`


License
=======

Distributed under the 3-Clause BSD license
(https://raw.github.com/saullocastro/bfsccylinder/main/LICENSE).

Contact: S.G.P.Castro@tudelft.nl
