# Changelog

The dates of the released versions are those of their tagged commits.

## Unreleased

### Python versions

- Python 3.9 is no longer supported, it reached its end of life in October 2025: Python 3.10 or newer is required, and the tests and wheels cover Python 3.10 to 3.14.

### New: follower pressure

- `update_fext_follower` and `update_KCfollower`, for both kinematics, with
  the module constants `KCfollower_SPARSE_SIZE = 1312` (Donnell-type) and
  `1456` (Sanders-type) and the element attribute `init_k_KCfollower`: the
  force vector and the unsymmetric load stiffness matrix of a pressure
  normal to the deformed mid-surface, with the area vector truncated to
  first order (`quadratic=0`) or complete (`quadratic=1`), and the dead
  part optional (`reference`). The kernels are written by
  `deriving_equations/*/derive_bfsccylinder_follower.py` and kept in sync
  by `regenerate.py`. New `tests/test_follower_pressure.py` and
  `tests/test_follower_pressure_validation.py`, with rings and finite
  cylinders against closed forms.
- The Notes of `bfsccylinder.bfsccylinder` state the conditions under which
  the load stiffness matrix of the first-order area vector is symmetric,
  for the element matrix and for the assembled matrix.

### Bug fixes

- `update_M` of the Sanders-type element uses the rotation `-w,y + v/R` of
  the Sanders-type kinematics in the rotary inertia, instead of the
  Donnell-type rotation `-w,y`. The mass matrix now reads `R` and couples
  `v` and `w`: `M_SPARSE_SIZE` of the Sanders-type module grows from 544 to
  920. The Donnell-type element is unchanged. New test of the mass matrix
  against an independent quadrature of the kinetic energy.
- The constructors of `BFSCCylinder` and `BFSCCylinderSanders` set
  `init_k_KCNL`, which a second call of a constructor left at its previous
  value. New test that the constructors set every `init_k_*` to zero.
- The natural frequency tests are deterministic and verified: the standard
  problem of `([K] + [M])^-1 [M]` is solved from a fixed starting vector,
  with the residual of each eigenpair checked and, by Sylvester's law of
  inertia, no lower eigenvalue missing, see
  `tests/natural_frequency_solver.py`. Before, `eigsh` in shift-invert mode
  started ARPACK from a random vector and failed in about 40 % of the runs.
  The reference values are unchanged.

### Documentation

- D1: functional docstrings of the modules, of `BFSCCylinder` and
  `BFSCCylinderSanders` and of every public function.
- D2: Notes and References of the theory: Hermite interpolation of the
  single-curvature BFSC element, the Donnell-type and Sanders-type
  kinematics, the element matrices, the internal forces and the
  quadrature.
- D3: Sphinx documentation in `doc/`, with a user guide (the element, the
  kinematics, assembling a model, the analyses, the follower pressure) and
  usage examples `ex_*.rst` with the code of the tests.
- Solving with structsolve: the callables of `structsolve.Analysis` for an
  element loop, the boundary conditions, the follower pressure and the
  convergence criterion, in `analyses.rst` and
  `ex_structsolve_nonlinear.rst`.
- Tutorials, Jupyter notebooks in `doc/source/tutorials/` stored with their
  outputs: natural frequencies (Sewall and Naumann, NASA TN D-4705, 1968)
  and axial buckling (Card and Jones, NASA TN D-3639, 1966) of
  stringer-stiffened cylinders, with the skin of `BFSCCylinderSanders`
  elements and blade stringers of the BFSC plate element of `bfscplate2d`,
  compared with the published tests and theories.

### Tests

- `structsolve>=0.9.0`, with its module `structsolve.convergence`, is a
  requirement of the tests, listed in
  `requirements.txt` and not in `install_requires`.
- `tests/test_nonlinear_axial_compression.py` solves the non-linear static
  analysis with `structsolve.Analysis` instead of hand-written
  Newton-Raphson iterations.
- New `tests/test_follower_pressure_structsolve.py`: the follower pressure
  on a cylinder with a free end with `structsolve.Analysis`, for both
  kinematics.

## 0.6.1 (2026-09-25)

- Cython compiler flags.

## 0.6.0 (2026-09-11)

- Fixed the tangent stiffness matrix, now the derivative of the internal
  force vector, for both kinematics: `fint` counted the quadratic membrane
  strain twice, and the geometric term of the stress of the quadratic
  membrane strain is added to `update_KCNL`, such that `update_KG` remains
  homogeneous of degree one in the displacements. `update_KC0` of the
  Sanders-type element had been missing the `-Su_y/(2R)` part of the
  twisting curvature.
- `regenerate.py` fails instead of skipping a missing derivation script,
  target function or block.
- README: the kinematics description includes the Sanders-Koiter type.

## 0.5.1 (2026-08-27)

- Continuous integration with Python 3.14.

## 0.5.0 (2026-08-27)

- Fixed the derivation and the implementation of the bending part of the
  Sanders-type equations.
- The shape functions renamed from `N` to `S`.
- Derivation results added; improved `eigsh` setup of the tests; updated
  actions and support of Python 3.14.

## 0.4.8 (2025-11-25)

- Updated Python version and improved deployment.

## 0.4.7 (2024-09-25)

- Fixed tolerances in the test results.

## 0.4.6 (2024-09-25)

- Updated parameters of the tests.

## 0.4.5 (2024-09-25)

- Fixed the compatibility with integer numbers.

## 0.4.3 (2024-09-25)

- Fixed `atol` of the tests.

## 0.4.2 (2024-09-25)

- Fixed the compatibility with integer numbers.

## 0.4.1 (2024-09-25)

- Updates of the continuous integration and of the deployment workflow.

## 0.4.0 (2023-04-12)

- Removed the dependency on the NumPy C API, improving the compatibility.
- Fixed an issue with the integer types `long` and `long long`, and an
  import error.
- Updated workflows; the tests use `INT`, `DOUBLE` and `DOF`.

## 0.3.12 (2021-12-04)

- Readable definition of `fint`, without change in the results.

## 0.3.11 (2021-12-04)

- Fixed the non-linear terms, where `v` was missing, and `fint`, which
  included terms of `KG`.

## 0.3.0 (2021-11-03)

- Sanders-type kinematics: the module `bfsccylinder.bfsccylinder_sanders`
  with the element `BFSCCylinderSanders`, accessed through
  `bfsccylinder.sanders`, with tests.
- Reorganized folders.

## 0.2.3 (2021-10-04)

- Release of 0.2.2.

## 0.2.2 (2021-10-04)

- License from the 2-clause BSD license to the 3-clause BSD license.
- Fixed the load of the non-linear test; faster coverage compilation;
  README integrated with PyPI.

## 0.2.0 (2021-05-26)

- Geometrically non-linear analyses: `update_KCNL` and `update_fint`.
- Improved call of `assign_constant_ABD`.

## 0.1.5 (2021-03-18)

- Faster test and coverage.

## 0.1.4 (2021-03-18)

- Updated badges.

## 0.1.3 (2021-03-18)

- New version of composites; better automated deployment.

## 0.1.1 (2020-08-22)

- First tagged version, after the commit `REL: Version 0.1.0` of
  2020-08-21: the BFSC cylinder element `BFSCCylinder`, with `update_KC0`,
  `update_KG`, with a fix of its calculation, `update_KG_constant_stress`,
  new after 0.1.0, and `update_M`, the Gauss-Legendre points and weights of
  `bfsccylinder.quadrature` and `assign_constant_ABD` of
  `bfsccylinder.utils`.
