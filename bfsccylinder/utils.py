"""
Utilities (:mod:`bfsccylinder.utils`)

.. currentmodule:: bfsccylinder.utils

The function :func:`.assign_constant_ABD` copies the stiffnesses of a
laminate to every integration point of an element.

"""
def assign_constant_ABD(bfsccylinder, prop):
    """Assign the same laminate stiffnesses to all integration points

    Every entry of the arrays ``A11, A12, A16, A22, A26, A66``, ``B11, B12,
    B16, B22, B26, B66`` and ``D11, D12, D16, D22, D26, D66`` of the element,
    one entry per integration point, is set to the attribute with the same
    name of ``prop``. The arrays are changed in place. Since the ``update_*``
    functions read these arrays at each integration point, they can also be
    set point by point, for stiffnesses that vary over the element.

    Parameters
    ----------
    bfsccylinder : :class:`.BFSCCylinder` or :class:`.BFSCCylinderSanders`
        Element whose stiffness arrays, of shape ``(nint, nint)``, are
        updated.
    prop : object
        Object with the float attributes ``A11, A12, A16, A22, A26, A66``
        (force per unit length), ``B11, B12, B16, B22, B26, B66`` (force) and
        ``D11, D12, D16, D22, D26, D66`` (force times length), e.g. the
        :class:`composites.core.Laminate` returned by
        :func:`composites.utils.laminated_plate` or
        :func:`composites.utils.isotropic_plate`.

    """
    bfsccylinder.A11[:] = prop.A11
    bfsccylinder.A12[:] = prop.A12
    bfsccylinder.A16[:] = prop.A16
    bfsccylinder.A22[:] = prop.A22
    bfsccylinder.A26[:] = prop.A26
    bfsccylinder.A66[:] = prop.A66
    bfsccylinder.B11[:] = prop.B11
    bfsccylinder.B12[:] = prop.B12
    bfsccylinder.B16[:] = prop.B16
    bfsccylinder.B22[:] = prop.B22
    bfsccylinder.B26[:] = prop.B26
    bfsccylinder.B66[:] = prop.B66
    bfsccylinder.D11[:] = prop.D11
    bfsccylinder.D12[:] = prop.D12
    bfsccylinder.D16[:] = prop.D16
    bfsccylinder.D22[:] = prop.D22
    bfsccylinder.D26[:] = prop.D26
    bfsccylinder.D66[:] = prop.D66

