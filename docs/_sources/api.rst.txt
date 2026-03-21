:orphan:

API reference
=============

.. currentmodule:: porems


.. _generation_api:

Molecule
--------

.. autosummary::
    :toctree: generated/

    molecule.Molecule
    atom.Atom
    store.Store
    connectivity.GraphBond
    connectivity.GraphAngle
    connectivity.AttachmentRecord
    connectivity.AssembledStructureGraph


.. _pore_api:

Pore
----

.. autosummary::
    :toctree: generated/

    pore.BindingSite
    pore.SurfacePreparationDiagnostics
    pore.Pore
    system.ShapeAttachmentSummary
    system.RoughnessProfile
    system.SurfaceAreaSummary
    system.SurfaceAllocationStats
    system.AllocationSummary
    system.PoreKit
    system.PoreCylinder
    system.PoreSlit
    system.PoreCapsule
    system.PoreAmorphCylinder


.. _pattern_api:

Pattern
~~~~~~~

.. autosummary::
    :toctree: generated/

    pattern.Pattern
    pattern.AlphaCristobalit
    pattern.BetaCristobalit


.. _shape_api:

Shape
~~~~~

.. autosummary::
    :toctree: generated/

    shape.ShapeConfig
    shape.ShapeSection
    shape.ShapeSpec
    shape.CylinderConfig
    shape.SphereConfig
    shape.CuboidConfig
    shape.ConeConfig
    shape.Shape
    shape.Cylinder
    shape.Sphere
    shape.Cuboid
    shape.Cone


.. _optimization_api:

Optimization
~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/

    dice.Dice
    matrix.Matrix


.. _slit_api:

Slit Preparation
----------------

.. autosummary::
    :toctree: generated/

    slit.SiliconStateFractions
    slit.ExperimentalSiliconStateTarget
    slit.AmorphousSlitConfig
    slit.SiliconStateComposition
    slit.SlitPreparationReport
    slit.SlitPreparationResult
    slit.SilaneAttachmentConfig
    slit.FunctionalizedAmorphousSlitConfig
    slit.FunctionalizedSlitResult
    slit.prepare_amorphous_slit_surface
    slit.write_bare_amorphous_slit
    slit.prepare_functionalized_amorphous_slit_surface
    slit.write_functionalized_amorphous_slit


.. _utils_api:

Utilities
---------

.. autosummary::
    :toctree: generated/

    utils
    generic
    geometry
    database
