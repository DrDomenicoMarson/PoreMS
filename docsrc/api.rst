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


.. _pore_api:

Pore
----

.. autosummary::
    :toctree: generated/

    pore.BindingSite
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

    dice.SearchExecution
    dice.SearchPolicy
    dice.Dice
    matrix.Matrix


.. _workflow_api:

Workflow
--------

.. autosummary::
    :toctree: generated/

    workflows.SurfaceCompositionTarget
    workflows.BareAmorphousSlitConfig
    workflows.SurfaceComposition
    workflows.SlitBuildReport
    workflows.SlitBuildResult
    workflows.build_periodic_amorphous_slit
    workflows.write_bare_amorphous_slit_study


.. _utils_api:

Utilities
---------

.. autosummary::
    :toctree: generated/

    utils
    generic
    geometry
    database
