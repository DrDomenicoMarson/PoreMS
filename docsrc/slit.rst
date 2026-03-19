:orphan:

.. raw:: html

  <div class="container-fluid">
    <div class="row">
      <div class="col-md-10">
        <div style="text-align: justify; text-justify: inter-word;">

Slit Preparation Guide
======================

This page collects build-oriented examples for creating silica pore models and
surface functionalizations with PoreMS. Downstream simulation protocols are
left to the user so that the package documentation stays focused on structure
generation and surface chemistry.


Bare Amorphous Slit Preparation
-------------------------------

PoreMS exposes a high-level API for preparing a periodic bare amorphous silica
slit together with a structured surface report.

.. code-block:: python

  import porems as pms

  config = pms.AmorphousSlitConfig(
      name="bare_amorphous_silica_slit",
      slit_width_nm=7.0,
      repeat_y=2,
  )

  result = pms.prepare_amorphous_slit_surface(config)
  print(result.report.prepared_surface)

  pms.write_bare_amorphous_slit("output/bare_amorphous_slit", config)

``prepare_amorphous_slit_surface(...)`` returns a
``SlitPreparationResult`` containing an attach-ready ``PoreKit`` system and a
``SlitPreparationReport`` with the prepared surface composition and siloxane
summary. ``write_bare_amorphous_slit(...)`` finalizes the prepared slit and
stores the main structure files together with a JSON report. Object backups are
written only when ``write_object_files=True`` is requested explicitly.

The slit-preparation API is designed for the periodic bare-silica slit builder:

* zero exterior sites after preparation
* replicated amorphous template slabs along ``y``
* surface-state targeting through ``Q2/Q3/Q4`` fractions
* bare-slit exports written in one call through
  ``write_bare_amorphous_slit(...)``


Create surface molecules
------------------------

Trimethylsilyl or for short TMS is a simple surface group that can be imported
from the PoreMS package. Assuming a new surface group structure is to be created,
following code block can be used as a base.

.. code-block:: python

  import porems as pms

  tms = pms.Molecule("tms", "TMS")
  tms.set_charge(0.96)
  compress = 30

  b = {"sio": 0.155, "sic": 0.186, "ch": 0.109}
  a = {"ccc": 30.00, "cch": 109.47}

  # Create tail
  tms.add("Si", [0, 0, 0])
  tms.add("O", 0, r=b["sio"])
  tms.add("Si", 1, bond=[1, 0], r=b["sio"], theta=180)

  # Add carbons
  for i in range(3):
      tms.add("C", 2, bond=[2, 1], r=b["sic"], theta=a["cch"]+compress, phi=60+120*i)

  # Add hydrogens
  for i in range(3, 5+1):
      for j in range(3):
          tms.add("H", i, bond=[i, 2], r=b["ch"], theta=a["cch"]+compress, phi=60+120*j)

.. figure::  /pics/flow/tms.png
 :align: center
 :width: 30%
 :name: fig1

.. note::

  Parametrization must be carried out by the user. Topology generation should
  be performed for both a singular binding site and a geminal binding site.


Search execution policy
-----------------------

Connectivity and proximity searches in PoreMS are controlled through
``Dice.find(...)`` and an explicit execution policy.

.. code-block:: python

  import porems as pms

  block = pms.BetaCristobalit().generate([2, 2, 2], "z")
  dice = pms.Dice(block, 0.2, True)

  policy = pms.SearchPolicy(
      execution=pms.SearchExecution.AUTO,
      processes=4,
  )
  bonds = dice.find(None, ["Si", "O"], [0.145, 0.165], policy=policy)

``AUTO`` uses multiprocessing only when the current ``__main__`` module can be
re-imported safely by worker processes. Otherwise PoreMS falls back to serial
execution and emits a Python warning. If ``PROCESSES`` is requested
explicitly, the caller must use a file-backed ``__main__`` entrypoint such as a
normal script or ``python -m unittest ...``.


Create pore system
------------------

Next step is to create a pore structure functionalized with the created TMS
surface group.

.. code-block:: python

  import porems as pms

  pore = pms.PoreCylinder([10, 10, 10], 6, 5.5)

  pore.attach(pms.gen.tms(), 0, [0, 1], 100, "in")
  pore.attach(pms.gen.tms(), 0, [0, 1], 100, "ex")

  pore.finalize()

.. figure::  /pics/flow/pore.png
 :align: center
 :width: 50%
 :name: fig2

Once the generation is done, store the structure. If you also want Python-side
backup objects for later inspection, request them explicitly. The same helper
also writes a master topology with the number of residues and a topology
containing grid molecule parameters.

.. code-block:: python

    pore.store()

Stored outputs
--------------

The ``store()`` helpers write the generated structure together with the
companion files needed to keep the silica model reproducible:

* structure files such as ``.gro`` for coordinates and periodic box lengths
* topology helpers such as ``.top`` and ``grid.itp`` for the silica scaffold
* YAML or JSON metadata summaries for the generated geometry and surface state

Serialized ``.obj`` backups remain available as an explicit opt-in through
``write_object_files=True`` when calling the storage helpers.

PoreMS intentionally stops at generating the silica model and its companion
build artifacts. Any downstream simulation setup should be defined separately
by the user for the specific force field and engine they choose.


.. raw:: html

        </div>
      </div>
    </div>
  </div>
