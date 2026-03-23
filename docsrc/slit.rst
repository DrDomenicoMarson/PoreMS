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

PoreMS exposes a dedicated high-level API for periodic amorphous silica slit
workflows. The input target is always specified through experimental
``Q2/Q3/Q4/T2/T3`` fractions over all Si atoms in the sample. For bare slits,
``T2`` and ``T3`` remain zero.

.. code-block:: python

  import porems as pms

  config = pms.AmorphousSlitConfig(
      name="bare_amorphous_silica_slit",
      slit_width_nm=7.0,
      repeat_y=2,
      surface_target=pms.ExperimentalSiliconStateTarget(
          q2_fraction=66 / 40000,
          q3_fraction=650 / 40000,
          q4_fraction=1.0 - ((66 + 650) / 40000),
      ),
  )

  result = pms.prepare_amorphous_slit_surface(config)
  print(result.report.final_surface)
  print(result.silica_topology.to_yaml())

  result = pms.write_bare_amorphous_slit("output/bare_amorphous_slit", config)
  print(result.bare_charge_diagnostics.is_neutral)

``prepare_amorphous_slit_surface(...)`` returns a
``SlitPreparationResult`` containing an attach-ready ``PoreKit`` system and a
``SlitPreparationReport`` with the converted alpha-aware target, the prepared
bare surface, the final surface composition, and surface-preparation
diagnostics such as stripped silicon counts, removed orphan oxygens, inserted
bridge oxygens, and the final valid surface/scaffold oxygen counts. The same
result also exposes the resolved editable silica topology model used by slit
topology export.

``write_bare_amorphous_slit(...)`` finalizes the prepared slit and stores the
main structure files together with a self-contained full-slab ``.itp`` /
``.top`` pair, YAML metadata, and a JSON report. Object backups are written
only when ``write_object_files=True`` is requested explicitly. For inspection,
the same writer can also emit a ``.pdb`` file, and ``write_pdb_conect=True``
adds ``CONECT`` records for the assembled bond graph, including silica
scaffold bonds, siloxane bridges, ligand-internal bonds, and graft junctions.
For larger systems, ``write_cif=True`` writes an mmCIF file, and
``write_cif_bonds=True`` adds the same inspection-oriented connectivity via an
``_struct_conn`` bond loop.

The slit-preparation API is designed for the periodic bare-silica slit builder:

* zero exterior sites after preparation
* replicated amorphous template slabs along ``y``
* surface-state targeting through alpha-aware ``Q2/Q3/Q4/T2/T3`` fractions
* bare-slit exports written in one call through
  ``write_bare_amorphous_slit(...)``


Inspecting or Overriding the Active Silica Topology
---------------------------------------------------

The slit export path uses one resolved editable silica topology model for atom
types, partial charges, and bonded terms. The package defaults can be inspected
and overridden directly.

.. code-block:: python

  import porems as pms

  silica_model = pms.default_silica_topology()
  print(silica_model.to_yaml())

  silica_model.atom_assignments.silanol_oxygen.charge = "-0.750000"

  config = pms.AmorphousSlitConfig(
      name="bare_with_custom_silica_model",
      silica_topology=silica_model,
  )

``default_silica_topology()`` always returns a fresh editable copy, so local
changes never mutate the package defaults. The same model can be passed through
``AmorphousSlitConfig.silica_topology`` for both bare and functionalized slit
workflows.


Functionalized (Grafted) Amorphous Slit Preparation
---------------------------------------------------

Exact functionalized targets use the same experimental target object together
with a silane attachment definition. ``T2`` states are created from geminal
sites and ``T3`` states from single sites. In this slit path, siloxane bridges
introduced during Q-state editing are folded back into the silica scaffold as
regular framework oxygens instead of being exported as standalone ``SLX``
residues.

.. code-block:: python

  import porems as pms

  slit_config = pms.AmorphousSlitConfig(
      name="functionalized_amorphous_silica_slit",
      slit_width_nm=7.0,
      repeat_y=1,
      surface_target=pms.ExperimentalSiliconStateTarget(
          q2_fraction=63 / 20000,
          q3_fraction=648 / 20000,
          q4_fraction=1.0 - ((63 + 648 + 3 + 4) / 20000),
          t2_fraction=3 / 20000,
          t3_fraction=4 / 20000,
      ),
  )

  functionalized = pms.FunctionalizedAmorphousSlitConfig(
      slit_config=slit_config,
      ligand=pms.SilaneAttachmentConfig(
          molecule=pms.gen.tms(),
          mount=0,
          axis=(0, 1),
      ),
      progress_settings=pms.FunctionalizedSlitProgressConfig(),
  )

  result = pms.prepare_functionalized_amorphous_slit_surface(functionalized)
  print(result.report.final_surface)

  result = pms.write_functionalized_amorphous_slit(
      "output/functionalized_coordinates",
      functionalized,
  )
  print(result.charge_diagnostics)

``FunctionalizedSlitProgressConfig`` enables built-in ``tqdm`` progress bars
for the exact functionalized slit workflow. Auto mode shows progress in
interactive terminals and notebooks while staying quiet in typical non-
interactive test and batch contexts. Functionalized coordinate export works
with just the ligand coordinates. In that case the writer stores coordinates,
YAML metadata, and the JSON report, but it skips functionalized slit
``.top`` / ``.itp`` files because no explicit flat topology bundle was given.


Required Inputs for Functionalized Full Topology Export
-------------------------------------------------------

Self-contained functionalized slit topology export requires
``SilaneTopologyConfig`` on the attachment definition, including for TMS.

The supplied flat ITP is interpreted as one base post-condensation ``T3``
fragment. In practical PoreMS terms, the exporter expects:

* one self-contained flat GROMACS ``.itp`` file for the base ``T3`` fragment
* atom names in that file matching the atom names in the configured
  ``Molecule``
* a base-fragment total charge matching the active silica topology model
* explicit ``SilaneGeminalCrossTerms`` whenever the requested surface includes
  ``T2`` sites

Under the current default silica model, the expected base ``T3`` fragment
charge is ``+0.96``. If you override ``AmorphousSlitConfig.silica_topology``,
the expected charge target may change and should be checked against the active
model before export.

The current reference files ``scripts/_top/tms.itp`` and
``scripts/_top/tmsg.itp`` are useful examples of the required bonded-term
layout, but they should be treated as parameter sources, not as automatically
valid turnkey inputs for the strict functionalized slit exporter. The actual
``SilaneTopologyConfig.itp_path`` file must already satisfy the naming and
charge contract for the chosen ``Molecule`` and silica model.


Functionalized Full-Topology Export Example
-------------------------------------------

The following example shows the extra inputs required when the functionalized
slit should also write one self-contained ``.top`` / ``.itp`` pair.

.. code-block:: python

  from pathlib import Path
  import porems as pms

  slit_config = pms.AmorphousSlitConfig(
      name="functionalized_amorphous_silica_slit",
      slit_width_nm=7.0,
      repeat_y=1,
      surface_target=pms.ExperimentalSiliconStateTarget(
          q2_fraction=63 / 20000,
          q3_fraction=648 / 20000,
          q4_fraction=1.0 - ((63 + 648 + 3 + 4) / 20000),
          t2_fraction=3 / 20000,
          t3_fraction=4 / 20000,
      ),
  )

  topology = pms.SilaneTopologyConfig(
      itp_path=str(Path("path/to/tms_base_t3.itp")),
      moleculetype_name="TMS",
      geminal_cross_terms=pms.SilaneGeminalCrossTerms(
          first_ligand_atom_name="O1",
          geminal_oxygen_mount_ligand_angle=pms.GromacsAngleParameters.harmonic(
              angle_deg=105.56,
              force_constant=384.223760,
          ),
          geminal_dihedrals=(
              pms.GeminalMountDihedralSpec(
                  fourth_atom_name="Si2",
                  function=1,
                  parameters=("0.00000", "1.60387", "3"),
              ),
          ),
      ),
  )

  functionalized = pms.FunctionalizedAmorphousSlitConfig(
      slit_config=slit_config,
      ligand=pms.SilaneAttachmentConfig(
          molecule=pms.gen.tms(),
          mount=0,
          axis=(0, 1),
          topology=topology,
      ),
      progress_settings=pms.FunctionalizedSlitProgressConfig(),
  )

  result = pms.write_functionalized_amorphous_slit(
      "output/functionalized_full_topology",
      functionalized,
  )
  print(result.charge_diagnostics.is_valid)

The numerical geminal angle and dihedral terms above are example values derived
from the current repository-side TMS/TMSG reference data. If your grafted
fragment uses different atom names or different bonded terms, update
``SilaneGeminalCrossTerms`` accordingly. The exporter will not infer those
cross terms from the coordinate fragment.

Stored outputs
--------------

The ``store()`` helpers write the generated structure together with the
companion files needed to keep the silica model reproducible:

* structure files such as ``.gro`` for coordinates and periodic box lengths
* slit exports: self-contained ``.top`` / ``.itp`` files for the finalized slit
* generic pore exports: legacy helper ``.top`` and ``grid.itp`` files
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
