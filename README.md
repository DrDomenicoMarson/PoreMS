<img src="https://github.com/porems/PoreMS/blob/main/docsrc/pics/logo_text_sub.svg" width="60%">

--------------------------------------

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/PoreMS/PoreMS/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14028652.svg)](https://doi.org/10.5281/zenodo.14028652)
[![Build Status](https://github.com/PoreMS/PoreMS/actions/workflows/workflow.yml/badge.svg)](https://github.com/PoreMS/PoreMS/actions/workflows/workflow.yml)
[![codecov](https://codecov.io/gh/PoreMS/PoreMS/branch/main/graph/badge.svg)](https://codecov.io/gh/PoreMS/PoreMS)

## Documentation

Online documentation is available at [porems.github.io/PoreMS](https://porems.github.io/PoreMS/).

<img src="https://github.com/porems/PoreMS/blob/main/docsrc/pics/pore.svg" width="60%">

The docs include an example for generating [molecules](https://porems.github.io/PoreMS/molecule.html) and [pores](https://porems.github.io/PoreMS/pore.html), and an [API reference](https://porems.github.io/PoreMS/api.html). Visit [process](https://porems.github.io/PoreMS/process.html) for an overview of the programs operating principle.

The [slit preparation guide](https://porems.github.io/PoreMS/slit.html) shows how to build silica pore systems, control surface chemistry, and export the resulting structure and topology files.

## Dependencies

PoreMS targets Python 3.14.

Installation requires [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/), [PyYAML](https://pyyaml.org/) and [tqdm](https://tqdm.github.io/).

## Bare and Grafted Amorphous Slit Workflows

PoreMS provides a dedicated high-level workflow for periodic amorphous silica
slits. The target is always given as `Q2/Q3/Q4/T2/T3` fractions over all Si
atoms in the sample. Bare slits therefore use `t2_fraction=0.0` and
`t3_fraction=0.0`, while functionalized slits can target both residual `Q`
states and grafted `T` states in one configuration.

### Bare slit preparation and export

```python
import porems as pms

config = pms.AmorphousSlitConfig(
    name="bare_amorphous_silica_slit",
    slit_width_nm=7.0,
    repeat_y=2,
    surface_target=pms.ExperimentalSiliconStateTarget(
        q2_fraction=66 / 40000,
        q3_fraction=650 / 40000,
    ),
)

result = pms.prepare_amorphous_slit_surface(config)
print(result.report.final_surface)
print(result.silica_topology.to_yaml())

result = pms.write_bare_amorphous_slit("output/bare_amorphous_slit", config)
print(result.bare_charge_diagnostics.is_neutral)
```

`ExperimentalSiliconStateTarget.q4_fraction` can be omitted. When omitted, it
is derived as the remaining fraction after `Q2`, `Q3`, `T2`, and `T3`.

`prepare_amorphous_slit_surface(...)` returns a `SlitPreparationResult` with an
attach-ready `PoreKit`, a structured `SlitPreparationReport`, and the resolved
editable silica topology model used by slit topology export. The finalized bare
writer produces a self-contained `<name>.itp` + `<name>.top` pair together with
`<name>.gro`, optional inspection-oriented `.pdb` / `.cif` files, `<name>.yml`,
and `<name>_report.json`.

### Inspecting or overriding the silica topology model

```python
import porems as pms

silica_model = pms.default_silica_topology()
print(silica_model.to_yaml())

silica_model.atom_assignments.silanol_oxygen.charge = "-0.750000"

config = pms.AmorphousSlitConfig(
    name="bare_with_custom_silica_model",
    silica_topology=silica_model,
)
```

`default_silica_topology()` returns a fresh editable copy each time. The same
model can be passed through `AmorphousSlitConfig.silica_topology` for both bare
and functionalized slit workflows.

### Functionalized (grafted) slit: coordinates only

```python
import porems as pms

slit_config = pms.AmorphousSlitConfig(
    name="functionalized_amorphous_silica_slit",
    slit_width_nm=7.0,
    repeat_y=1,
    surface_target=pms.ExperimentalSiliconStateTarget(
        q2_fraction=63 / 20000,
        q3_fraction=648 / 20000,
        t2_fraction=3 / 20000,
        t3_fraction=4 / 20000,
    ),
)

config = pms.FunctionalizedAmorphousSlitConfig(
    slit_config=slit_config,
    ligand=pms.SilaneAttachmentConfig(
        molecule=pms.gen.tms(),
        mount=0,
        axis=(0, 1),
    ),
    progress_settings=pms.FunctionalizedSlitProgressConfig(),
)

result = pms.prepare_functionalized_amorphous_slit_surface(config)
print(result.report.final_surface)

result = pms.write_functionalized_amorphous_slit(
    "output/functionalized_coordinates",
    config,
)
print(result.charge_diagnostics)
```

This coordinate-only path requires only the graft fragment coordinates. The
writer stores the finalized coordinates, YAML, and JSON report, but it does not
write functionalized slit `.top` / `.itp` files unless `SilaneTopologyConfig`
is supplied.

### Functionalized slit full-topology export

```python
from pathlib import Path
import porems as pms

slit_config = pms.AmorphousSlitConfig(
    name="functionalized_amorphous_silica_slit",
    slit_width_nm=7.0,
    repeat_y=1,
    surface_target=pms.ExperimentalSiliconStateTarget(
        q2_fraction=63 / 20000,
        q3_fraction=648 / 20000,
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

config = pms.FunctionalizedAmorphousSlitConfig(
    slit_config=slit_config,
    ligand=pms.SilaneAttachmentConfig(
        molecule=pms.gen.tms(),
        mount=0,
        axis=(0, 1),
        topology=topology,
    ),
)

result = pms.write_functionalized_amorphous_slit(
    "output/functionalized_full_topology",
    config,
)
print(result.charge_diagnostics.is_valid)
```

Required inputs for functionalized full-topology export:

- one self-contained flat GROMACS `.itp` file describing a base post-condensation `T3` fragment
- atom names in that `.itp` matching the atom names in the configured `Molecule`
- a base-fragment total charge matching the active silica model
- explicit `SilaneGeminalCrossTerms` whenever the target includes `T2` sites

Under the current default silica model, the expected base `T3` fragment charge
is `+0.825`. If you pass a custom `silica_topology`, that charge target can
change and should be checked against `default_silica_topology()` or your
modified model before export.

The current reference files `scripts/_top/tms.itp` and `scripts/_top/tmsg.itp`
are useful examples of the required bonded-term layout, but they should be
treated as parameter sources, not as automatically valid turnkey inputs for the
strict functionalized slit exporter. The file you pass through
`SilaneTopologyConfig.itp_path` must already satisfy the current naming and
charge contract for your chosen `Molecule` and silica model.


## Installation

Create a Python 3.14 environment, then install the repository from the
repository root in editable mode:

    pip install -r requirements.txt
    pip install -e .[test]

Use the repository directly in editable mode for local work.


## Testing

Run the test suite from the repository root after installing the test extra:

    pytest


## Development

PoreMS development takes place on Github: [www.github.com/porems/PoreMS](https://github.com/porems/PoreMS)

The current repository/documentation version is exposed as `porems.__version__`.

Please submit any reproducible bugs you encounter to the [issue tracker](https://github.com/porems/PoreMS/issues).


## How to Cite PoreMS

When citing PoreMS please use the following: **Kraus et al., Molecular Simulation, 2021, DOI: [10.1080/08927022.2020.1871478](https://doi.org/10.1080/08927022.2020.1871478)**

Additionaly, to assure reproducability of the generated pore systems, please cite the **Zenodo DOI** corresponding to the used PoreMS version. (Current DOI is listed in the badges.)

## Published Work
* Probst et al., 2025. Ring-Expansion Metathesis Polymerization under Confinement. Journal of the American Chemical Society, doi:[doi.org/10.1021/jacs.4c18171](https://doi.org/10.1021/jacs.4c18171)
  - Data-Repository: doi:[]()
* Högler et al., 2024. Influence of Ionic Liquid Film Thickness and Flow Rate on Macrocyclization Efficiency and Selectivity in Supported Ionic Liquid-Liquid Phase Catalysis. Chemistry – A European Journal, doi:[doi.org/10.1002/chem.202403237](https://doi.org/10.1002/chem.202403237)
  - Data-Repository: doi:[10.18419/DARUS-4063](https://doi.org/10.18419/DARUS-4063)
* Nguyen et al., 2024. Effects of Surfaces and Confinement on Formic Acid Dehydrogenation Catalyzed by an Immobilized Ru–H Complex: Insights from Molecular Simulation and Neutron Scattering. ACS Catalysis, doi:[doi.org/10.1021/acscatal.4c02626](https://doi.org/10.1021/acscatal.4c02626)
  - Data-Repository: doi:[doi.org/10.18419/DARUS-3584](https://doi.org/10.18419/DARUS-3584)
* Kraus et al., 2023. Axial Diffusion in Liquid-Saturated Cylindrical Silica Pore Models. The Journal of Physical Chemistry C, doi:[10.1021/acs.jpcc.3c01974](https://doi.org/10.1021/acs.jpcc.3c01974).
  - Data-Repository: doi:[10.18419/darus-3067](https://doi.org/10.18419/darus-3067)
* Kraus and Hansen, 2022. An atomistic view on the uptake of aromatic compounds by cyclodextrin immobilized on mesoporous silica. Adsorption, doi:[10.1007/s10450-022-00356-w](https://doi.org/10.1007/s10450-022-00356-w).
  - Data-Repository: doi:[10.18419/darus-2154](https://doi.org/10.18419/darus-2154)
* Kraus et al., 2021. PoreMS: a software tool for generating silica pore models with user-defined surface functionalisation and pore dimensions. Molecular Simulation, 47(4), pp.306-316, doi:[10.1080/08927022.2020.1871478](https://doi.org/10.1080/08927022.2020.1871478).
  - Data-Repository: doi:[10.18419/darus-1170](https://doi.org/10.18419/darus-1170)
* Ziegler et al., 2021. Confinement Effects for Efficient Macrocyclization Reactions with Supported Cationic Molybdenum Imido Alkylidene N-Heterocyclic Carbene Complexes. ACS Catalysis, 11(18), pp. 11570-11578, doi:[10.1021/acscatal.1c03057](https://doi.org/10.1021/acscatal.1c03057)
  - Data-Repository: doi:[10.18419/darus-1752](https://doi.org/10.18419/darus-1752)
* Kobayashi et al., 2021. Confined Ru-catalysts in a Two-phase Heptane/Ionic Liquid Solution: Modeling Aspects. ChemCatChem, 13(2), pp.739-746, doi:[10.1002/cctc.202001596](https://doi.org/10.1002/cctc.202001596).
  - Data-Repository: doi:[10.18419/darus-1138](https://doi.org/10.18419/darus-1138)
* Ziegler et al., 2019. Olefin Metathesis in Confined Geometries: A Biomimetic Approach toward Selective Macrocyclization. Journal of the American Chemical Society, 141(48), pp.19014-19022, doi:[10.1021/jacs.9b08776](https://doi.org/10.1021/jacs.9b08776).
  - Data-Repository: doi:[10.18419/darus-477](https://doi.org/10.18419/darus-477)
