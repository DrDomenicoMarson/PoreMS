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

Installation requires [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/) and [PyYAML](https://pyyaml.org/).

## Search Execution Policy

`Dice.find_parallel(...)` has been replaced by `Dice.find(...)`, which uses an explicit execution policy.

```python
import porems as pms

block = pms.BetaCristobalit().generate([2, 2, 2], "z")
dice = pms.Dice(block, 0.2, True)

policy = pms.SearchPolicy(
    execution=pms.SearchExecution.AUTO,
    processes=4,
)
bond_list = dice.find(None, ["Si", "O"], [0.145, 0.165], policy=policy)
```

`AUTO` uses multiprocessing only when the current `__main__` module is safely importable by worker processes. Otherwise it falls back to serial execution and emits a Python warning.

If you explicitly request `SearchExecution.PROCESSES`, run PoreMS from a file-backed `__main__` entrypoint such as a normal script or `python -m unittest ...`.


## Bare Amorphous Slit Preparation

PoreMS exposes a high-level slit-preparation API for building periodic bare
amorphous silica slits with controlled exposed-surface `Q2/Q3/Q4` fractions.

```python
import porems as pms

config = pms.AmorphousSlitConfig(
    name="bare_amorphous_silica_slit",
    slit_width_nm=7.0,
    repeat_y=2,
)

result = pms.prepare_amorphous_slit_surface(config)
print(result.report.prepared_surface)

pms.write_bare_amorphous_slit("output/bare_amorphous_slit", config)
```

`prepare_amorphous_slit_surface(...)` returns a `SlitPreparationResult`
containing an attach-ready `PoreKit` system and a structured
`SlitPreparationReport`.
`write_bare_amorphous_slit(...)` finalizes and stores the generated bare slit
together with a JSON report in the selected output directory. Object backups are
written only when `write_object_files=True` is requested explicitly.


## Installation

Create a Python 3.14 environment, then install the repository from the
repository root in editable mode:

    pip install -r requirements.txt
    pip install -e .

Use the repository directly in editable mode for local work.


## Testing

Run the test suite from the repository root after the editable install:

    python -m pytest tests/ -q


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
