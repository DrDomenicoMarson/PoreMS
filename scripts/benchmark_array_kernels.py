#!/usr/bin/env python3
"""Benchmark the array-backed connectivity and slit-preparation kernels."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from statistics import mean
from time import perf_counter

import numpy as np
import porems as pms

from porems._numba_kernels import minimum_clearance_against_batch


os.environ.setdefault("MPLCONFIGDIR", "/tmp/porems_benchmark_mpl")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class BenchmarkResult:
    """One benchmark measurement summary.

    Parameters
    ----------
    name : str
        Human-readable benchmark name.
    repeats : int
        Number of timing repeats.
    timings_s : list[float]
        Wall-clock timings in seconds.
    mean_s : float
        Arithmetic mean of ``timings_s``.
    """

    name: str
    repeats: int
    timings_s: list[float]
    mean_s: float


def benchmark_build(repeats):
    """Benchmark ``PoreKit.build()`` on the cristobalite fixture.

    Parameters
    ----------
    repeats : int
        Number of timing repeats.

    Returns
    -------
    result : BenchmarkResult
        Timing summary for the build benchmark.
    """
    timings = []
    for _repeat in range(repeats):
        block = pms.BetaCristobalit().generate([6, 6, 6], "z")
        kit = pms.PoreKit()
        kit.structure(block)
        start = perf_counter()
        kit.build()
        timings.append(perf_counter() - start)

    return BenchmarkResult(
        name="porekit_build_beta_cristobalit_6x6x6",
        repeats=repeats,
        timings_s=timings,
        mean_s=mean(timings),
    )


def benchmark_slit_prepare(repeats):
    """Benchmark ``prepare_amorphous_slit_surface()`` on the repeat-y-1 case.

    Parameters
    ----------
    repeats : int
        Number of timing repeats.

    Returns
    -------
    result : BenchmarkResult
        Timing summary for the slit-preparation benchmark.
    """
    config = pms.AmorphousSlitConfig(
        name="benchmark_repeat_y1",
        repeat_y=1,
        surface_target=pms.ExperimentalSiliconStateTarget(
            q2_fraction=0.069,
            q3_fraction=0.681,
            q4_fraction=0.25,
            alpha_override=1.0,
        ),
    )

    timings = []
    for _repeat in range(repeats):
        start = perf_counter()
        pms.prepare_amorphous_slit_surface(config=config)
        timings.append(perf_counter() - start)

    return BenchmarkResult(
        name="prepare_amorphous_slit_surface_repeat_y1",
        repeats=repeats,
        timings_s=timings,
        mean_s=mean(timings),
    )


def _build_prepared_cylinder_pore():
    """Build a prepared cylinder pore and its interior site list.

    Returns
    -------
    result : tuple[pms.Pore, list[int]]
        Prepared pore object and the interior silicon site ids used for the
        attachment benchmark.
    """
    pattern = pms.BetaCristobalit()
    pattern.generate([6, 6, 6], "z")
    block = pattern.get_block()

    dice = pms.Dice(block, 0.4, True)
    bond_list = dice.find(None, ["Si", "O"], [0.155 - 1e-2, 0.155 + 1e-2])
    matrix = pms.Matrix(bond_list)
    pore = pms.Pore(block, matrix)
    pore.exterior()

    centroid = block.centroid()
    central = pms.geom.unit(pms.geom.rotate([0, 0, 1], [1, 0, 0], 0, True))
    cylinder = pms.Cylinder(
        pms.CylinderConfig(
            centroid=tuple(centroid),
            central=tuple(central),
            length=6,
            diameter=4,
        )
    )

    del_list = [
        atom_id
        for atom_id, atom in enumerate(block.get_atom_list())
        if cylinder.is_in(atom.get_pos())
    ]
    matrix.strip(del_list)

    pore.prepare()
    pore.amorph()
    pore.sites()
    site_list = pore.get_sites()
    site_in = [
        site_key
        for site_key, site_val in site_list.items()
        if site_val.site_type == "in"
    ]
    for site in site_in:
        site_list[site].normal = cylinder.normal

    return pore, site_in


def _warm_numba_attachment_kernels():
    """Compile the Numba-backed attachment kernels before benchmarking.

    Returns
    -------
    None
        The function exists for its compilation side effect only.
    """
    minimum_clearance_against_batch(
        np.zeros((1, 3), dtype=float),
        np.zeros(1, dtype=float),
        np.zeros((1, 3), dtype=float),
        np.zeros(1, dtype=float),
        np.zeros(1, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        np.ones(3, dtype=float),
        0.85,
    )


def benchmark_pore_attach(repeats):
    """Benchmark the interior attachment phase on a prepared cylinder pore.

    Parameters
    ----------
    repeats : int
        Number of timing repeats.

    Returns
    -------
    result : BenchmarkResult
        Timing summary for the attachment benchmark.
    """
    _warm_numba_attachment_kernels()
    timings = []
    for _repeat in range(repeats):
        random.seed(0)
        pore, site_in = _build_prepared_cylinder_pore()
        start = perf_counter()
        pore.attach(pms.gen.tms(), 0, [0, 1], site_in, 100, site_type="in")
        timings.append(perf_counter() - start)

    return BenchmarkResult(
        name="pore_attach_tms_interior_100_sites",
        repeats=repeats,
        timings_s=timings,
        mean_s=mean(timings),
    )


def _functionalized_slit_config(ligand_name):
    """Return the script-like functionalized slit configuration.

    Parameters
    ----------
    ligand_name : str
        Ligand selector. Supported values are ``"TEPS"`` and ``"TMS"``.

    Returns
    -------
    config : pms.FunctionalizedAmorphousSlitConfig
        Functionalized slit configuration matching the build script settings.

    Raises
    ------
    ValueError
        Raised when ``ligand_name`` is not recognized.
    """
    ligand_key = ligand_name.upper()
    if ligand_key == "TEPS":
        molecule = pms.Molecule(
            "TEPS",
            "TEPS",
            os.path.join(SCRIPT_DIR, "TEPS.pdb"),
        )
        surface_target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=1.01 / 100.0,
            q3_fraction=12.75 / 100.0,
            t2_fraction=6.22 / 100.0,
            t3_fraction=11.44 / 100.0,
            alpha_override=0.328,
        )
    elif ligand_key == "TMS":
        molecule = pms.gen.tms()
        surface_target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=1.01 / 100.0,
            q3_fraction=12.75 / 100.0,
            t2_fraction=6.22 / 100.0,
            t3_fraction=11.44 / 100.0,
            alpha_override=0.328,
        )
    else:
        raise ValueError(f"Unsupported functionalized slit ligand {ligand_name!r}.")

    return pms.FunctionalizedAmorphousSlitConfig(
        slit_config=pms.AmorphousSlitConfig(
            name=f"benchmark_{ligand_key.lower()}",
            slit_width_nm=7.0,
            repeat_y=1,
            temperature_k=308.0,
            surface_target=surface_target,
        ),
        ligand=pms.SilaneAttachmentConfig(
            molecule=molecule,
            mount=0,
            rotate_about_axis=True,
            rotate_step_deg=30.0,
            axis=(0, 1),
        ),
        progress_settings=pms.FunctionalizedSlitProgressConfig(enabled=False),
    )


def benchmark_functionalized_slit(ligand_name, repeats):
    """Benchmark the script-like functionalized slit preparation path.

    Parameters
    ----------
    ligand_name : str
        Ligand selector forwarded to :func:`_functionalized_slit_config`.
    repeats : int
        Number of timing repeats.

    Returns
    -------
    result : BenchmarkResult
        Timing summary for the requested functionalized slit case.
    """
    _warm_numba_attachment_kernels()
    timings = []
    for _repeat in range(repeats):
        random.seed(0)
        np.random.seed(0)
        config = _functionalized_slit_config(ligand_name)
        start = perf_counter()
        pms.prepare_functionalized_amorphous_slit_surface(config=config)
        timings.append(perf_counter() - start)

    return BenchmarkResult(
        name=f"prepare_functionalized_amorphous_slit_surface_{ligand_name.lower()}_repeat_y1",
        repeats=repeats,
        timings_s=timings,
        mean_s=mean(timings),
    )


def main():
    """Run the available benchmark scenarios."""
    parser = argparse.ArgumentParser(
        description="Benchmark the array-backed PoreMS kernels."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timing repeats for each benchmark.",
    )
    parser.add_argument(
        "--functionalized-ligand",
        action="append",
        default=[],
        choices=("TEPS", "TMS"),
        help=(
            "Optional script-like functionalized slit benchmark to run. "
            "Repeat the flag to benchmark multiple ligands."
        ),
    )
    args = parser.parse_args()

    results = [
        benchmark_build(args.repeats),
        benchmark_slit_prepare(args.repeats),
        benchmark_pore_attach(args.repeats),
    ]
    for ligand_name in args.functionalized_ligand:
        results.append(benchmark_functionalized_slit(ligand_name, args.repeats))
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
