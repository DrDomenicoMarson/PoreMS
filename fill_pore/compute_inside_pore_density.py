#!/usr/bin/env python3
"""Estimate thymol density inside an already merged slit system.

The script assumes all ``THY`` molecules in the input GRO file already belong
to the pore region. It therefore:

1. Counts the retained ``THY`` molecules in the merged structure.
2. Treats every non-``THY`` atom as part of the slit framework.
3. Reuses the Monte Carlo accessible-volume estimator from the merge workflow.

The output is a plain-text log with box-average and probe-dependent accessible
thymol densities.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fill_silica_pore import (
    DEFAULT_DENSITY_PROBE_RADII_NM,
    DensityEstimate,
    GroSystem,
    build_residue_spans,
    compute_density_estimate,
    load_gro_system,
    validate_slit_coordinates,
)


@dataclass(frozen=True)
class PoreDensityConfig:
    """User-configurable settings for pore-density estimation.

    Attributes:
        input_path: Path to the merged slit-plus-thymol GRO file.
        log_path: Path to the plain-text density log that will be written.
        thymol_resname: Residue name used to identify thymol molecules.
        density_probe_radii_nm: Probe radii in nm used for accessible-volume
            estimation.
        density_sample_count: Number of Monte Carlo samples used for each
            repeat.
        density_seed_count: Number of independent Monte Carlo repeats per probe
            radius.
    """

    input_path: Path
    log_path: Path
    thymol_resname: str
    density_probe_radii_nm: tuple[float, ...]
    density_sample_count: int
    density_seed_count: int


@dataclass(frozen=True)
class PoreDensityReport:
    """Summary of a pore-density calculation for one merged system.

    Attributes:
        thymol_molecule_count: Number of ``THY`` residues in the merged system.
        thymol_atom_count: Number of ``THY`` atoms in the merged system.
        framework_atom_count: Number of non-``THY`` atoms treated as framework.
        framework_residue_count: Number of non-``THY`` residues treated as
            framework.
        density_estimate: Box-average and accessible-volume density estimates.
    """

    thymol_molecule_count: int
    thymol_atom_count: int
    framework_atom_count: int
    framework_residue_count: int
    density_estimate: DensityEstimate


def parse_arguments() -> PoreDensityConfig:
    """Parse command-line arguments into a :class:`PoreDensityConfig`.

    Returns:
        Parsed pore-density configuration.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Estimate THY density inside an already merged slit structure by "
            "counting THY molecules and computing framework-only accessible "
            "volume with the same Monte Carlo method used during merging."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("merged_thymol_slit_ring_check.gro"),
        help="Merged slit-plus-thymol GRO file. Default: %(default)s",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help=(
            "Optional output log path. If omitted, the script writes "
            "<input_stem>_density.log next to the input GRO."
        ),
    )
    parser.add_argument(
        "--thymol-resname",
        default="THY",
        help="Residue name used to identify thymol molecules. Default: %(default)s",
    )
    parser.add_argument(
        "--density-probe-radius",
        type=float,
        action="append",
        help=(
            "Probe radius in nm used to estimate accessible pore volume. "
            "Repeat this option to request multiple probe radii. The default "
            "set is 0.00, 0.14, and 0.20 nm."
        ),
    )
    parser.add_argument(
        "--density-samples",
        type=int,
        default=200000,
        help=(
            "Number of Monte Carlo sample points used for each density repeat. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--density-seed-count",
        type=int,
        default=5,
        help=(
            "Number of independent Monte Carlo repeats used for each probe "
            "radius. Seeds are drawn from system entropy and written to the "
            "log. Default: %(default)s"
        ),
    )
    args = parser.parse_args()

    probe_radii = tuple(
        float(radius)
        for radius in (
            args.density_probe_radius
            if args.density_probe_radius is not None
            else DEFAULT_DENSITY_PROBE_RADII_NM
        )
    )
    log_path = args.log
    if log_path is None:
        log_path = args.input.with_name(f"{args.input.stem}_density.log")

    return PoreDensityConfig(
        input_path=args.input,
        log_path=log_path,
        thymol_resname=args.thymol_resname,
        density_probe_radii_nm=probe_radii,
        density_sample_count=args.density_samples,
        density_seed_count=args.density_seed_count,
    )


def validate_config(config: PoreDensityConfig, merged_system: GroSystem) -> None:
    """Validate CLI settings against the loaded merged system.

    Args:
        config: User-provided pore-density configuration.
        merged_system: Loaded merged slit-plus-thymol system.

    Raises:
        ValueError: If the requested setup is incompatible with the input data.
    """

    if not config.density_probe_radii_nm:
        raise ValueError("At least one density probe radius must be provided.")

    if any(radius < 0.0 for radius in config.density_probe_radii_nm):
        raise ValueError("All density probe radii must be non-negative.")

    if config.density_sample_count <= 0:
        raise ValueError("The density sample count must be strictly positive.")

    if config.density_seed_count <= 0:
        raise ValueError("The density seed count must be strictly positive.")

    if config.thymol_resname not in merged_system.residue_names:
        available_residues = sorted(set(merged_system.residue_names))
        raise ValueError(
            f"Residue name {config.thymol_resname!r} was not found in {config.input_path}. "
            f"Available residue names: {', '.join(available_residues)}"
        )

    if all(residue_name == config.thymol_resname for residue_name in merged_system.residue_names):
        raise ValueError(
            "The input GRO file does not contain any non-THY atoms, so no slit "
            "framework is available for accessible-volume estimation."
        )


def build_framework_system(merged_system: GroSystem, thymol_resname: str) -> GroSystem:
    """Extract the non-THY framework from a merged slit-plus-thymol system.

    Args:
        merged_system: Loaded merged system containing slit framework and THY.
        thymol_resname: Residue name used to identify thymol molecules.

    Returns:
        A new :class:`GroSystem` containing only the non-THY framework atoms.
    """

    framework_atom_mask = np.array(
        [residue_name != thymol_resname for residue_name in merged_system.residue_names],
        dtype=bool,
    )
    framework_residue_ids = merged_system.residue_ids[framework_atom_mask].copy()
    framework_residue_names = [
        residue_name
        for atom_index, residue_name in enumerate(merged_system.residue_names)
        if framework_atom_mask[atom_index]
    ]
    framework_atom_names = [
        atom_name
        for atom_index, atom_name in enumerate(merged_system.atom_names)
        if framework_atom_mask[atom_index]
    ]
    framework_atom_ids = merged_system.atom_ids[framework_atom_mask].copy()
    framework_coordinates = merged_system.coordinates[framework_atom_mask].copy()
    framework_velocities = None
    if merged_system.velocities is not None:
        framework_velocities = merged_system.velocities[framework_atom_mask].copy()

    framework_residue_spans, framework_atom_to_residue_index = build_residue_spans(
        residue_ids=framework_residue_ids,
        residue_names=framework_residue_names,
    )
    return GroSystem(
        title=f"{merged_system.title} [framework only]",
        residue_ids=framework_residue_ids,
        residue_names=framework_residue_names,
        atom_names=framework_atom_names,
        atom_ids=framework_atom_ids,
        coordinates=framework_coordinates,
        velocities=framework_velocities,
        box_lengths=merged_system.box_lengths.copy(),
        residue_spans=tuple(framework_residue_spans),
        atom_to_residue_index=framework_atom_to_residue_index,
    )


def count_thymol(merged_system: GroSystem, thymol_resname: str) -> tuple[int, int]:
    """Count THY residues and THY atoms in a merged system.

    Args:
        merged_system: Loaded merged slit-plus-thymol system.
        thymol_resname: Residue name used to identify thymol molecules.

    Returns:
        Two items: number of THY residues and number of THY atoms.
    """

    thymol_residue_count = 0
    thymol_atom_count = 0
    for residue_span in merged_system.residue_spans:
        if residue_span.residue_name != thymol_resname:
            continue
        thymol_residue_count += 1
        thymol_atom_count += residue_span.stop - residue_span.start
    return thymol_residue_count, thymol_atom_count


def build_log_text(config: PoreDensityConfig, report: PoreDensityReport) -> str:
    """Build the plain-text pore-density log.

    Args:
        config: Configuration used for the density calculation.
        report: Density results gathered for the merged system.

    Returns:
        Human-readable text log.
    """

    lines = [
        "Merged slit pore density estimate",
        f"input_gro = {config.input_path}",
        f"log_file = {config.log_path}",
        f"target_residue = {config.thymol_resname}",
        f"thymol_molecule_count = {report.thymol_molecule_count}",
        f"thymol_atom_count = {report.thymol_atom_count}",
        f"framework_atom_count = {report.framework_atom_count}",
        f"framework_residue_count = {report.framework_residue_count}",
        f"thymol_molecule_mass_da = {report.density_estimate.thymol_molecule_mass_da:.5f}",
        f"total_thymol_mass_da = {report.density_estimate.total_thymol_mass_da:.5f}",
        f"box_volume_nm3 = {report.density_estimate.box_volume_nm3:.5f}",
        f"box_average_thymol_density_g_cm3 = {report.density_estimate.box_average_density_g_cm3:.5f}",
        f"density_sample_count_per_seed = {report.density_estimate.sample_count_per_seed}",
        f"density_seed_count = {report.density_estimate.seed_count}",
    ]

    for probe_index, probe_estimate in enumerate(report.density_estimate.probe_estimates, start=1):
        lines.append(f"density_probe_radius_nm[{probe_index}] = {probe_estimate.probe_radius_nm:.3f}")
        lines.append(
            "density_seed_values[{index}] = {values}".format(
                index=probe_index,
                values=" ".join(str(seed_value) for seed_value in probe_estimate.seed_values),
            )
        )
        lines.append(
            "density_accessible_fraction_values[{index}] = {values}".format(
                index=probe_index,
                values=" ".join(f"{value:.6f}" for value in probe_estimate.accessible_fractions),
            )
        )
        lines.append(
            "density_accessible_volume_values_nm3[{index}] = {values}".format(
                index=probe_index,
                values=" ".join(f"{value:.6f}" for value in probe_estimate.accessible_volumes_nm3),
            )
        )
        lines.append(
            "density_accessible_density_values_g_cm3[{index}] = {values}".format(
                index=probe_index,
                values=" ".join(
                    "inf" if not np.isfinite(value) else f"{value:.6f}"
                    for value in probe_estimate.accessible_densities_g_cm3
                ),
            )
        )
        lines.append(
            f"density_accessible_fraction_mean[{probe_index}] = "
            f"{probe_estimate.accessible_fraction_mean:.6f}"
        )
        lines.append(
            f"density_accessible_fraction_std[{probe_index}] = "
            f"{probe_estimate.accessible_fraction_std:.6f}"
        )
        lines.append(
            f"density_accessible_volume_mean_nm3[{probe_index}] = "
            f"{probe_estimate.accessible_volume_mean_nm3:.6f}"
        )
        lines.append(
            f"density_accessible_volume_std_nm3[{probe_index}] = "
            f"{probe_estimate.accessible_volume_std_nm3:.6f}"
        )
        density_mean_text = (
            "inf"
            if not np.isfinite(probe_estimate.accessible_density_mean_g_cm3)
            else f"{probe_estimate.accessible_density_mean_g_cm3:.6f}"
        )
        density_std_text = (
            "inf"
            if not np.isfinite(probe_estimate.accessible_density_std_g_cm3)
            else f"{probe_estimate.accessible_density_std_g_cm3:.6f}"
        )
        lines.append(
            f"density_accessible_density_mean_g_cm3[{probe_index}] = {density_mean_text}"
        )
        lines.append(
            f"density_accessible_density_std_g_cm3[{probe_index}] = {density_std_text}"
        )

    return "\n".join(lines) + "\n"


def estimate_pore_density(config: PoreDensityConfig) -> PoreDensityReport:
    """Run the pore-density workflow for one merged GRO file.

    Args:
        config: Density-estimation settings and file paths.

    Returns:
        Density summary for the merged system.
    """

    merged_system = load_gro_system(config.input_path)
    validate_config(config, merged_system)

    framework_system = build_framework_system(
        merged_system=merged_system,
        thymol_resname=config.thymol_resname,
    )
    validate_slit_coordinates(framework_system)
    thymol_molecule_count, thymol_atom_count = count_thymol(
        merged_system=merged_system,
        thymol_resname=config.thymol_resname,
    )
    density_estimate = compute_density_estimate(
        thymol_system=merged_system,
        slit_system=framework_system,
        slit_coordinates=framework_system.coordinates,
        final_box_lengths=merged_system.box_lengths,
        target_resname=config.thymol_resname,
        remaining_thymol_molecules=thymol_molecule_count,
        probe_radii_nm=config.density_probe_radii_nm,
        sample_count=config.density_sample_count,
        seed_count=config.density_seed_count,
    )
    report = PoreDensityReport(
        thymol_molecule_count=thymol_molecule_count,
        thymol_atom_count=thymol_atom_count,
        framework_atom_count=framework_system.atom_count,
        framework_residue_count=len(framework_system.residue_spans),
        density_estimate=density_estimate,
    )
    log_text = build_log_text(config, report)
    config.log_path.write_text(log_text, encoding="utf-8")
    print(log_text, end="")
    return report


def main() -> None:
    """Parse CLI arguments and run the pore-density workflow."""

    config = parse_arguments()
    estimate_pore_density(config)


if __name__ == "__main__":
    main()
