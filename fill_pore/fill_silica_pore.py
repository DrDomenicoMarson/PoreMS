#!/usr/bin/env python3
"""Fill a grafted silica slit with thymol using slit-plane and ring checks.

The script center-crops a larger thymol reservoir box to the slit cell,
removes target thymol residues that fall outside automatically detected slit
surface planes, removes residues that fail a small all-atom cutoff, then
applies an explicit symmetric ring-threading check:

1. THY atoms outside the detected slit interval normal to the confining faces.
2. THY bonds crossing slit phenyl rings.
3. Slit bonds crossing THY phenyl rings.

The final output is a merged GRO file, rotated so the detected slit normal lies
along ``z``, plus a text log with molecule counts, ring-check statistics, and
multi-probe density estimates inside the slit.
"""

from __future__ import annotations

import argparse
import secrets
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
BoolArray = NDArray[np.bool_]

COVALENT_RADII_NM = {
    "H": 0.031,
    "C": 0.076,
    "O": 0.066,
    "Si": 0.111,
}

ATOMIC_MASSES_DA = {
    "H": 1.008,
    "C": 12.011,
    "O": 15.999,
    "Si": 28.085,
}

VDW_RADII_NM = {
    "H": 0.120,
    "C": 0.170,
    "O": 0.152,
    "Si": 0.210,
}

DEFAULT_DENSITY_PROBE_RADII_NM = (0.00, 0.14, 0.20)
BOX_LINE_TOLERANCE_NM = 1.0e-6
SLIT_COORDINATE_TOLERANCE_NM = 1.0e-6
GRAMS_PER_DA = 1.66053906660e-24
CUBIC_CENTIMETERS_PER_NM3 = 1.0e-21
AXIS_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class ResidueSpan:
    """Contiguous atom range for one residue in a GRO file.

    Attributes:
        residue_id: Residue identifier read from the GRO file.
        residue_name: Residue name read from the GRO file.
        start: Inclusive atom index at which the residue starts.
        stop: Exclusive atom index at which the residue ends.
    """

    residue_id: int
    residue_name: str
    start: int
    stop: int


@dataclass(frozen=True)
class GroSystem:
    """In-memory representation of the subset of GRO data needed for merging.

    Attributes:
        title: Title line from the GRO file.
        residue_ids: Residue identifiers for all atoms.
        residue_names: Residue names for all atoms.
        atom_names: Atom names for all atoms.
        atom_ids: Atom identifiers for all atoms.
        coordinates: Cartesian coordinates in nm with shape ``(n_atoms, 3)``.
        velocities: Optional Cartesian velocities in nm/ps with shape
            ``(n_atoms, 3)``. The value is ``None`` when the input GRO does not
            provide velocities.
        box_lengths: Orthorhombic box lengths in nm.
        residue_spans: Contiguous residue spans in file order.
        atom_to_residue_index: Mapping from atom index to index in
            ``residue_spans``.
    """

    title: str
    residue_ids: IntArray
    residue_names: list[str]
    atom_names: list[str]
    atom_ids: IntArray
    coordinates: FloatArray
    velocities: FloatArray | None
    box_lengths: FloatArray
    residue_spans: tuple[ResidueSpan, ...]
    atom_to_residue_index: IntArray

    @property
    def atom_count(self) -> int:
        """Return the number of atoms in the system."""

        return int(self.coordinates.shape[0])


@dataclass(frozen=True)
class MergeConfig:
    """User-configurable settings for the merge operation.

    Attributes:
        thymol_path: Path to the GRO file containing the thymol box.
        slit_path: Path to the GRO file containing the grafted silica slit.
        output_path: Path to the merged GRO file that will be written.
        log_path: Path to the text log file that will be written.
        thymol_resname: Residue name used to identify removable thymol molecules.
        general_cutoff_nm: Lower all-atom clash cutoff in nm.
        ring_atom_prefix: Prefix that identifies phenyl-ring atoms such as
            ``CA1`` or ``CA6``.
        ring_plane_tolerance_nm: Maximum distance in nm between a bond and a
            ring plane for that contact to count as crossing-like.
        ring_polygon_padding_nm: Extra in-plane padding in nm applied when
            deciding whether a projected bond point falls inside a phenyl ring.
        include_hydrogen_bonds_in_ring_check: Whether bonds to hydrogen atoms
            participate in the explicit ring-crossing tests. The default is
            ``True``.
        use_surface_plane_filter: Whether THY residues should be removed when
            any atom falls outside the automatically detected slit-surface
            planes.
        surface_plane_padding_nm: Signed padding in nm applied on each side of
            the detected slit interval before THY selection. Positive values
            shrink the allowed slit region, while negative values expand it
            deeper into the silica matrix.
        density_probe_radii_nm: Probe radii in nm used to estimate accessible
            slit volume for the thymol density estimate.
        density_sample_count: Number of Monte Carlo points used for each density
            repeat.
        density_seed_count: Number of independent density repeats.
        wrap_output: Whether to wrap retained THY residues back into the final
            box before writing the output GRO file.
    """

    thymol_path: Path
    slit_path: Path
    output_path: Path
    log_path: Path
    thymol_resname: str
    general_cutoff_nm: float
    ring_atom_prefix: str
    ring_plane_tolerance_nm: float
    ring_polygon_padding_nm: float
    include_hydrogen_bonds_in_ring_check: bool
    use_surface_plane_filter: bool
    surface_plane_padding_nm: float
    density_probe_radii_nm: tuple[float, ...]
    density_sample_count: int
    density_seed_count: int
    wrap_output: bool


@dataclass(frozen=True)
class SurfacePlaneRegion:
    """Axis-aligned slit interval inferred from hydroxylated surface Si atoms.

    Attributes:
        axis_index: Coordinate axis normal to the two confining slit planes.
        axis_name: Human-readable coordinate axis label.
        lower_plane_nm: Lower plane position in wrapped coordinates.
        upper_plane_nm: Upper plane position in wrapped coordinates.
        interval_wraps: Whether the accessible interval crosses the periodic box
            boundary on the selected axis.
        accessible_width_nm: Width of the accessible interval before padding.
        padding_nm: Inward padding applied on each side of the accessible
            interval.
        surface_si_atom_count: Number of hydroxylated surface Si atoms used to
            infer the interval.
    """

    axis_index: int
    axis_name: str
    lower_plane_nm: float
    upper_plane_nm: float
    interval_wraps: bool
    accessible_width_nm: float
    padding_nm: float
    surface_si_atom_count: int


@dataclass(frozen=True)
class SurfacePlaneSelection:
    """Result of filtering THY residues against the detected slit planes.

    Attributes:
        selected_residue_mask: Updated residue mask after the slit-plane filter.
        removed_residue_mask: Residue mask marking THY residues removed by the
            slit-plane filter.
        plane_region: Detected slit interval used for the selection.
    """

    selected_residue_mask: BoolArray
    removed_residue_mask: BoolArray
    plane_region: SurfacePlaneRegion


@dataclass(frozen=True)
class BondDefinition:
    """One guessed covalent bond in a residue template.

    Attributes:
        start_atom_index: Atom index within the residue.
        stop_atom_index: Atom index within the residue.
        start_atom_name: Atom name at the start of the bond.
        stop_atom_name: Atom name at the end of the bond.
    """

    start_atom_index: int
    stop_atom_index: int
    start_atom_name: str
    stop_atom_name: str


@dataclass(frozen=True)
class ResidueBondTemplate:
    """Guessed covalent-bond template for one residue topology.

    Attributes:
        residue_name: Residue name associated with the template.
        atom_names: Ordered atom names used to define the template.
        bond_definitions: Guessed covalent bonds for that topology.
    """

    residue_name: str
    atom_names: tuple[str, ...]
    bond_definitions: tuple[BondDefinition, ...]


@dataclass(frozen=True)
class RingTemplate:
    """Local atom-index template for one phenyl ring.

    Attributes:
        residue_name: Residue name associated with the ring template.
        atom_prefix: Prefix used to identify the ring atoms.
        local_atom_indices: Local residue atom indices for the six ring atoms.
        ring_atom_names: Ordered ring atom names stored for reference.
    """

    residue_name: str
    atom_prefix: str
    local_atom_indices: tuple[int, ...]
    ring_atom_names: tuple[str, ...]


@dataclass(frozen=True)
class RingGeometry:
    """Geometric model of one phenyl ring.

    Attributes:
        residue_index: Index of the parent residue in ``GroSystem.residue_spans``.
        residue_name: Residue name of the parent residue.
        center: Ring center in a box image convenient for geometric testing.
        wrapped_center: Ring center wrapped into the final simulation box.
        normal: Unit normal vector of the ring plane.
        basis_u: First in-plane unit basis vector.
        basis_v: Second in-plane unit basis vector.
        polygon_2d: Ordered 2D polygon of the ring atoms after projection into
            the ring plane basis.
        max_radius_nm: Largest in-plane atom distance from the ring center.
    """

    residue_index: int
    residue_name: str
    center: FloatArray
    wrapped_center: FloatArray
    normal: FloatArray
    basis_u: FloatArray
    basis_v: FloatArray
    polygon_2d: FloatArray
    max_radius_nm: float


@dataclass(frozen=True)
class BondSegmentGeometry:
    """Geometric representation of one residue bond segment.

    Attributes:
        residue_index: Index of the parent residue in ``GroSystem.residue_spans``.
        residue_name: Residue name of the parent residue.
        start_atom_name: Atom name at the start of the bond.
        stop_atom_name: Atom name at the end of the bond.
        start_point: Start point of the bond segment in nm.
        stop_point: End point of the bond segment in nm.
        midpoint: Midpoint of the bond segment in the chosen box image.
        wrapped_midpoint: Midpoint wrapped into the final simulation box.
        half_length_nm: Half of the bond-segment length in nm.
    """

    residue_index: int
    residue_name: str
    start_atom_name: str
    stop_atom_name: str
    start_point: FloatArray
    stop_point: FloatArray
    midpoint: FloatArray
    wrapped_midpoint: FloatArray
    half_length_nm: float


@dataclass(frozen=True)
class DensityProbeEstimate:
    """Density-estimate summary for one probe radius.

    Attributes:
        probe_radius_nm: Probe radius in nm.
        seed_values: Actual random seeds used for the repeated estimates.
        accessible_fractions: Accessible slit fractions for each repeat.
        accessible_volumes_nm3: Accessible slit volumes for each repeat in nm^3.
        accessible_densities_g_cm3: Thymol densities in the accessible slit
            volume for each repeat.
        accessible_fraction_mean: Mean accessible fraction across repeats.
        accessible_fraction_std: Standard deviation of accessible fraction across
            repeats.
        accessible_volume_mean_nm3: Mean accessible volume across repeats.
        accessible_volume_std_nm3: Standard deviation of accessible volume across
            repeats.
        accessible_density_mean_g_cm3: Mean accessible density across repeats.
        accessible_density_std_g_cm3: Standard deviation of accessible density
            across repeats.
    """

    probe_radius_nm: float
    seed_values: tuple[int, ...]
    accessible_fractions: tuple[float, ...]
    accessible_volumes_nm3: tuple[float, ...]
    accessible_densities_g_cm3: tuple[float, ...]
    accessible_fraction_mean: float
    accessible_fraction_std: float
    accessible_volume_mean_nm3: float
    accessible_volume_std_nm3: float
    accessible_density_mean_g_cm3: float
    accessible_density_std_g_cm3: float


@dataclass(frozen=True)
class DensityEstimate:
    """Estimated thymol density metrics for the final slit filling.

    Attributes:
        thymol_molecule_mass_da: Mass of one target thymol molecule in daltons.
        total_thymol_mass_da: Total mass of all remaining thymol molecules in
            daltons.
        box_volume_nm3: Final periodic box volume in nm^3.
        box_average_density_g_cm3: Density obtained by dividing the thymol mass
            by the full periodic box volume.
        sample_count_per_seed: Number of Monte Carlo samples used for each
            density repeat.
        seed_count: Number of independent density repeats.
        probe_estimates: Per-probe accessible-volume and accessible-density
            summaries.
    """

    thymol_molecule_mass_da: float
    total_thymol_mass_da: float
    box_volume_nm3: float
    box_average_density_g_cm3: float
    sample_count_per_seed: int
    seed_count: int
    probe_estimates: tuple[DensityProbeEstimate, ...]


@dataclass(frozen=True)
class ClashSelection:
    """Residue-level clash results for the general and ring-crossing filters.

    Attributes:
        removed_residue_mask: Residues removed by any clash rule.
        removed_by_general_mask: Residues removed by the lower all-atom cutoff.
        removed_by_forward_ring_mask: Residues removed because a THY bond
            crosses a slit phenyl ring.
        removed_by_reverse_ring_mask: Residues removed because a slit bond
            crosses a THY phenyl ring.
        removed_by_any_ring_mask: Residues removed by either ring-crossing rule.
    """

    removed_residue_mask: BoolArray
    removed_by_general_mask: BoolArray
    removed_by_forward_ring_mask: BoolArray
    removed_by_reverse_ring_mask: BoolArray
    removed_by_any_ring_mask: BoolArray


@dataclass(frozen=True)
class RingCheckCache:
    """Cached geometry data reused after clash detection.

    Attributes:
        slit_ring_geometries: Slit phenyl-ring geometries used in the forward
            THY-bond check.
        target_bond_template: Bond template used for THY residues.
        target_ring_template: Ring template used for THY residues.
        target_ring_geometries: THY ring geometries built for cropped target
            residues.
        slit_bond_templates: Unique cached slit bond templates.
        slit_bond_geometries: Slit bond segments used in the reverse ring check.
    """

    slit_ring_geometries: tuple[RingGeometry, ...]
    target_bond_template: tuple[BondDefinition, ...]
    target_ring_template: RingTemplate
    target_ring_geometries: tuple[RingGeometry, ...]
    slit_bond_templates: tuple[ResidueBondTemplate, ...]
    slit_bond_geometries: tuple[BondSegmentGeometry, ...]


@dataclass(frozen=True)
class MergeReport:
    """Summary of the performed merge.

    Attributes:
        initial_thymol_molecules: Number of target THY residues found in the
            thymol input file.
        cropped_thymol_molecules: Number of THY residues fully inside the
            centered slit-sized crop before clash filtering.
        removed_outside_crop_thymol_molecules: Number of THY residues discarded
            because they lie at least partly outside the centered crop.
        surface_plane_region: Automatically detected slit interval used for the
            optional surface-plane filter, or ``None`` when that filter is
            disabled.
        surface_plane_filtered_thymol_molecules: Number of THY residues that
            remain after the optional surface-plane filter.
        removed_by_surface_plane_thymol_molecules: Number of THY residues
            removed because at least one atom falls outside the detected slit
            planes.
        removed_by_general_cutoff_thymol_molecules: Number of THY residues
            removed by the lower all-atom cutoff.
        removed_by_forward_ring_thymol_molecules: Number of THY residues removed
            because a THY bond crosses a slit phenyl ring.
        removed_by_reverse_ring_thymol_molecules: Number of THY residues removed
            because a slit bond crosses a THY phenyl ring.
        removed_by_any_ring_thymol_molecules: Number of THY residues removed by
            either ring-crossing rule.
        removed_by_general_only_thymol_molecules: Number of THY residues removed
            only by the lower all-atom cutoff.
        removed_by_forward_ring_only_thymol_molecules: Number of THY residues
            removed only by the forward ring-crossing rule.
        removed_by_reverse_ring_only_thymol_molecules: Number of THY residues
            removed only by the reverse ring-crossing rule.
        removed_by_general_and_forward_ring_only_thymol_molecules: Number of THY
            residues removed by the general cutoff and forward ring rule, but not
            by the reverse ring rule.
        removed_by_general_and_reverse_ring_only_thymol_molecules: Number of THY
            residues removed by the general cutoff and reverse ring rule, but not
            by the forward ring rule.
        removed_by_forward_and_reverse_ring_only_thymol_molecules: Number of THY
            residues removed by both ring-crossing rules, but not by the general
            cutoff.
        removed_by_general_and_forward_and_reverse_ring_thymol_molecules: Number
            of THY residues removed by all three clash rules.
        removed_by_clash_thymol_molecules: Number of THY residues removed by any
            clash rule after crop selection.
        removed_thymol_molecules: Total number of removed THY residues.
        remaining_thymol_molecules: Number of THY residues kept in the merged
            output.
        slit_phenyl_ring_count: Number of slit phenyl rings used in the
            forward ring-crossing check.
        cropped_thymol_ring_count: Number of THY rings built for cropped THY
            residues.
        thymol_bonds_checked_per_molecule: Number of guessed THY bonds tested
            against slit phenyl rings for each THY molecule.
        slit_bond_template_count: Number of unique slit residue bond templates.
        slit_bond_count_checked: Number of slit bond segments tested against THY
            rings in the reverse direction.
        density_estimate: Estimated thymol density metrics for the final slit
            filling.
        slit_atom_count: Number of atoms copied from the slit structure.
        final_atom_count: Number of atoms written to the merged GRO file.
        final_residue_count: Number of residues written to the merged GRO file.
        output_axis_permutation: Axis permutation applied when writing the final
            merged coordinates so that the slit normal lies along ``z``.
        crop_window_start_nm: Lower corner of the centered crop window in the
            original thymol box.
        output_box_nm: Orthorhombic box lengths written to the merged GRO file.
    """

    initial_thymol_molecules: int
    cropped_thymol_molecules: int
    removed_outside_crop_thymol_molecules: int
    surface_plane_region: SurfacePlaneRegion | None
    surface_plane_filtered_thymol_molecules: int
    removed_by_surface_plane_thymol_molecules: int
    removed_by_general_cutoff_thymol_molecules: int
    removed_by_forward_ring_thymol_molecules: int
    removed_by_reverse_ring_thymol_molecules: int
    removed_by_any_ring_thymol_molecules: int
    removed_by_general_only_thymol_molecules: int
    removed_by_forward_ring_only_thymol_molecules: int
    removed_by_reverse_ring_only_thymol_molecules: int
    removed_by_general_and_forward_ring_only_thymol_molecules: int
    removed_by_general_and_reverse_ring_only_thymol_molecules: int
    removed_by_forward_and_reverse_ring_only_thymol_molecules: int
    removed_by_general_and_forward_and_reverse_ring_thymol_molecules: int
    removed_by_clash_thymol_molecules: int
    removed_thymol_molecules: int
    remaining_thymol_molecules: int
    slit_phenyl_ring_count: int
    cropped_thymol_ring_count: int
    thymol_bonds_checked_per_molecule: int
    slit_bond_template_count: int
    slit_bond_count_checked: int
    density_estimate: DensityEstimate
    slit_atom_count: int
    final_atom_count: int
    final_residue_count: int
    output_axis_permutation: tuple[int, int, int]
    crop_window_start_nm: FloatArray
    output_box_nm: FloatArray


def parse_arguments() -> MergeConfig:
    """Parse command-line arguments into a :class:`MergeConfig`.

    Returns:
        Parsed merge configuration.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Center-crop a larger thymol box to the slit cell, remove clashing "
            "THY molecules outside the detected slit planes or within a small "
            "all-atom cutoff, then reject THY residues involved in symmetric "
            "phenyl-ring crossings."
        )
    )
    parser.add_argument(
        "--thymol",
        type=Path,
        default=Path("confout.gro"),
        help="GRO file containing the thymol box. Default: %(default)s",
    )
    parser.add_argument(
        "--slit",
        type=Path,
        default=Path("msn_9_1.gro"),
        help="GRO file containing the grafted silica slit. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("merged_thymol_slit_ring_check.gro"),
        help="Path to the merged GRO file. Default: %(default)s",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("merged_thymol_slit_ring_check.log"),
        help="Path to the text log file. Default: %(default)s",
    )
    parser.add_argument(
        "--thymol-resname",
        default="THY",
        help="Residue name that identifies removable thymol molecules.",
    )
    parser.add_argument(
        "--general-cutoff",
        type=float,
        default=0.18,
        help=(
            "All-atom clash cutoff in nm.  Thymol residues with any atom closer "
            "than this distance to any slit atom are removed.  The default "
            "(0.18 nm) is intentionally small because the explicit phenyl-ring "
            "check handles aromatic threading separately.  For a single-cutoff "
            "workflow without the ring check, a value around 0.35 nm is typical."
        ),
    )
    parser.add_argument(
        "--ring-atom-prefix",
        default="CA",
        help="Atom-name prefix used to identify phenyl-ring atoms. Default: %(default)s",
    )
    parser.add_argument(
        "--ring-plane-tolerance",
        type=float,
        default=0.04,
        help=(
            "Maximum distance in nm between a bond and a ring plane for the "
            "contact to count as crossing-like. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--ring-polygon-padding",
        type=float,
        default=0.02,
        help=(
            "Extra in-plane padding in nm added around the phenyl polygon "
            "during the ring-crossing tests. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--exclude-hydrogen-bonds-from-ring-check",
        action="store_true",
        help="Skip O-H and C-H bonds in the explicit ring-crossing tests.",
    )
    parser.add_argument(
        "--disable-surface-plane-filter",
        action="store_true",
        help=(
            "Skip the automatic slit-plane filter that removes THY residues "
            "outside the hydroxylated surface Si planes."
        ),
    )
    parser.add_argument(
        "--surface-plane-padding",
        type=float,
        default=0.0,
        help=(
            "Signed padding in nm applied on each side of the detected slit "
            "interval before THY selection. Positive values shrink the "
            "allowed region; negative values expand it into the matrix. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--density-probe-radius",
        type=float,
        action="append",
        help=(
            "Probe radius in nm used to estimate accessible slit volume.  "
            "Repeat this option to request multiple probe radii.  The default "
            "set is 0.00, 0.14, and 0.20 nm."
        ),
    )
    parser.add_argument(
        "--density-samples",
        type=int,
        default=200000,
        help=(
            "Number of Monte Carlo sample points used for each density repeat.  "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--density-seed-count",
        type=int,
        default=5,
        help=(
            "Number of independent Monte Carlo repeats used for each probe "
            "radius.  Seeds are drawn from system entropy and written to the "
            "log.  Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--no-wrap-output",
        action="store_true",
        help="Write coordinates exactly as merged instead of wrapping kept THY residues into the final box.",
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
    return MergeConfig(
        thymol_path=args.thymol,
        slit_path=args.slit,
        output_path=args.output,
        log_path=args.log,
        thymol_resname=args.thymol_resname,
        general_cutoff_nm=args.general_cutoff,
        ring_atom_prefix=args.ring_atom_prefix,
        ring_plane_tolerance_nm=args.ring_plane_tolerance,
        ring_polygon_padding_nm=args.ring_polygon_padding,
        include_hydrogen_bonds_in_ring_check=not args.exclude_hydrogen_bonds_from_ring_check,
        use_surface_plane_filter=not args.disable_surface_plane_filter,
        surface_plane_padding_nm=args.surface_plane_padding,
        density_probe_radii_nm=probe_radii,
        density_sample_count=args.density_samples,
        density_seed_count=args.density_seed_count,
        wrap_output=not args.no_wrap_output,
    )


def load_gro_system(path: Path) -> GroSystem:
    """Load a GRO file into a :class:`GroSystem`.

    Args:
        path: Path to the GRO file.

    Returns:
        Parsed GRO system.

    Raises:
        ValueError: If the file does not contain an orthorhombic box.
    """

    with path.open("r", encoding="utf-8") as handle:
        title = handle.readline().rstrip("\n")
        atom_count = int(handle.readline().strip())

        residue_ids = np.empty(atom_count, dtype=np.int32)
        atom_ids = np.empty(atom_count, dtype=np.int32)
        coordinates = np.empty((atom_count, 3), dtype=np.float64)
        velocities = np.zeros((atom_count, 3), dtype=np.float64)
        has_velocities = False
        residue_names: list[str] = []
        atom_names: list[str] = []

        for atom_index in range(atom_count):
            line = handle.readline().rstrip("\n")
            residue_ids[atom_index] = int(line[0:5])
            residue_names.append(line[5:10].strip())
            atom_names.append(line[10:15].strip())
            atom_ids[atom_index] = int(line[15:20])
            coordinates[atom_index, 0] = float(line[20:28])
            coordinates[atom_index, 1] = float(line[28:36])
            coordinates[atom_index, 2] = float(line[36:44])

            if len(line) >= 68:
                velocities[atom_index, 0] = float(line[44:52])
                velocities[atom_index, 1] = float(line[52:60])
                velocities[atom_index, 2] = float(line[60:68])
                has_velocities = True

        box_values = [float(value) for value in handle.readline().split()]

    if len(box_values) == 3:
        orthorhombic_box = np.array(box_values, dtype=np.float64)
    elif len(box_values) == 9:
        off_diagonal_values = np.array(box_values[3:9], dtype=np.float64)
        if np.any(np.abs(off_diagonal_values) > BOX_LINE_TOLERANCE_NM):
            raise ValueError(
                f"{path} uses a non-orthorhombic 9-value GRO box with non-negligible "
                f"off-diagonal terms: {off_diagonal_values.tolist()}"
            )
        orthorhombic_box = np.array(box_values[:3], dtype=np.float64)
    else:
        raise ValueError(
            f"{path} uses a non-orthorhombic box with {len(box_values)} values; "
            "this script currently supports only orthorhombic GRO boxes."
        )

    residue_spans, atom_to_residue_index = build_residue_spans(residue_ids, residue_names)
    return GroSystem(
        title=title,
        residue_ids=residue_ids,
        residue_names=residue_names,
        atom_names=atom_names,
        atom_ids=atom_ids,
        coordinates=coordinates,
        velocities=velocities if has_velocities else None,
        box_lengths=orthorhombic_box,
        residue_spans=tuple(residue_spans),
        atom_to_residue_index=atom_to_residue_index,
    )


def build_residue_spans(
    residue_ids: IntArray,
    residue_names: list[str],
) -> tuple[list[ResidueSpan], IntArray]:
    """Build contiguous residue spans for atoms read from a GRO file.

    Args:
        residue_ids: Residue identifiers for all atoms.
        residue_names: Residue names for all atoms.

    Returns:
        Two items: the residue spans in file order, and the atom-to-residue-span
        index mapping.
    """

    spans: list[ResidueSpan] = []
    atom_to_residue_index = np.empty(residue_ids.shape[0], dtype=np.int32)

    start = 0
    span_index = -1
    while start < residue_ids.shape[0]:
        residue_id = int(residue_ids[start])
        residue_name = residue_names[start]
        stop = start + 1
        while stop < residue_ids.shape[0]:
            if residue_ids[stop] != residue_id or residue_names[stop] != residue_name:
                break
            stop += 1

        span_index += 1
        spans.append(
            ResidueSpan(
                residue_id=residue_id,
                residue_name=residue_name,
                start=start,
                stop=stop,
            )
        )
        atom_to_residue_index[start:stop] = span_index
        start = stop

    return spans, atom_to_residue_index


def validate_slit_coordinates(
    slit_system: GroSystem,
    tolerance_nm: float = SLIT_COORDINATE_TOLERANCE_NM,
) -> None:
    """Warn when slit coordinates lie noticeably outside their stated box.

    Args:
        slit_system: Loaded slit system.
        tolerance_nm: Allowed coordinate tolerance in nm.
    """

    below_box = (-tolerance_nm) - slit_system.coordinates
    above_box = slit_system.coordinates - (slit_system.box_lengths + tolerance_nm)
    out_of_range_mask = np.any((below_box > 0.0) | (above_box > 0.0), axis=1)
    if not np.any(out_of_range_mask):
        return

    worst_below_nm = float(np.max(np.maximum(below_box, 0.0)))
    worst_above_nm = float(np.max(np.maximum(above_box, 0.0)))
    worst_offset_nm = max(worst_below_nm, worst_above_nm)
    warnings.warn(
        (
            f"{np.count_nonzero(out_of_range_mask)} slit atoms fall outside the nominal slit box "
            f"range [-{tolerance_nm:.1e}, box + {tolerance_nm:.1e}); worst offset = "
            f"{worst_offset_nm:.6f} nm."
        ),
        stacklevel=2,
    )


def validate_config(config: MergeConfig, thymol_system: GroSystem, slit_system: GroSystem) -> None:
    """Validate merge settings against the loaded systems.

    Args:
        config: User-provided merge configuration.
        thymol_system: Loaded thymol system.
        slit_system: Loaded slit system.

    Raises:
        ValueError: If any geometric or density-estimation parameter is invalid,
            the thymol residue is missing, or the slit box does not fit inside
            the thymol box for center-cropping.
    """

    if config.general_cutoff_nm <= 0.0:
        raise ValueError("The general clash cutoff must be strictly positive.")

    if config.ring_plane_tolerance_nm < 0.0:
        raise ValueError("The ring-plane tolerance must be non-negative.")

    if config.ring_polygon_padding_nm < 0.0:
        raise ValueError("The ring polygon padding must be non-negative.")

    if not np.isfinite(config.surface_plane_padding_nm):
        raise ValueError("The surface-plane padding must be finite.")

    if not config.density_probe_radii_nm:
        raise ValueError("At least one density probe radius must be provided.")

    if any(radius < 0.0 for radius in config.density_probe_radii_nm):
        raise ValueError("All density probe radii must be non-negative.")

    if config.density_sample_count <= 0:
        raise ValueError("The density sample count must be strictly positive.")

    if config.density_seed_count <= 0:
        raise ValueError("The density seed count must be strictly positive.")

    if config.thymol_resname not in thymol_system.residue_names:
        available_residues = sorted(set(thymol_system.residue_names))
        raise ValueError(
            f"Residue name {config.thymol_resname!r} was not found in {config.thymol_path}. "
            f"Available residue names: {', '.join(available_residues)}"
        )

    if np.any(slit_system.box_lengths > thymol_system.box_lengths):
        raise ValueError(
            "The slit box is larger than the thymol box in at least one dimension, "
            "so the thymol reservoir cannot be center-cropped into the slit cell."
        )


def wrap_positions(coordinates: FloatArray, box_lengths: FloatArray) -> FloatArray:
    """Wrap Cartesian coordinates into an orthorhombic simulation box.

    Args:
        coordinates: Cartesian coordinates with shape ``(n_atoms, 3)``.
        box_lengths: Orthorhombic box lengths in nm.

    Returns:
        Wrapped coordinates in the interval ``[0, box_length)`` for each axis.
    """

    return np.mod(coordinates, box_lengths)


def wrap_residues(
    system: GroSystem,
    coordinates: FloatArray,
    keep_atom_mask: BoolArray,
    box_lengths: FloatArray,
) -> FloatArray:
    """Wrap kept residues into the final box by whole-residue image shifts.

    Args:
        system: Source system that provides residue spans.
        coordinates: Coordinates to wrap.
        keep_atom_mask: Boolean mask over atoms that marks residues kept in the
            final output.
        box_lengths: Orthorhombic box lengths in nm.

    Returns:
        Coordinates where each kept residue has been shifted by a whole-box
        image so that its center of geometry lies in ``[0, box)``.
    """

    wrapped_coordinates = coordinates.copy()
    for residue_span in system.residue_spans:
        if not np.all(keep_atom_mask[residue_span.start:residue_span.stop]):
            continue

        residue_coordinates = wrapped_coordinates[residue_span.start:residue_span.stop]
        center_of_geometry = np.mean(residue_coordinates, axis=0)
        image_shift = box_lengths * np.floor(center_of_geometry / box_lengths)
        wrapped_coordinates[residue_span.start:residue_span.stop] = (
            residue_coordinates - image_shift
        )

    return wrapped_coordinates


def infer_element_from_atom_name(atom_name: str) -> str:
    """Infer a chemical element symbol from a GRO atom name.

    Args:
        atom_name: Atom name read from the GRO file.

    Returns:
        Inferred element symbol.

    Raises:
        ValueError: If the atom name does not contain alphabetic characters or
            maps to an unsupported element.
    """

    letters = "".join(character for character in atom_name if character.isalpha())
    if not letters:
        raise ValueError(f"Cannot infer an element from atom name {atom_name!r}.")

    normalized_letters = letters[0].upper() + letters[1:].lower()
    if normalized_letters.startswith("Si"):
        element = "Si"
    else:
        element = normalized_letters[0]

    if element not in COVALENT_RADII_NM:
        raise ValueError(
            f"Unsupported element {element!r} inferred from atom name {atom_name!r}."
        )

    return element


def find_hydroxylated_surface_silicon_indices(slit_system: GroSystem) -> IntArray:
    """Return slit Si atoms that belong to hydroxylated surface residues.

    A residue is treated as hydroxylated when it contains at least one Si atom,
    at least one O atom, and at least one H atom. The Si atoms from those
    residues are used to infer the two confining slit planes.

    Args:
        slit_system: Loaded slit system.

    Returns:
        Integer atom indices of hydroxylated surface Si atoms.

    Raises:
        ValueError: If no hydroxylated surface Si atoms can be identified.
    """

    silicon_indices: list[int] = []
    for residue_span in slit_system.residue_spans:
        residue_elements = tuple(
            infer_element_from_atom_name(atom_name)
            for atom_name in slit_system.atom_names[residue_span.start:residue_span.stop]
        )
        has_oxygen = any(element == "O" for element in residue_elements)
        has_hydrogen = any(element == "H" for element in residue_elements)
        if not has_oxygen or not has_hydrogen:
            continue

        for local_atom_index, element in enumerate(residue_elements):
            if element == "Si":
                silicon_indices.append(residue_span.start + local_atom_index)

    if not silicon_indices:
        raise ValueError(
            "Could not identify any hydroxylated surface Si atoms in the slit "
            "structure, so the automatic slit-plane filter cannot be built."
        )

    return np.array(silicon_indices, dtype=np.int32)


def infer_slit_surface_plane_region(
    slit_system: GroSystem,
    slit_coordinates: FloatArray,
    box_lengths: FloatArray,
    padding_nm: float,
) -> SurfacePlaneRegion:
    """Infer the accessible slit interval from hydroxylated surface Si atoms.

    The algorithm wraps the candidate surface Si atoms into the slit box, then
    searches each axis for the largest periodic gap. The axis with the largest
    such gap is treated as the slit normal, and that gap is interpreted as the
    accessible region between the two confining surface planes.

    Args:
        slit_system: Loaded slit system.
        slit_coordinates: Slit coordinates in the final slit reference frame.
        box_lengths: Orthorhombic box lengths in nm.
        padding_nm: Signed padding in nm applied on each side of the detected
            slit interval. Positive values shrink the interval and negative
            values expand it.

    Returns:
        Detected slit-plane region.

    Raises:
        ValueError: If too few candidate surface Si atoms are available or if a
            positive padding would eliminate the accessible region.
    """

    surface_silicon_indices = find_hydroxylated_surface_silicon_indices(slit_system)
    if surface_silicon_indices.size < 2:
        raise ValueError(
            "At least two hydroxylated surface Si atoms are required to infer "
            "the slit planes."
        )

    wrapped_surface_coordinates = wrap_positions(
        slit_coordinates[surface_silicon_indices],
        box_lengths,
    )

    best_axis_index = 0
    best_gap_index = 0
    best_gap_nm = -np.inf
    best_axis_values = np.empty(0, dtype=np.float64)

    for axis_index in range(3):
        axis_values = np.sort(wrapped_surface_coordinates[:, axis_index])
        periodic_gaps = np.diff(axis_values)
        wrap_gap = axis_values[0] + box_lengths[axis_index] - axis_values[-1]
        all_gaps = np.concatenate((periodic_gaps, np.array([wrap_gap], dtype=np.float64)))
        gap_index = int(np.argmax(all_gaps))
        gap_nm = float(all_gaps[gap_index])
        if gap_nm > best_gap_nm:
            best_axis_index = axis_index
            best_gap_index = gap_index
            best_gap_nm = gap_nm
            best_axis_values = axis_values

    if best_gap_index < (best_axis_values.size - 1):
        lower_plane_nm = float(best_axis_values[best_gap_index])
        upper_plane_nm = float(best_axis_values[best_gap_index + 1])
        interval_wraps = False
        accessible_width_nm = upper_plane_nm - lower_plane_nm
    else:
        lower_plane_nm = float(best_axis_values[-1])
        upper_plane_nm = float(best_axis_values[0])
        interval_wraps = True
        accessible_width_nm = upper_plane_nm + box_lengths[best_axis_index] - lower_plane_nm

    if accessible_width_nm <= (2.0 * padding_nm):
        raise ValueError(
            "The detected slit interval is narrower than twice the requested "
            "positive surface-plane padding."
        )

    return SurfacePlaneRegion(
        axis_index=best_axis_index,
        axis_name=AXIS_NAMES[best_axis_index],
        lower_plane_nm=lower_plane_nm,
        upper_plane_nm=upper_plane_nm,
        interval_wraps=interval_wraps,
        accessible_width_nm=float(accessible_width_nm),
        padding_nm=padding_nm,
        surface_si_atom_count=int(surface_silicon_indices.size),
    )


def coordinate_inside_surface_plane_region(
    coordinate_nm: float,
    box_length_nm: float,
    plane_region: SurfacePlaneRegion,
) -> bool:
    """Return whether one wrapped coordinate lies inside the slit interval.

    Args:
        coordinate_nm: Coordinate component along the slit-normal axis.
        box_length_nm: Box length along that axis in nm.
        plane_region: Detected slit interval and padding.

    Returns:
        ``True`` when the coordinate lies inside the padded accessible interval.
    """

    wrapped_coordinate_nm = float(np.mod(coordinate_nm, box_length_nm))
    tolerance_nm = 1.0e-12

    if plane_region.interval_wraps:
        lower_limit_nm = plane_region.lower_plane_nm + plane_region.padding_nm
        upper_limit_nm = plane_region.upper_plane_nm - plane_region.padding_nm
        return (
            wrapped_coordinate_nm >= (lower_limit_nm - tolerance_nm)
            or wrapped_coordinate_nm <= (upper_limit_nm + tolerance_nm)
        )

    lower_limit_nm = plane_region.lower_plane_nm + plane_region.padding_nm
    upper_limit_nm = plane_region.upper_plane_nm - plane_region.padding_nm
    return (
        lower_limit_nm - tolerance_nm
        <= wrapped_coordinate_nm
        <= upper_limit_nm + tolerance_nm
    )


def build_output_axis_permutation(normal_axis_index: int) -> tuple[int, int, int]:
    """Build the axis permutation that moves the slit normal onto ``z``.

    Args:
        normal_axis_index: Index of the detected slit-normal axis in the current
            coordinate frame.

    Returns:
        Axis permutation such that ``coordinates[:, permutation]`` produces the
        output frame with the slit normal along ``z``.

    Raises:
        ValueError: If the axis index is not one of ``0``, ``1``, or ``2``.
    """

    if normal_axis_index == 2:
        return (0, 1, 2)
    if normal_axis_index == 1:
        return (0, 2, 1)
    if normal_axis_index == 0:
        return (2, 1, 0)
    raise ValueError(f"Unsupported axis index {normal_axis_index}.")


def permute_coordinate_axes(
    coordinates: FloatArray,
    axis_permutation: tuple[int, int, int],
) -> FloatArray:
    """Permute Cartesian coordinate axes.

    Args:
        coordinates: Cartesian coordinates with shape ``(n_atoms, 3)``.
        axis_permutation: Axis permutation applied to the last coordinate axis.

    Returns:
        Coordinates expressed in the permuted frame.
    """

    return coordinates[:, axis_permutation].copy()


def permute_box_axes(
    box_lengths: FloatArray,
    axis_permutation: tuple[int, int, int],
) -> FloatArray:
    """Permute orthorhombic box lengths consistently with rotated coordinates.

    Args:
        box_lengths: Orthorhombic box lengths in nm.
        axis_permutation: Axis permutation applied to the coordinate frame.

    Returns:
        Box lengths in the permuted frame.
    """

    return box_lengths[np.array(axis_permutation, dtype=np.int32)].copy()


def apply_surface_plane_filter(
    thymol_system: GroSystem,
    translated_thymol_coordinates: FloatArray,
    selected_residue_mask: BoolArray,
    target_resname: str,
    plane_region: SurfacePlaneRegion,
    box_lengths: FloatArray,
) -> SurfacePlaneSelection:
    """Remove THY residues that fall outside the detected slit planes.

    Args:
        thymol_system: Loaded thymol system.
        translated_thymol_coordinates: THY coordinates already translated into
            the slit reference frame.
        selected_residue_mask: Residue mask after center-cropping.
        target_resname: Residue name to filter against the slit planes.
        plane_region: Detected slit interval used for the selection.
        box_lengths: Orthorhombic box lengths in nm.

    Returns:
        Updated residue selection together with the slit-plane removal mask.
    """

    filtered_residue_mask = selected_residue_mask.copy()
    removed_residue_mask = np.zeros(len(thymol_system.residue_spans), dtype=bool)
    wrapped_coordinates = wrap_positions(translated_thymol_coordinates, box_lengths)

    for residue_index, residue_span in enumerate(thymol_system.residue_spans):
        if not selected_residue_mask[residue_index] or residue_span.residue_name != target_resname:
            continue

        axis_coordinates = wrapped_coordinates[
            residue_span.start:residue_span.stop,
            plane_region.axis_index,
        ]
        if all(
            coordinate_inside_surface_plane_region(
                coordinate_nm=float(coordinate_nm),
                box_length_nm=float(box_lengths[plane_region.axis_index]),
                plane_region=plane_region,
            )
            for coordinate_nm in axis_coordinates
        ):
            continue

        filtered_residue_mask[residue_index] = False
        removed_residue_mask[residue_index] = True

    return SurfacePlaneSelection(
        selected_residue_mask=filtered_residue_mask,
        removed_residue_mask=removed_residue_mask,
        plane_region=plane_region,
    )


def unwrap_residue_coordinates(
    residue_coordinates: FloatArray,
    box_lengths: FloatArray,
) -> FloatArray:
    """Unwrap one residue relative to its first atom under orthorhombic PBC.

    Args:
        residue_coordinates: Coordinates for one residue with shape
            ``(n_atoms_in_residue, 3)``.
        box_lengths: Orthorhombic box lengths in nm.

    Returns:
        Residue coordinates made contiguous by minimum-image displacements
        relative to the first atom of the residue.
    """

    unwrapped = residue_coordinates.copy()
    reference = unwrapped[0].copy()

    for atom_index in range(1, unwrapped.shape[0]):
        displacement = unwrapped[atom_index] - reference
        displacement -= box_lengths * np.round(displacement / box_lengths)
        unwrapped[atom_index] = reference + displacement

    return unwrapped


def guess_bond_definitions(
    atom_names: tuple[str, ...],
    residue_coordinates: FloatArray,
    include_hydrogen_bonds: bool,
) -> tuple[BondDefinition, ...]:
    """Guess covalent bonds for one residue from atom names and coordinates.

    Args:
        atom_names: Ordered atom names in the residue.
        residue_coordinates: Residue coordinates in a contiguous box image.
        include_hydrogen_bonds: Whether bonds involving hydrogen atoms should be
            returned.

    Returns:
        Guessed covalent bonds for the residue.
    """

    bond_definitions: list[BondDefinition] = []
    element_symbols = [infer_element_from_atom_name(atom_name) for atom_name in atom_names]
    atom_count = len(atom_names)
    scale_factor = 1.25

    for start_atom_index in range(atom_count):
        for stop_atom_index in range(start_atom_index + 1, atom_count):
            start_element = element_symbols[start_atom_index]
            stop_element = element_symbols[stop_atom_index]
            if (
                not include_hydrogen_bonds
                and (start_element == "H" or stop_element == "H")
            ):
                continue

            distance_nm = float(
                np.linalg.norm(
                    residue_coordinates[stop_atom_index] - residue_coordinates[start_atom_index]
                )
            )
            cutoff_nm = scale_factor * (
                COVALENT_RADII_NM[start_element] + COVALENT_RADII_NM[stop_element]
            )
            if 0.05 < distance_nm <= cutoff_nm:
                bond_definitions.append(
                    BondDefinition(
                        start_atom_index=start_atom_index,
                        stop_atom_index=stop_atom_index,
                        start_atom_name=atom_names[start_atom_index],
                        stop_atom_name=atom_names[stop_atom_index],
                    )
                )

    return tuple(bond_definitions)


def guess_target_residue_bonds(
    thymol_system: GroSystem,
    target_resname: str,
    include_hydrogen_bonds: bool,
) -> tuple[BondDefinition, ...]:
    """Guess covalent bonds for the target THY residue template.

    Args:
        thymol_system: Loaded thymol system.
        target_resname: Residue name used to select the template residue.
        include_hydrogen_bonds: Whether bonds to hydrogen atoms should be kept
            in the returned template.

    Returns:
        Guessed bond template for the target residue.

    Raises:
        ValueError: If no matching residue exists or no bonds are guessed.
    """

    for residue_span in thymol_system.residue_spans:
        if residue_span.residue_name != target_resname:
            continue

        residue_coordinates = unwrap_residue_coordinates(
            thymol_system.coordinates[residue_span.start:residue_span.stop],
            thymol_system.box_lengths,
        )
        atom_names = tuple(thymol_system.atom_names[residue_span.start:residue_span.stop])
        bond_definitions = guess_bond_definitions(
            atom_names=atom_names,
            residue_coordinates=residue_coordinates,
            include_hydrogen_bonds=include_hydrogen_bonds,
        )
        if not bond_definitions:
            raise ValueError(
                f"No covalent bonds were guessed for residue name {target_resname!r}."
            )
        return bond_definitions

    raise ValueError(f"No residue named {target_resname!r} was found for bond guessing.")


def build_ring_template_from_atom_names(
    residue_name: str,
    atom_names: tuple[str, ...],
    ring_atom_prefix: str,
) -> RingTemplate | None:
    """Build a phenyl-ring template from a residue atom-name list.

    Args:
        residue_name: Residue name associated with the atom names.
        atom_names: Ordered atom names in one residue.
        ring_atom_prefix: Prefix that identifies ring atoms.

    Returns:
        A ring template when exactly six atoms match the prefix, otherwise
        ``None``.
    """

    local_atom_indices = tuple(
        atom_index for atom_index, atom_name in enumerate(atom_names) if atom_name.startswith(ring_atom_prefix)
    )
    if len(local_atom_indices) != 6:
        return None

    return RingTemplate(
        residue_name=residue_name,
        atom_prefix=ring_atom_prefix,
        local_atom_indices=local_atom_indices,
        ring_atom_names=tuple(atom_names[atom_index] for atom_index in local_atom_indices),
    )


def build_target_ring_template(
    thymol_system: GroSystem,
    target_resname: str,
    ring_atom_prefix: str,
) -> RingTemplate:
    """Build the phenyl-ring template for the target THY residue.

    Args:
        thymol_system: Loaded thymol system.
        target_resname: Residue name used to select the target residue template.
        ring_atom_prefix: Prefix that identifies ring atoms.

    Returns:
        Ring template for the target THY residue.

    Raises:
        ValueError: If no matching phenyl ring can be identified.
    """

    for residue_span in thymol_system.residue_spans:
        if residue_span.residue_name != target_resname:
            continue

        atom_names = tuple(thymol_system.atom_names[residue_span.start:residue_span.stop])
        ring_template = build_ring_template_from_atom_names(
            residue_name=target_resname,
            atom_names=atom_names,
            ring_atom_prefix=ring_atom_prefix,
        )
        if ring_template is None:
            raise ValueError(
                f"Residue name {target_resname!r} does not contain exactly six atoms "
                f"with prefix {ring_atom_prefix!r}."
            )
        return ring_template

    raise ValueError(f"No residue named {target_resname!r} was found for ring-template building.")


def compute_target_residue_mass_da(
    thymol_system: GroSystem,
    target_resname: str,
) -> float:
    """Compute the mass of one target THY residue from its atom names.

    Args:
        thymol_system: Loaded thymol system.
        target_resname: Residue name used to select the target molecule.

    Returns:
        Mass of one target residue in daltons.

    Raises:
        ValueError: If no matching residue exists.
    """

    for residue_span in thymol_system.residue_spans:
        if residue_span.residue_name != target_resname:
            continue

        return float(
            sum(
                ATOMIC_MASSES_DA[
                    infer_element_from_atom_name(thymol_system.atom_names[atom_index])
                ]
                for atom_index in range(residue_span.start, residue_span.stop)
            )
        )

    raise ValueError(f"No residue named {target_resname!r} was found for mass estimation.")


def compute_repeated_std(values: tuple[float, ...]) -> float:
    """Compute the standard deviation across repeated estimates.

    Args:
        values: Repeated estimate values.

    Returns:
        Standard deviation across repeats. Returns ``0.0`` when fewer than two
        repeats are present or when any value is non-finite.
    """

    if len(values) < 2:
        return 0.0

    array = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        return float("inf")
    return float(np.std(array, ddof=1))


def estimate_accessible_volume_nm3(
    slit_system: GroSystem,
    slit_coordinates: FloatArray,
    final_box_lengths: FloatArray,
    probe_radius_nm: float,
    sample_count: int,
    random_seed: int,
) -> tuple[float, float]:
    """Estimate slit accessible volume with one Monte Carlo repeat.

    The slit framework is treated as a union of van der Waals spheres enlarged
    by ``probe_radius_nm``. The accessible volume is the fraction of uniformly
    sampled box points that fall outside those enlarged spheres.

    Args:
        slit_system: Loaded slit system.
        slit_coordinates: Slit coordinates in the final slit reference frame.
        final_box_lengths: Final orthorhombic box lengths in nm.
        probe_radius_nm: Probe radius added to each slit atom van der Waals
            radius.
        sample_count: Number of uniformly sampled box points for this repeat.
        random_seed: Random seed used for this repeat.

    Returns:
        Two items: accessible volume in nm^3 and accessible fraction.
    """

    slit_wrapped = wrap_positions(slit_coordinates, final_box_lengths)
    exclusion_radii_nm = np.array(
        [
            VDW_RADII_NM[infer_element_from_atom_name(atom_name)] + probe_radius_nm
            for atom_name in slit_system.atom_names
        ],
        dtype=np.float64,
    )
    maximum_exclusion_radius_nm = float(np.max(exclusion_radii_nm))
    exclusion_radii_squared_nm2 = exclusion_radii_nm * exclusion_radii_nm
    slit_tree = cKDTree(slit_wrapped, boxsize=final_box_lengths)
    random_number_generator = np.random.default_rng(random_seed)

    accessible_point_count = 0
    batch_size = 10000
    box_volume_nm3 = float(np.prod(final_box_lengths))

    for batch_start in range(0, sample_count, batch_size):
        current_batch_size = min(batch_size, sample_count - batch_start)
        sample_points = random_number_generator.random((current_batch_size, 3)) * final_box_lengths
        neighbor_lists = slit_tree.query_ball_point(
            sample_points,
            r=maximum_exclusion_radius_nm,
            workers=-1,
        )
        neighbor_counts = np.fromiter(
            (len(neighbor_indices) for neighbor_indices in neighbor_lists),
            dtype=np.int32,
            count=current_batch_size,
        )
        total_neighbor_count = int(np.sum(neighbor_counts))
        if total_neighbor_count == 0:
            accessible_point_count += current_batch_size
            continue

        sample_indices = np.repeat(np.arange(current_batch_size, dtype=np.int32), neighbor_counts)
        atom_indices = np.concatenate(
            [np.asarray(neighbor_indices, dtype=np.int32) for neighbor_indices in neighbor_lists if neighbor_indices]
        )
        delta_vectors = slit_wrapped[atom_indices] - sample_points[sample_indices]
        delta_vectors -= final_box_lengths * np.round(delta_vectors / final_box_lengths)
        squared_distances_nm2 = np.einsum("ij,ij->i", delta_vectors, delta_vectors)
        excluded_pairs = squared_distances_nm2 <= exclusion_radii_squared_nm2[atom_indices]
        excluded_points = np.zeros(current_batch_size, dtype=bool)
        if np.any(excluded_pairs):
            excluded_points[np.unique(sample_indices[excluded_pairs])] = True
        accessible_point_count += int(np.count_nonzero(~excluded_points))

    accessible_fraction = accessible_point_count / sample_count
    accessible_volume_nm3 = accessible_fraction * box_volume_nm3
    return accessible_volume_nm3, accessible_fraction


def compute_density_estimate(
    thymol_system: GroSystem,
    slit_system: GroSystem,
    slit_coordinates: FloatArray,
    final_box_lengths: FloatArray,
    target_resname: str,
    remaining_thymol_molecules: int,
    probe_radii_nm: tuple[float, ...],
    sample_count: int,
    seed_count: int,
) -> DensityEstimate:
    """Compute box-average and slit-accessible thymol density estimates.

    Args:
        thymol_system: Loaded thymol system.
        slit_system: Loaded slit system.
        slit_coordinates: Slit coordinates in the final slit reference frame.
        final_box_lengths: Final orthorhombic box lengths in nm.
        target_resname: Residue name that identifies thymol molecules.
        remaining_thymol_molecules: Number of THY molecules retained in the
            merged output.
        probe_radii_nm: Probe radii in nm used for accessible-volume estimates.
        sample_count: Number of Monte Carlo points used for each repeat.
        seed_count: Number of independent repeats per probe radius.

    Returns:
        Density metrics for the final slit filling.
    """

    thymol_molecule_mass_da = compute_target_residue_mass_da(thymol_system, target_resname)
    total_thymol_mass_da = float(remaining_thymol_molecules) * thymol_molecule_mass_da
    box_volume_nm3 = float(np.prod(final_box_lengths))
    box_average_density_g_cm3 = (
        total_thymol_mass_da * GRAMS_PER_DA / (box_volume_nm3 * CUBIC_CENTIMETERS_PER_NM3)
    )

    probe_estimates: list[DensityProbeEstimate] = []
    for probe_radius_nm in probe_radii_nm:
        seed_values = tuple(int(secrets.randbits(63)) for _ in range(seed_count))
        accessible_volumes_nm3: list[float] = []
        accessible_fractions: list[float] = []
        accessible_densities_g_cm3: list[float] = []

        for seed_value in seed_values:
            accessible_volume_nm3, accessible_fraction = estimate_accessible_volume_nm3(
                slit_system=slit_system,
                slit_coordinates=slit_coordinates,
                final_box_lengths=final_box_lengths,
                probe_radius_nm=probe_radius_nm,
                sample_count=sample_count,
                random_seed=seed_value,
            )
            accessible_volumes_nm3.append(accessible_volume_nm3)
            accessible_fractions.append(accessible_fraction)

            if accessible_volume_nm3 <= 0.0:
                warnings.warn(
                    (
                        "Accessible slit volume is non-positive for density probe radius "
                        f"{probe_radius_nm:.3f} nm and seed {seed_value}; reporting infinite "
                        "accessible density for this repeat."
                    ),
                    stacklevel=2,
                )
                accessible_densities_g_cm3.append(float("inf"))
            else:
                accessible_densities_g_cm3.append(
                    total_thymol_mass_da
                    * GRAMS_PER_DA
                    / (accessible_volume_nm3 * CUBIC_CENTIMETERS_PER_NM3)
                )

        accessible_volume_values = tuple(accessible_volumes_nm3)
        accessible_fraction_values = tuple(accessible_fractions)
        accessible_density_values = tuple(accessible_densities_g_cm3)

        volume_mean_nm3 = float(np.mean(np.array(accessible_volume_values, dtype=np.float64)))
        fraction_mean = float(np.mean(np.array(accessible_fraction_values, dtype=np.float64)))
        if all(np.isfinite(value) for value in accessible_density_values):
            density_mean_g_cm3 = float(
                np.mean(np.array(accessible_density_values, dtype=np.float64))
            )
        else:
            density_mean_g_cm3 = float("inf")

        probe_estimates.append(
            DensityProbeEstimate(
                probe_radius_nm=probe_radius_nm,
                seed_values=seed_values,
                accessible_fractions=accessible_fraction_values,
                accessible_volumes_nm3=accessible_volume_values,
                accessible_densities_g_cm3=accessible_density_values,
                accessible_fraction_mean=fraction_mean,
                accessible_fraction_std=compute_repeated_std(accessible_fraction_values),
                accessible_volume_mean_nm3=volume_mean_nm3,
                accessible_volume_std_nm3=compute_repeated_std(accessible_volume_values),
                accessible_density_mean_g_cm3=density_mean_g_cm3,
                accessible_density_std_g_cm3=compute_repeated_std(accessible_density_values),
            )
        )

    return DensityEstimate(
        thymol_molecule_mass_da=thymol_molecule_mass_da,
        total_thymol_mass_da=total_thymol_mass_da,
        box_volume_nm3=box_volume_nm3,
        box_average_density_g_cm3=box_average_density_g_cm3,
        sample_count_per_seed=sample_count,
        seed_count=seed_count,
        probe_estimates=tuple(probe_estimates),
    )


def build_ring_geometry(
    residue_index: int,
    residue_name: str,
    residue_coordinates: FloatArray,
    ring_template: RingTemplate,
    final_box_lengths: FloatArray,
) -> RingGeometry:
    """Build one phenyl-ring geometry from residue coordinates and a template.

    Args:
        residue_index: Residue index in ``GroSystem.residue_spans``.
        residue_name: Residue name of the parent residue.
        residue_coordinates: Coordinates of the full residue.
        ring_template: Template that identifies the six ring atoms.
        final_box_lengths: Final orthorhombic box lengths in nm.

    Returns:
        Geometric model of the phenyl ring.
    """

    ring_coordinates = unwrap_residue_coordinates(
        residue_coordinates[np.array(ring_template.local_atom_indices, dtype=np.int32)],
        final_box_lengths,
    )
    ring_center = np.mean(ring_coordinates, axis=0)
    wrapped_center = wrap_positions(ring_center[np.newaxis, :], final_box_lengths)[0]
    image_shift = ring_center - wrapped_center
    ring_coordinates = ring_coordinates - image_shift
    ring_center = ring_center - image_shift

    centered_coordinates = ring_coordinates - ring_center
    _, _, right_singular_vectors = np.linalg.svd(centered_coordinates, full_matrices=False)
    basis_u = right_singular_vectors[0] / np.linalg.norm(right_singular_vectors[0])
    basis_v = right_singular_vectors[1] / np.linalg.norm(right_singular_vectors[1])
    normal = right_singular_vectors[2] / np.linalg.norm(right_singular_vectors[2])

    ring_coordinates_2d = np.column_stack(
        (centered_coordinates @ basis_u, centered_coordinates @ basis_v)
    )
    angles = np.arctan2(ring_coordinates_2d[:, 1], ring_coordinates_2d[:, 0])
    order = np.argsort(angles)
    polygon_2d = ring_coordinates_2d[order]
    max_radius_nm = float(np.max(np.linalg.norm(polygon_2d, axis=1)))

    return RingGeometry(
        residue_index=residue_index,
        residue_name=residue_name,
        center=ring_center,
        wrapped_center=wrapped_center,
        normal=normal,
        basis_u=basis_u,
        basis_v=basis_v,
        polygon_2d=polygon_2d,
        max_radius_nm=max_radius_nm,
    )


def build_slit_ring_geometries(
    slit_system: GroSystem,
    slit_coordinates: FloatArray,
    final_box_lengths: FloatArray,
    ring_atom_prefix: str,
) -> tuple[RingGeometry, ...]:
    """Build phenyl-ring geometries for slit residues that contain one ring.

    Args:
        slit_system: Loaded slit system.
        slit_coordinates: Slit coordinates in the final slit reference frame.
        final_box_lengths: Final orthorhombic box lengths in nm.
        ring_atom_prefix: Prefix that identifies ring atoms.

    Returns:
        Slit phenyl-ring geometries.
    """

    ring_geometries: list[RingGeometry] = []
    for residue_index, residue_span in enumerate(slit_system.residue_spans):
        atom_names = tuple(slit_system.atom_names[residue_span.start:residue_span.stop])
        ring_template = build_ring_template_from_atom_names(
            residue_name=residue_span.residue_name,
            atom_names=atom_names,
            ring_atom_prefix=ring_atom_prefix,
        )
        if ring_template is None:
            continue

        residue_coordinates = slit_coordinates[residue_span.start:residue_span.stop]
        ring_geometries.append(
            build_ring_geometry(
                residue_index=residue_index,
                residue_name=residue_span.residue_name,
                residue_coordinates=residue_coordinates,
                ring_template=ring_template,
                final_box_lengths=final_box_lengths,
            )
        )

    return tuple(ring_geometries)


def build_target_ring_geometries(
    thymol_system: GroSystem,
    translated_thymol_coordinates: FloatArray,
    selected_residue_mask: BoolArray,
    target_resname: str,
    target_ring_template: RingTemplate,
    final_box_lengths: FloatArray,
) -> tuple[RingGeometry, ...]:
    """Build THY phenyl-ring geometries for the selected THY residues.

    Args:
        thymol_system: Loaded thymol system.
        translated_thymol_coordinates: THY coordinates in the slit reference
            frame.
        selected_residue_mask: Residue mask that marks THY residues selected for
            clash checking.
        target_resname: Target THY residue name.
        target_ring_template: Ring template for THY residues.
        final_box_lengths: Final orthorhombic box lengths in nm.

    Returns:
        Phenyl-ring geometries for the selected THY residues.
    """

    ring_geometries: list[RingGeometry] = []
    for residue_index, residue_span in enumerate(thymol_system.residue_spans):
        if not selected_residue_mask[residue_index] or residue_span.residue_name != target_resname:
            continue

        residue_coordinates = translated_thymol_coordinates[residue_span.start:residue_span.stop]
        ring_geometries.append(
            build_ring_geometry(
                residue_index=residue_index,
                residue_name=residue_span.residue_name,
                residue_coordinates=residue_coordinates,
                ring_template=target_ring_template,
                final_box_lengths=final_box_lengths,
            )
        )

    return tuple(ring_geometries)


def build_slit_bond_geometries(
    slit_system: GroSystem,
    slit_coordinates: FloatArray,
    final_box_lengths: FloatArray,
) -> tuple[tuple[ResidueBondTemplate, ...], tuple[BondSegmentGeometry, ...]]:
    """Build cached slit bond templates and bond-segment geometries.

    Args:
        slit_system: Loaded slit system.
        slit_coordinates: Slit coordinates in the final slit reference frame.
        final_box_lengths: Final orthorhombic box lengths in nm.

    Returns:
        Two items: the unique cached slit bond templates and the geometric bond
        segments built for all slit residues.
    """

    template_cache: dict[tuple[str, tuple[str, ...]], ResidueBondTemplate] = {}
    bond_geometries: list[BondSegmentGeometry] = []

    for residue_index, residue_span in enumerate(slit_system.residue_spans):
        atom_names = tuple(slit_system.atom_names[residue_span.start:residue_span.stop])
        template_key = (residue_span.residue_name, atom_names)
        residue_coordinates = unwrap_residue_coordinates(
            slit_coordinates[residue_span.start:residue_span.stop],
            final_box_lengths,
        )

        residue_bond_template = template_cache.get(template_key)
        if residue_bond_template is None:
            residue_bond_template = ResidueBondTemplate(
                residue_name=residue_span.residue_name,
                atom_names=atom_names,
                bond_definitions=guess_bond_definitions(
                    atom_names=atom_names,
                    residue_coordinates=residue_coordinates,
                    include_hydrogen_bonds=True,
                ),
            )
            template_cache[template_key] = residue_bond_template

        for bond_definition in residue_bond_template.bond_definitions:
            start_point = residue_coordinates[bond_definition.start_atom_index]
            stop_point = residue_coordinates[bond_definition.stop_atom_index]
            midpoint = 0.5 * (start_point + stop_point)
            wrapped_midpoint = wrap_positions(midpoint[np.newaxis, :], final_box_lengths)[0]
            bond_geometries.append(
                BondSegmentGeometry(
                    residue_index=residue_index,
                    residue_name=residue_span.residue_name,
                    start_atom_name=bond_definition.start_atom_name,
                    stop_atom_name=bond_definition.stop_atom_name,
                    start_point=start_point,
                    stop_point=stop_point,
                    midpoint=midpoint,
                    wrapped_midpoint=wrapped_midpoint,
                    half_length_nm=0.5 * float(np.linalg.norm(stop_point - start_point)),
                )
            )

    return tuple(template_cache.values()), tuple(bond_geometries)


def point_inside_polygon_with_padding(
    point_2d: FloatArray,
    polygon_2d: FloatArray,
    padding_nm: float,
) -> bool:
    """Check whether a 2D point lies inside or very near a polygon.

    Args:
        point_2d: Query point in the ring-plane basis.
        polygon_2d: Ordered 2D polygon vertices.
        padding_nm: Allowed distance outside the polygon boundary.

    Returns:
        ``True`` when the point lies inside the polygon or within
        ``padding_nm`` of its boundary.
    """

    inside = False
    point_x = float(point_2d[0])
    point_y = float(point_2d[1])
    vertex_count = polygon_2d.shape[0]

    for vertex_index in range(vertex_count):
        next_index = (vertex_index + 1) % vertex_count
        x1, y1 = polygon_2d[vertex_index]
        x2, y2 = polygon_2d[next_index]
        intersects = ((y1 > point_y) != (y2 > point_y)) and (
            point_x < (x2 - x1) * (point_y - y1) / (y2 - y1 + 1.0e-12) + x1
        )
        if intersects:
            inside = not inside

    if inside or padding_nm <= 0.0:
        return inside

    minimum_distance_nm = np.inf
    for vertex_index in range(vertex_count):
        next_index = (vertex_index + 1) % vertex_count
        segment_start = polygon_2d[vertex_index]
        segment_stop = polygon_2d[next_index]
        segment = segment_stop - segment_start
        denominator = float(np.dot(segment, segment))
        if denominator <= 1.0e-12:
            projection = segment_start
        else:
            fraction = float(
                np.clip(
                    np.dot(point_2d - segment_start, segment) / denominator,
                    0.0,
                    1.0,
                )
            )
            projection = segment_start + fraction * segment
        minimum_distance_nm = min(
            minimum_distance_nm,
            float(np.linalg.norm(point_2d - projection)),
        )

    return minimum_distance_nm <= padding_nm


def shift_residue_near_reference(
    residue_coordinates: FloatArray,
    reference_point: FloatArray,
    box_lengths: FloatArray,
) -> FloatArray:
    """Shift one residue by a whole-box image so it sits near a reference point.

    Args:
        residue_coordinates: Contiguous residue coordinates.
        reference_point: Point that the shifted residue should lie near.
        box_lengths: Orthorhombic box lengths in nm.

    Returns:
        Shifted residue coordinates that preserve the original residue geometry.
    """

    residue_center = np.mean(residue_coordinates, axis=0)
    image_shift = box_lengths * np.round((residue_center - reference_point) / box_lengths)
    return residue_coordinates - image_shift


def shift_bond_near_reference(
    start_point: FloatArray,
    stop_point: FloatArray,
    reference_point: FloatArray,
    box_lengths: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Shift one bond by a whole-box image so it sits near a reference point.

    Args:
        start_point: Start point of the bond segment.
        stop_point: End point of the bond segment.
        reference_point: Point that the shifted bond should lie near.
        box_lengths: Orthorhombic box lengths in nm.

    Returns:
        Start and stop points of the shifted bond segment.
    """

    midpoint = 0.5 * (start_point + stop_point)
    image_shift = box_lengths * np.round((midpoint - reference_point) / box_lengths)
    return start_point - image_shift, stop_point - image_shift


def bond_crosses_ring(
    bond_start: FloatArray,
    bond_stop: FloatArray,
    ring_geometry: RingGeometry,
    plane_tolerance_nm: float,
    polygon_padding_nm: float,
) -> bool:
    """Check whether a bond passes through a phenyl-ring polygon.

    Args:
        bond_start: Start point of the bond segment in nm.
        bond_stop: End point of the bond segment in nm.
        ring_geometry: Geometric model of the phenyl ring.
        plane_tolerance_nm: Allowed distance from the ring plane in nm.
        polygon_padding_nm: In-plane padding around the ring polygon in nm.

    Returns:
        ``True`` when the bond comes close enough to the ring plane and projects
        inside the phenyl polygon.
    """

    bond_direction = bond_stop - bond_start
    denominator = float(np.dot(ring_geometry.normal, bond_direction))
    if abs(denominator) > 1.0e-12:
        fraction = float(
            np.clip(
                np.dot(ring_geometry.normal, ring_geometry.center - bond_start) / denominator,
                0.0,
                1.0,
            )
        )
    else:
        start_distance = abs(float(np.dot(bond_start - ring_geometry.center, ring_geometry.normal)))
        stop_distance = abs(float(np.dot(bond_stop - ring_geometry.center, ring_geometry.normal)))
        fraction = 0.0 if start_distance <= stop_distance else 1.0

    closest_point = bond_start + fraction * bond_direction
    plane_distance_nm = abs(
        float(np.dot(closest_point - ring_geometry.center, ring_geometry.normal))
    )
    if plane_distance_nm > plane_tolerance_nm:
        return False

    projected_point = closest_point - ring_geometry.center
    projected_point_2d = np.array(
        [
            np.dot(projected_point, ring_geometry.basis_u),
            np.dot(projected_point, ring_geometry.basis_v),
        ],
        dtype=np.float64,
    )
    if np.linalg.norm(projected_point_2d) > (ring_geometry.max_radius_nm + polygon_padding_nm):
        return False

    return point_inside_polygon_with_padding(
        point_2d=projected_point_2d,
        polygon_2d=ring_geometry.polygon_2d,
        padding_nm=polygon_padding_nm,
    )


def center_crop_thymol_residues(
    thymol_system: GroSystem,
    final_box_lengths: FloatArray,
) -> tuple[FloatArray, BoolArray, FloatArray]:
    """Center-crop thymol residues from the larger reservoir box to the slit cell.

    A residue is kept only when all of its atoms fit inside the centered crop
    window after residue-level unwrapping in the original thymol box.

    Args:
        thymol_system: Loaded thymol system that acts as the reservoir.
        final_box_lengths: Target slit box lengths in nm.

    Returns:
        Three items: translated thymol coordinates in the slit reference frame,
        a boolean residue mask marking residues fully inside the crop, and the
        lower corner of the centered crop window in the original thymol box.
    """

    crop_window_start = 0.5 * (thymol_system.box_lengths - final_box_lengths)
    crop_window_stop = crop_window_start + final_box_lengths
    translated_coordinates = np.zeros_like(thymol_system.coordinates)
    selected_residues = np.zeros(len(thymol_system.residue_spans), dtype=bool)
    tolerance = 1.0e-6

    for residue_index, residue_span in enumerate(thymol_system.residue_spans):
        residue_coordinates = unwrap_residue_coordinates(
            thymol_system.coordinates[residue_span.start:residue_span.stop],
            thymol_system.box_lengths,
        )
        is_inside_crop = bool(
            np.all(residue_coordinates >= (crop_window_start - tolerance))
            and np.all(residue_coordinates < (crop_window_stop + tolerance))
        )
        if is_inside_crop:
            translated_coordinates[residue_span.start:residue_span.stop] = (
                residue_coordinates - crop_window_start
            )
            selected_residues[residue_index] = True

    return translated_coordinates, selected_residues, crop_window_start


def identify_clashing_target_residues(
    thymol_system: GroSystem,
    slit_system: GroSystem,
    translated_thymol_coordinates: FloatArray,
    selected_residue_mask: BoolArray,
    slit_coordinates: FloatArray,
    final_box_lengths: FloatArray,
    target_resname: str,
    general_cutoff_nm: float,
    ring_atom_prefix: str,
    ring_plane_tolerance_nm: float,
    ring_polygon_padding_nm: float,
    include_hydrogen_bonds_in_ring_check: bool,
) -> tuple[ClashSelection, RingCheckCache]:
    """Mark target residues that clash with the slit structure.

    The distance calculation uses periodic minimum-image distances in the final
    orthorhombic box. A target residue is removed when any of its atoms comes
    within ``general_cutoff_nm`` of any slit atom, when a THY bond crosses a
    slit phenyl ring, or when a slit bond crosses a THY phenyl ring.

    Args:
        thymol_system: Loaded thymol system.
        slit_system: Loaded slit system.
        translated_thymol_coordinates: THY coordinates already translated into
            the slit reference frame.
        selected_residue_mask: Boolean mask over THY residues where ``True``
            means the residue survived the geometric prefilters applied before
            clash detection.
        slit_coordinates: Slit coordinates in the final slit reference frame.
        final_box_lengths: Box lengths used for the merged output.
        target_resname: Residue name to remove on clash.
        general_cutoff_nm: Lower all-atom cutoff in nm.
        ring_atom_prefix: Prefix that identifies phenyl-ring atoms.
        ring_plane_tolerance_nm: Allowed out-of-plane distance for the explicit
            ring-crossing tests.
        ring_polygon_padding_nm: In-plane polygon padding for the ring checks.
        include_hydrogen_bonds_in_ring_check: Whether THY bonds to hydrogen
            atoms should participate in the explicit ring-crossing tests.

    Returns:
        Two items: residue-level clash masks and cached ring/bond geometries
        reused later in reporting.
    """

    residue_count = len(thymol_system.residue_spans)
    removed_by_general = np.zeros(residue_count, dtype=bool)
    removed_by_forward_ring = np.zeros(residue_count, dtype=bool)
    removed_by_reverse_ring = np.zeros(residue_count, dtype=bool)

    slit_wrapped = wrap_positions(slit_coordinates, final_box_lengths)
    slit_tree = cKDTree(slit_wrapped, boxsize=final_box_lengths)

    candidate_atoms = np.zeros(thymol_system.atom_count, dtype=bool)
    for residue_index, residue_span in enumerate(thymol_system.residue_spans):
        if selected_residue_mask[residue_index] and residue_span.residue_name == target_resname:
            candidate_atoms[residue_span.start:residue_span.stop] = True

    if np.any(candidate_atoms):
        candidate_atom_indices = np.flatnonzero(candidate_atoms)
        wrapped_candidate_coordinates = wrap_positions(
            translated_thymol_coordinates[candidate_atom_indices],
            final_box_lengths,
        )
        nearest_distances, _ = slit_tree.query(
            wrapped_candidate_coordinates,
            k=1,
            workers=-1,
        )
        clashing_candidate_atoms = np.zeros(thymol_system.atom_count, dtype=bool)
        clashing_candidate_atoms[candidate_atom_indices] = nearest_distances <= general_cutoff_nm
        if np.any(clashing_candidate_atoms):
            residue_indices = thymol_system.atom_to_residue_index[clashing_candidate_atoms]
            removed_by_general[np.unique(residue_indices)] = True

    target_bond_template = guess_target_residue_bonds(
        thymol_system=thymol_system,
        target_resname=target_resname,
        include_hydrogen_bonds=include_hydrogen_bonds_in_ring_check,
    )
    target_ring_template = build_target_ring_template(
        thymol_system=thymol_system,
        target_resname=target_resname,
        ring_atom_prefix=ring_atom_prefix,
    )
    target_ring_geometries = build_target_ring_geometries(
        thymol_system=thymol_system,
        translated_thymol_coordinates=translated_thymol_coordinates,
        selected_residue_mask=selected_residue_mask,
        target_resname=target_resname,
        target_ring_template=target_ring_template,
        final_box_lengths=final_box_lengths,
    )
    slit_ring_geometries = build_slit_ring_geometries(
        slit_system=slit_system,
        slit_coordinates=slit_coordinates,
        final_box_lengths=final_box_lengths,
        ring_atom_prefix=ring_atom_prefix,
    )
    slit_bond_templates, slit_bond_geometries = build_slit_bond_geometries(
        slit_system=slit_system,
        slit_coordinates=slit_coordinates,
        final_box_lengths=final_box_lengths,
    )
    ring_check_cache = RingCheckCache(
        slit_ring_geometries=slit_ring_geometries,
        target_bond_template=target_bond_template,
        target_ring_template=target_ring_template,
        target_ring_geometries=target_ring_geometries,
        slit_bond_templates=slit_bond_templates,
        slit_bond_geometries=slit_bond_geometries,
    )

    if slit_ring_geometries:
        slit_ring_center_tree = cKDTree(
            np.array([ring_geometry.wrapped_center for ring_geometry in slit_ring_geometries]),
            boxsize=final_box_lengths,
        )
        maximum_slit_ring_radius_nm = max(
            ring_geometry.max_radius_nm for ring_geometry in slit_ring_geometries
        )
        template_span = next(
            residue_span
            for residue_span in thymol_system.residue_spans
            if residue_span.residue_name == target_resname
        )
        template_coordinates = unwrap_residue_coordinates(
            thymol_system.coordinates[template_span.start:template_span.stop],
            thymol_system.box_lengths,
        )
        maximum_target_bond_length_nm = max(
            float(
                np.linalg.norm(
                    template_coordinates[bond_definition.stop_atom_index]
                    - template_coordinates[bond_definition.start_atom_index]
                )
            )
            for bond_definition in target_bond_template
        )

        for residue_index, residue_span in enumerate(thymol_system.residue_spans):
            if not selected_residue_mask[residue_index] or residue_span.residue_name != target_resname:
                continue

            residue_coordinates = translated_thymol_coordinates[residue_span.start:residue_span.stop]
            residue_center = np.mean(residue_coordinates, axis=0)
            residue_center_wrapped = wrap_positions(
                residue_center[np.newaxis, :],
                final_box_lengths,
            )[0]
            residue_radius_nm = float(
                np.max(np.linalg.norm(residue_coordinates - residue_center, axis=1))
            )
            query_radius_nm = (
                residue_radius_nm
                + maximum_slit_ring_radius_nm
                + maximum_target_bond_length_nm
                + ring_plane_tolerance_nm
                + ring_polygon_padding_nm
            )
            candidate_ring_indices = slit_ring_center_tree.query_ball_point(
                residue_center_wrapped,
                r=query_radius_nm,
            )
            if not candidate_ring_indices:
                continue

            for ring_index in candidate_ring_indices:
                ring_geometry = slit_ring_geometries[ring_index]
                residue_near_ring = shift_residue_near_reference(
                    residue_coordinates=residue_coordinates,
                    reference_point=ring_geometry.center,
                    box_lengths=final_box_lengths,
                )
                for bond_definition in target_bond_template:
                    if bond_crosses_ring(
                        bond_start=residue_near_ring[bond_definition.start_atom_index],
                        bond_stop=residue_near_ring[bond_definition.stop_atom_index],
                        ring_geometry=ring_geometry,
                        plane_tolerance_nm=ring_plane_tolerance_nm,
                        polygon_padding_nm=ring_polygon_padding_nm,
                    ):
                        removed_by_forward_ring[residue_index] = True
                        break
                if removed_by_forward_ring[residue_index]:
                    break

    if target_ring_geometries and slit_bond_geometries:
        slit_bond_midpoint_tree = cKDTree(
            np.array([bond_geometry.wrapped_midpoint for bond_geometry in slit_bond_geometries]),
            boxsize=final_box_lengths,
        )
        maximum_slit_bond_half_length_nm = max(
            bond_geometry.half_length_nm for bond_geometry in slit_bond_geometries
        )

        for target_ring_geometry in target_ring_geometries:
            candidate_bond_indices = slit_bond_midpoint_tree.query_ball_point(
                target_ring_geometry.wrapped_center,
                r=(
                    target_ring_geometry.max_radius_nm
                    + maximum_slit_bond_half_length_nm
                    + ring_plane_tolerance_nm
                    + ring_polygon_padding_nm
                ),
            )
            if not candidate_bond_indices:
                continue

            for bond_index in candidate_bond_indices:
                bond_geometry = slit_bond_geometries[bond_index]
                shifted_start_point, shifted_stop_point = shift_bond_near_reference(
                    start_point=bond_geometry.start_point,
                    stop_point=bond_geometry.stop_point,
                    reference_point=target_ring_geometry.center,
                    box_lengths=final_box_lengths,
                )
                if bond_crosses_ring(
                    bond_start=shifted_start_point,
                    bond_stop=shifted_stop_point,
                    ring_geometry=target_ring_geometry,
                    plane_tolerance_nm=ring_plane_tolerance_nm,
                    polygon_padding_nm=ring_polygon_padding_nm,
                ):
                    removed_by_reverse_ring[target_ring_geometry.residue_index] = True
                    break

    removed_by_any_ring = removed_by_forward_ring | removed_by_reverse_ring
    clash_selection = ClashSelection(
        removed_residue_mask=removed_by_general | removed_by_any_ring,
        removed_by_general_mask=removed_by_general,
        removed_by_forward_ring_mask=removed_by_forward_ring,
        removed_by_reverse_ring_mask=removed_by_reverse_ring,
        removed_by_any_ring_mask=removed_by_any_ring,
    )
    return clash_selection, ring_check_cache


def build_kept_thymol_atom_mask(
    thymol_system: GroSystem,
    selected_residue_mask: BoolArray,
    removed_residue_mask: BoolArray,
) -> BoolArray:
    """Build an atom mask for residues that survive the full THY filtering flow.

    Args:
        thymol_system: Loaded thymol system.
        selected_residue_mask: Boolean mask over residue spans where ``True``
            marks residues that survived the geometric prefilters.
        removed_residue_mask: Boolean mask over residue spans where ``True``
            marks residues to drop after clash filtering.

    Returns:
        Boolean mask over THY atoms where ``True`` marks atoms to keep.
    """

    keep_mask = np.zeros(thymol_system.atom_count, dtype=bool)
    for residue_index, residue_span in enumerate(thymol_system.residue_spans):
        if selected_residue_mask[residue_index] and not removed_residue_mask[residue_index]:
            keep_mask[residue_span.start:residue_span.stop] = True
    return keep_mask


def format_gro_atom_line(
    residue_id: int,
    residue_name: str,
    atom_name: str,
    atom_id: int,
    coordinate: FloatArray,
    velocity: FloatArray | None,
) -> str:
    """Format one atom line in GRO syntax.

    Args:
        residue_id: Residue identifier to write.
        residue_name: Residue name to write.
        atom_name: Atom name to write.
        atom_id: Atom identifier to write.
        coordinate: Cartesian coordinates in nm.
        velocity: Optional Cartesian velocities in nm/ps.

    Returns:
        One GRO-formatted atom line.
    """

    line = (
        f"{residue_id % 100000:5d}"
        f"{residue_name[:5]:<5}"
        f"{atom_name[:5]:>5}"
        f"{atom_id % 100000:5d}"
        f"{coordinate[0]:8.3f}"
        f"{coordinate[1]:8.3f}"
        f"{coordinate[2]:8.3f}"
    )
    if velocity is not None:
        line += f"{velocity[0]:8.4f}{velocity[1]:8.4f}{velocity[2]:8.4f}"
    return line + "\n"


def write_system_atoms(
    handle: TextIO,
    system: GroSystem,
    coordinates: FloatArray,
    keep_atom_mask: BoolArray | None,
    write_velocities: bool,
    starting_residue_id: int,
    starting_atom_id: int,
) -> tuple[int, int]:
    """Write selected atoms from one system to an open GRO file.

    Args:
        handle: Open text handle for the output GRO file.
        system: Source system.
        coordinates: Coordinates to write for the source system.
        keep_atom_mask: Optional boolean atom mask. Atoms with ``False`` are
            skipped. When the mask removes at least one atom of a residue, the
            entire residue is skipped.
        write_velocities: Whether the output file should include velocities.
        starting_residue_id: First residue identifier to write.
        starting_atom_id: First atom identifier to write.

    Returns:
        Two items: the next available residue identifier and the next available
        atom identifier after writing this system.
    """

    residue_id = starting_residue_id
    atom_id = starting_atom_id

    for residue_span in system.residue_spans:
        if keep_atom_mask is not None and not np.all(
            keep_atom_mask[residue_span.start:residue_span.stop]
        ):
            continue

        for atom_index in range(residue_span.start, residue_span.stop):
            velocity = None
            if write_velocities:
                if system.velocities is None:
                    velocity = np.zeros(3, dtype=np.float64)
                else:
                    velocity = system.velocities[atom_index]

            handle.write(
                format_gro_atom_line(
                    residue_id=residue_id,
                    residue_name=system.residue_names[atom_index],
                    atom_name=system.atom_names[atom_index],
                    atom_id=atom_id,
                    coordinate=coordinates[atom_index],
                    velocity=velocity,
                )
            )
            atom_id += 1

        residue_id += 1

    return residue_id, atom_id


def write_merged_gro(
    config: MergeConfig,
    slit_system: GroSystem,
    slit_coordinates: FloatArray,
    thymol_system: GroSystem,
    thymol_coordinates: FloatArray,
    kept_thymol_mask: BoolArray,
    final_box_lengths: FloatArray,
) -> tuple[int, int]:
    """Write the merged GRO file and return the written counts.

    Args:
        config: Merge configuration.
        slit_system: Slit structure to write first.
        slit_coordinates: Slit coordinates in the final slit reference frame.
        thymol_system: THY structure to write after filtering.
        thymol_coordinates: THY coordinates after optional residue wrapping.
        kept_thymol_mask: Boolean mask over THY atoms to keep.
        final_box_lengths: Orthorhombic box lengths written to the output.

    Returns:
        Two items: final atom count and final residue count.
    """

    final_atom_count = slit_system.atom_count + int(np.count_nonzero(kept_thymol_mask))
    write_velocities = slit_system.velocities is not None or thymol_system.velocities is not None

    final_residue_count = len(slit_system.residue_spans)
    for residue_span in thymol_system.residue_spans:
        if np.all(kept_thymol_mask[residue_span.start:residue_span.stop]):
            final_residue_count += 1

    with config.output_path.open("w", encoding="utf-8") as handle:
        handle.write("Merged slit + ring-check filtered thymol\n")
        handle.write(f"{final_atom_count}\n")

        next_residue_id, next_atom_id = write_system_atoms(
            handle=handle,
            system=slit_system,
            coordinates=slit_coordinates,
            keep_atom_mask=None,
            write_velocities=write_velocities,
            starting_residue_id=1,
            starting_atom_id=1,
        )
        write_system_atoms(
            handle=handle,
            system=thymol_system,
            coordinates=thymol_coordinates,
            keep_atom_mask=kept_thymol_mask,
            write_velocities=write_velocities,
            starting_residue_id=next_residue_id,
            starting_atom_id=next_atom_id,
        )
        handle.write(
            f"{final_box_lengths[0]:10.5f}{final_box_lengths[1]:10.5f}{final_box_lengths[2]:10.5f}\n"
        )

    return final_atom_count, final_residue_count


def build_log_text(config: MergeConfig, report: MergeReport) -> str:
    """Build the plain-text merge summary written to the log file.

    Args:
        config: Merge configuration used for the run.
        report: Merge statistics gathered during the run.

    Returns:
        Human-readable text log.
    """

    lines = [
        "Merged slit + ring-check filtered thymol",
        f"thymol_input = {config.thymol_path}",
        f"slit_input = {config.slit_path}",
        f"output_gro = {config.output_path}",
        f"log_file = {config.log_path}",
        f"target_residue = {config.thymol_resname}",
        f"general_cutoff_nm = {config.general_cutoff_nm:.3f}",
        f"ring_atom_prefix = {config.ring_atom_prefix}",
        f"ring_plane_tolerance_nm = {config.ring_plane_tolerance_nm:.3f}",
        f"ring_polygon_padding_nm = {config.ring_polygon_padding_nm:.3f}",
        f"include_hydrogen_bonds_in_ring_check = {config.include_hydrogen_bonds_in_ring_check}",
        f"surface_plane_filter_enabled = {config.use_surface_plane_filter}",
        f"surface_plane_padding_nm = {config.surface_plane_padding_nm:.3f}",
        "output_axis_order = {axes}".format(
            axes=" ".join(AXIS_NAMES[axis_index] for axis_index in report.output_axis_permutation)
        ),
        "output_surface_normal_axis = z",
        f"crop_window_start_nm = {report.crop_window_start_nm[0]:.5f} {report.crop_window_start_nm[1]:.5f} {report.crop_window_start_nm[2]:.5f}",
        f"output_box_nm = {report.output_box_nm[0]:.5f} {report.output_box_nm[1]:.5f} {report.output_box_nm[2]:.5f}",
        f"initial_thymol_molecules = {report.initial_thymol_molecules}",
        f"cropped_thymol_molecules = {report.cropped_thymol_molecules}",
        f"removed_outside_crop_thymol_molecules = {report.removed_outside_crop_thymol_molecules}",
        f"surface_plane_filtered_thymol_molecules = {report.surface_plane_filtered_thymol_molecules}",
        f"removed_by_surface_plane_thymol_molecules = {report.removed_by_surface_plane_thymol_molecules}",
        f"removed_by_general_cutoff_thymol_molecules = {report.removed_by_general_cutoff_thymol_molecules}",
        f"removed_by_forward_ring_thymol_molecules = {report.removed_by_forward_ring_thymol_molecules}",
        f"removed_by_reverse_ring_thymol_molecules = {report.removed_by_reverse_ring_thymol_molecules}",
        f"removed_by_any_ring_thymol_molecules = {report.removed_by_any_ring_thymol_molecules}",
        f"removed_by_general_only_thymol_molecules = {report.removed_by_general_only_thymol_molecules}",
        f"removed_by_forward_ring_only_thymol_molecules = {report.removed_by_forward_ring_only_thymol_molecules}",
        f"removed_by_reverse_ring_only_thymol_molecules = {report.removed_by_reverse_ring_only_thymol_molecules}",
        f"removed_by_general_and_forward_ring_only_thymol_molecules = {report.removed_by_general_and_forward_ring_only_thymol_molecules}",
        f"removed_by_general_and_reverse_ring_only_thymol_molecules = {report.removed_by_general_and_reverse_ring_only_thymol_molecules}",
        f"removed_by_forward_and_reverse_ring_only_thymol_molecules = {report.removed_by_forward_and_reverse_ring_only_thymol_molecules}",
        f"removed_by_general_and_forward_and_reverse_ring_thymol_molecules = {report.removed_by_general_and_forward_and_reverse_ring_thymol_molecules}",
        f"removed_by_clash_thymol_molecules = {report.removed_by_clash_thymol_molecules}",
        f"removed_thymol_molecules = {report.removed_thymol_molecules}",
        f"remaining_thymol_molecules = {report.remaining_thymol_molecules}",
        f"slit_phenyl_ring_count = {report.slit_phenyl_ring_count}",
        f"cropped_thymol_ring_count = {report.cropped_thymol_ring_count}",
        f"thymol_bonds_checked_per_molecule = {report.thymol_bonds_checked_per_molecule}",
        f"slit_bond_template_count = {report.slit_bond_template_count}",
        f"slit_bond_count_checked = {report.slit_bond_count_checked}",
        f"thymol_molecule_mass_da = {report.density_estimate.thymol_molecule_mass_da:.5f}",
        f"total_thymol_mass_da = {report.density_estimate.total_thymol_mass_da:.5f}",
        f"box_volume_nm3 = {report.density_estimate.box_volume_nm3:.5f}",
        f"box_average_thymol_density_g_cm3 = {report.density_estimate.box_average_density_g_cm3:.5f}",
        f"density_sample_count_per_seed = {report.density_estimate.sample_count_per_seed}",
        f"density_seed_count = {report.density_estimate.seed_count}",
    ]

    if report.surface_plane_region is None:
        lines.append("surface_plane_axis = none")
    else:
        lines.extend(
            [
                f"surface_plane_axis = {report.surface_plane_region.axis_name}",
                f"surface_plane_axis_index = {report.surface_plane_region.axis_index}",
                f"surface_plane_lower_nm = {report.surface_plane_region.lower_plane_nm:.5f}",
                f"surface_plane_upper_nm = {report.surface_plane_region.upper_plane_nm:.5f}",
                f"surface_plane_interval_wraps = {report.surface_plane_region.interval_wraps}",
                f"surface_plane_accessible_width_nm = {report.surface_plane_region.accessible_width_nm:.5f}",
                f"surface_plane_surface_si_atom_count = {report.surface_plane_region.surface_si_atom_count}",
            ]
        )

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

    lines.extend(
        [
            f"slit_atom_count = {report.slit_atom_count}",
            f"final_atom_count = {report.final_atom_count}",
            f"final_residue_count = {report.final_residue_count}",
            f"output_wrapped_into_box = {config.wrap_output}",
        ]
    )
    return "\n".join(lines) + "\n"


def merge_systems(config: MergeConfig) -> MergeReport:
    """Run the full merge workflow for the provided configuration.

    Args:
        config: Merge settings and output paths.

    Returns:
        Merge statistics collected while building the merged GRO file.
    """

    thymol_system = load_gro_system(config.thymol_path)
    slit_system = load_gro_system(config.slit_path)
    validate_slit_coordinates(slit_system)
    validate_config(config, thymol_system, slit_system)

    final_box_lengths = slit_system.box_lengths.copy()
    centered_slit_coordinates = slit_system.coordinates.copy()
    translated_thymol_coordinates, cropped_residue_mask, crop_window_start = center_crop_thymol_residues(
        thymol_system=thymol_system,
        final_box_lengths=final_box_lengths,
    )

    surface_plane_region = infer_slit_surface_plane_region(
        slit_system=slit_system,
        slit_coordinates=centered_slit_coordinates,
        box_lengths=final_box_lengths,
        padding_nm=config.surface_plane_padding_nm,
    )
    surface_plane_removed_mask = np.zeros(len(thymol_system.residue_spans), dtype=bool)
    selected_residue_mask = cropped_residue_mask
    if config.use_surface_plane_filter:
        surface_plane_selection = apply_surface_plane_filter(
            thymol_system=thymol_system,
            translated_thymol_coordinates=translated_thymol_coordinates,
            selected_residue_mask=cropped_residue_mask,
            target_resname=config.thymol_resname,
            plane_region=surface_plane_region,
            box_lengths=final_box_lengths,
        )
        selected_residue_mask = surface_plane_selection.selected_residue_mask
        surface_plane_removed_mask = surface_plane_selection.removed_residue_mask

    clash_selection, ring_check_cache = identify_clashing_target_residues(
        thymol_system=thymol_system,
        slit_system=slit_system,
        translated_thymol_coordinates=translated_thymol_coordinates,
        selected_residue_mask=selected_residue_mask,
        slit_coordinates=centered_slit_coordinates,
        final_box_lengths=final_box_lengths,
        target_resname=config.thymol_resname,
        general_cutoff_nm=config.general_cutoff_nm,
        ring_atom_prefix=config.ring_atom_prefix,
        ring_plane_tolerance_nm=config.ring_plane_tolerance_nm,
        ring_polygon_padding_nm=config.ring_polygon_padding_nm,
        include_hydrogen_bonds_in_ring_check=config.include_hydrogen_bonds_in_ring_check,
    )
    kept_thymol_mask = build_kept_thymol_atom_mask(
        thymol_system=thymol_system,
        selected_residue_mask=selected_residue_mask,
        removed_residue_mask=clash_selection.removed_residue_mask,
    )

    output_axis_permutation = build_output_axis_permutation(surface_plane_region.axis_index)
    output_box_lengths = permute_box_axes(
        box_lengths=final_box_lengths,
        axis_permutation=output_axis_permutation,
    )
    output_slit_coordinates = permute_coordinate_axes(
        coordinates=centered_slit_coordinates,
        axis_permutation=output_axis_permutation,
    )
    output_thymol_coordinates = permute_coordinate_axes(
        coordinates=translated_thymol_coordinates,
        axis_permutation=output_axis_permutation,
    )
    if config.wrap_output:
        output_thymol_coordinates = wrap_residues(
            system=thymol_system,
            coordinates=output_thymol_coordinates,
            keep_atom_mask=kept_thymol_mask,
            box_lengths=output_box_lengths,
        )

    final_atom_count, final_residue_count = write_merged_gro(
        config=config,
        slit_system=slit_system,
        slit_coordinates=output_slit_coordinates,
        thymol_system=thymol_system,
        thymol_coordinates=output_thymol_coordinates,
        kept_thymol_mask=kept_thymol_mask,
        final_box_lengths=output_box_lengths,
    )

    target_residue_mask = np.array(
        [residue_span.residue_name == config.thymol_resname for residue_span in thymol_system.residue_spans],
        dtype=bool,
    )
    general_mask = clash_selection.removed_by_general_mask & target_residue_mask
    forward_mask = clash_selection.removed_by_forward_ring_mask & target_residue_mask
    reverse_mask = clash_selection.removed_by_reverse_ring_mask & target_residue_mask
    any_ring_mask = clash_selection.removed_by_any_ring_mask & target_residue_mask
    removed_mask = clash_selection.removed_residue_mask & target_residue_mask
    cropped_mask = cropped_residue_mask & target_residue_mask
    surface_plane_mask = surface_plane_removed_mask & target_residue_mask
    surface_plane_filtered_mask = selected_residue_mask & target_residue_mask

    initial_thymol_molecules = int(np.count_nonzero(target_residue_mask))
    cropped_thymol_molecules = int(np.count_nonzero(cropped_mask))
    removed_outside_crop_thymol_molecules = initial_thymol_molecules - cropped_thymol_molecules
    surface_plane_filtered_thymol_molecules = int(np.count_nonzero(surface_plane_filtered_mask))
    removed_by_surface_plane_thymol_molecules = int(np.count_nonzero(surface_plane_mask))
    removed_by_general_cutoff_thymol_molecules = int(np.count_nonzero(general_mask))
    removed_by_forward_ring_thymol_molecules = int(np.count_nonzero(forward_mask))
    removed_by_reverse_ring_thymol_molecules = int(np.count_nonzero(reverse_mask))
    removed_by_any_ring_thymol_molecules = int(np.count_nonzero(any_ring_mask))

    removed_by_general_only_thymol_molecules = int(np.count_nonzero(general_mask & ~forward_mask & ~reverse_mask))
    removed_by_forward_ring_only_thymol_molecules = int(np.count_nonzero(~general_mask & forward_mask & ~reverse_mask))
    removed_by_reverse_ring_only_thymol_molecules = int(np.count_nonzero(~general_mask & ~forward_mask & reverse_mask))
    removed_by_general_and_forward_ring_only_thymol_molecules = int(
        np.count_nonzero(general_mask & forward_mask & ~reverse_mask)
    )
    removed_by_general_and_reverse_ring_only_thymol_molecules = int(
        np.count_nonzero(general_mask & ~forward_mask & reverse_mask)
    )
    removed_by_forward_and_reverse_ring_only_thymol_molecules = int(
        np.count_nonzero(~general_mask & forward_mask & reverse_mask)
    )
    removed_by_general_and_forward_and_reverse_ring_thymol_molecules = int(
        np.count_nonzero(general_mask & forward_mask & reverse_mask)
    )

    removed_by_clash_thymol_molecules = int(np.count_nonzero(removed_mask))
    removed_thymol_molecules = (
        removed_outside_crop_thymol_molecules
        + removed_by_surface_plane_thymol_molecules
        + removed_by_clash_thymol_molecules
    )
    remaining_thymol_molecules = initial_thymol_molecules - removed_thymol_molecules

    density_estimate = compute_density_estimate(
        thymol_system=thymol_system,
        slit_system=slit_system,
        slit_coordinates=centered_slit_coordinates,
        final_box_lengths=final_box_lengths,
        target_resname=config.thymol_resname,
        remaining_thymol_molecules=remaining_thymol_molecules,
        probe_radii_nm=config.density_probe_radii_nm,
        sample_count=config.density_sample_count,
        seed_count=config.density_seed_count,
    )

    report = MergeReport(
        initial_thymol_molecules=initial_thymol_molecules,
        cropped_thymol_molecules=cropped_thymol_molecules,
        removed_outside_crop_thymol_molecules=removed_outside_crop_thymol_molecules,
        surface_plane_region=surface_plane_region,
        surface_plane_filtered_thymol_molecules=surface_plane_filtered_thymol_molecules,
        removed_by_surface_plane_thymol_molecules=removed_by_surface_plane_thymol_molecules,
        removed_by_general_cutoff_thymol_molecules=removed_by_general_cutoff_thymol_molecules,
        removed_by_forward_ring_thymol_molecules=removed_by_forward_ring_thymol_molecules,
        removed_by_reverse_ring_thymol_molecules=removed_by_reverse_ring_thymol_molecules,
        removed_by_any_ring_thymol_molecules=removed_by_any_ring_thymol_molecules,
        removed_by_general_only_thymol_molecules=removed_by_general_only_thymol_molecules,
        removed_by_forward_ring_only_thymol_molecules=removed_by_forward_ring_only_thymol_molecules,
        removed_by_reverse_ring_only_thymol_molecules=removed_by_reverse_ring_only_thymol_molecules,
        removed_by_general_and_forward_ring_only_thymol_molecules=removed_by_general_and_forward_ring_only_thymol_molecules,
        removed_by_general_and_reverse_ring_only_thymol_molecules=removed_by_general_and_reverse_ring_only_thymol_molecules,
        removed_by_forward_and_reverse_ring_only_thymol_molecules=removed_by_forward_and_reverse_ring_only_thymol_molecules,
        removed_by_general_and_forward_and_reverse_ring_thymol_molecules=removed_by_general_and_forward_and_reverse_ring_thymol_molecules,
        removed_by_clash_thymol_molecules=removed_by_clash_thymol_molecules,
        removed_thymol_molecules=removed_thymol_molecules,
        remaining_thymol_molecules=remaining_thymol_molecules,
        slit_phenyl_ring_count=len(ring_check_cache.slit_ring_geometries),
        cropped_thymol_ring_count=len(ring_check_cache.target_ring_geometries),
        thymol_bonds_checked_per_molecule=len(ring_check_cache.target_bond_template),
        slit_bond_template_count=len(ring_check_cache.slit_bond_templates),
        slit_bond_count_checked=len(ring_check_cache.slit_bond_geometries),
        density_estimate=density_estimate,
        slit_atom_count=slit_system.atom_count,
        final_atom_count=final_atom_count,
        final_residue_count=final_residue_count,
        output_axis_permutation=output_axis_permutation,
        crop_window_start_nm=crop_window_start,
        output_box_nm=output_box_lengths,
    )

    log_text = build_log_text(config, report)
    config.log_path.write_text(log_text, encoding="utf-8")
    print(log_text, end="")
    return report


def main() -> None:
    """Parse CLI arguments and run the merge workflow."""

    config = parse_arguments()
    merge_systems(config)


if __name__ == "__main__":
    main()
