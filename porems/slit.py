################################################################################
# Slit Preparation Helpers                                                     #
#                                                                              #
"""High-level helpers for preparing amorphous silica slit surfaces."""
################################################################################


import copy
import json
import math
import os

from dataclasses import asdict, dataclass, field

import porems as pms


_BRIDGE_OFFSET_NM = 0.09
_BRIDGE_STERIC_DISTANCE_CUTOFF_NM = 0.30
_BRIDGE_STERIC_GRAPH_DEPTH = 6
_BRIDGE_MIN_CLEARANCE_BY_TYPE_NM = {
    "O": 0.18,
    "Si": 0.22,
    "H": 0.12,
}
_BRIDGE_CANDIDATE_ROTATIONS_DEG = (0, 45, -45, 90, -90, 135, -135, 180)


@dataclass(frozen=True)
class SiliconStateFractions:
    """Five-state silicon surface fractions.

    Parameters
    ----------
    q2_fraction : float
        Fraction of geminal silanol sites ``Q2`` over the modeled surface
        silicon population.
    q3_fraction : float
        Fraction of single silanol sites ``Q3`` over the modeled surface
        silicon population.
    q4_fraction : float
        Fraction of fully condensed surface silicon sites ``Q4`` over the
        modeled surface silicon population.
    t2_fraction : float, optional
        Fraction of geminally attached organosilicon sites ``T2`` over the
        modeled surface silicon population.
    t3_fraction : float, optional
        Fraction of singly attached organosilicon sites ``T3`` over the
        modeled surface silicon population.
    """

    q2_fraction: float
    q3_fraction: float
    q4_fraction: float
    t2_fraction: float = 0.0
    t3_fraction: float = 0.0

    def __post_init__(self):
        """Validate the fraction payload.

        Raises
        ------
        ValueError
            Raised when any fraction is negative or when the total fraction
            does not add up to one.
        """
        fractions = (
            self.q2_fraction,
            self.q3_fraction,
            self.q4_fraction,
            self.t2_fraction,
            self.t3_fraction,
        )
        if min(fractions) < 0:
            raise ValueError("Silicon-state fractions must be non-negative.")
        if abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError("Silicon-state fractions must add up to 1.0.")


@dataclass(frozen=True)
class ExperimentalSiliconStateTarget:
    """Experimental silicon-state fractions over all Si atoms in the sample.

    Parameters
    ----------
    q2_fraction : float
        Experimental ``Q2`` fraction over all Si atoms in the sample.
    q3_fraction : float
        Experimental ``Q3`` fraction over all Si atoms in the sample.
    q4_fraction : float
        Experimental ``Q4`` fraction over all Si atoms in the sample.
    t2_fraction : float, optional
        Experimental ``T2`` fraction over all Si atoms in the sample.
    t3_fraction : float, optional
        Experimental ``T3`` fraction over all Si atoms in the sample.
    alpha_override : float or None, optional
        Optional explicit surface-to-total silicon fraction used when mapping
        the experimental all-silicon ratios to the modeled slit surface. When
        ``None``, the builder derives ``alpha`` from the slit geometry.
    """

    q2_fraction: float
    q3_fraction: float
    q4_fraction: float
    t2_fraction: float = 0.0
    t3_fraction: float = 0.0
    alpha_override: float | None = None

    def __post_init__(self):
        """Validate the experimental target.

        Raises
        ------
        ValueError
            Raised when the fractions are invalid or when an explicit
            ``alpha`` override lies outside ``(0, 1]``.
        """
        SiliconStateFractions(
            self.q2_fraction,
            self.q3_fraction,
            self.q4_fraction,
            self.t2_fraction,
            self.t3_fraction,
        )
        if self.alpha_override is not None and not (0.0 < self.alpha_override <= 1.0):
            raise ValueError("The alpha override must be in the interval (0, 1].")


@dataclass(frozen=True)
class AmorphousSlitConfig:
    """Configuration for a periodic bare amorphous silica slit.

    Parameters
    ----------
    name : str, optional
        Base name used for stored slit files.
    slit_width_nm : float, optional
        Width of the slit in nanometers.
    repeat_y : int, optional
        Number of amorphous template copies stacked along the slit-normal
        direction.
    temperature_k : float, optional
        Target simulation temperature in Kelvin.
    surface_target : ExperimentalSiliconStateTarget, optional
        Experimental silicon-state ratios used to derive the modeled slit
        surface target. The default target is expressed over all Si atoms for
        the default slit geometry and therefore relies on the automatically
        derived ``alpha`` value.
    amorph_bond_range_nm : tuple, optional
        Accepted ``Si-O`` bond-length range for the amorphous template.
    siloxane_distance_range_nm : tuple, optional
        Accepted ``Si-Si`` distance range for custom siloxane formation.
    surface_fraction_tolerance : float, optional
        Allowed absolute fraction deviation per silicon state when the exact
        integer target cannot be realized on the current slit.
    template_split_pairs : tuple, optional
        Template-specific bond pairs that must be disconnected after
        reconstructing the amorphous connectivity matrix.
    """

    name: str = "bare_amorphous_silica_slit"
    slit_width_nm: float = 7.0
    repeat_y: int = 2
    temperature_k: float = 300.0
    surface_target: ExperimentalSiliconStateTarget = field(
        default_factory=lambda: ExperimentalSiliconStateTarget(
            q2_fraction=66 / 40000,
            q3_fraction=650 / 40000,
            q4_fraction=1.0 - ((66 + 650) / 40000),
        )
    )
    amorph_bond_range_nm: tuple[float, float] = (0.160 - 0.02, 0.160 + 0.02)
    siloxane_distance_range_nm: tuple[float, float] = (0.40, 0.65)
    surface_fraction_tolerance: float = 0.005
    template_split_pairs: tuple[tuple[int, int], ...] = ((57790, 2524),)

    def __post_init__(self):
        """Validate the slit configuration.

        Raises
        ------
        ValueError
            Raised when the slit width, y repetition count, or temperature is
            invalid.
        """
        if self.slit_width_nm <= 0:
            raise ValueError("The slit width must be positive.")
        if self.repeat_y < 1:
            raise ValueError("The amorphous slit requires at least one y-repeat.")
        if self.temperature_k <= 0:
            raise ValueError("The temperature must be positive.")
        if self.surface_fraction_tolerance < 0:
            raise ValueError("The surface fraction tolerance must be non-negative.")


@dataclass(frozen=True)
class SiliconStateComposition:
    """Integer five-state silicon surface composition.

    Parameters
    ----------
    total_surface_si : int
        Total number of tracked surface silicon atoms.
    q2_sites : int
        Number of residual ``Q2`` surface silicon sites.
    q3_sites : int
        Number of residual ``Q3`` surface silicon sites.
    q4_sites : int
        Number of residual ``Q4`` surface silicon sites.
    t2_sites : int, optional
        Number of attached ``T2`` sites.
    t3_sites : int, optional
        Number of attached ``T3`` sites.
    """

    total_surface_si: int
    q2_sites: int
    q3_sites: int
    q4_sites: int
    t2_sites: int = 0
    t3_sites: int = 0

    def __post_init__(self):
        """Validate the integer silicon-state counts.

        Raises
        ------
        ValueError
            Raised when the counts are invalid or do not add up to the tracked
            surface silicon count.
        """
        counts = (
            self.total_surface_si,
            self.q2_sites,
            self.q3_sites,
            self.q4_sites,
            self.t2_sites,
            self.t3_sites,
        )
        if min(counts) < 0:
            raise ValueError("Silicon-state counts must be non-negative.")
        if (
            self.q2_sites
            + self.q3_sites
            + self.q4_sites
            + self.t2_sites
            + self.t3_sites
            != self.total_surface_si
        ):
            raise ValueError("Silicon-state counts must add up to the total surface silicon count.")

    @property
    def q2_fraction(self):
        """Return the residual ``Q2`` fraction."""
        return self.q2_sites / self.total_surface_si if self.total_surface_si else 0.0

    @property
    def q3_fraction(self):
        """Return the residual ``Q3`` fraction."""
        return self.q3_sites / self.total_surface_si if self.total_surface_si else 0.0

    @property
    def q4_fraction(self):
        """Return the residual ``Q4`` fraction."""
        return self.q4_sites / self.total_surface_si if self.total_surface_si else 0.0

    @property
    def t2_fraction(self):
        """Return the ``T2`` fraction."""
        return self.t2_sites / self.total_surface_si if self.total_surface_si else 0.0

    @property
    def t3_fraction(self):
        """Return the ``T3`` fraction."""
        return self.t3_sites / self.total_surface_si if self.total_surface_si else 0.0


@dataclass(frozen=True)
class SlitPreparationReport:
    """Summary of a prepared or functionalized amorphous slit.

    Parameters
    ----------
    name : str
        Slit-system name.
    temperature_k : float
        Target simulation temperature in Kelvin.
    box_nm : list[float]
        Periodic simulation box in nanometers after slit preparation.
    slit_width_nm : float
        Requested slit width.
    wall_thickness_nm : float
        Silica wall thickness on each side of the slit.
    site_ex : int
        Number of exterior surface sites.
    siloxane_bridges : int
        Number of siloxane bridges introduced during surface editing.
    siloxane_distance_range_nm : tuple[float, float]
        Accepted ``Si-Si`` distance range used during custom surface editing.
    surface_fraction_tolerance : float
        Allowed absolute fraction deviation per silicon state for fallback
        target selection.
    alpha_auto : float
        Alpha value derived from the slit geometry.
    alpha_effective : float
        Alpha value actually used when converting experimental fractions to the
        modeled surface target.
    used_surface_tolerance : bool
        Whether the selected target was accepted through the tolerance fallback
        rather than matched exactly.
    experimental_target : ExperimentalSiliconStateTarget
        Experimental all-silicon target supplied by the caller.
    derived_surface_target : SiliconStateFractions
        Surface-only fractions derived from the experimental target and
        ``alpha``.
    initial_surface : SiliconStateComposition
        Surface composition before custom condensation.
    target_surface : SiliconStateComposition
        Selected integer final surface target.
    prepared_surface : SiliconStateComposition
        Surface composition after Q-state preparation and before optional
        grafting.
    final_surface : SiliconStateComposition
        Final post-grafting surface composition. For bare slits this is
        identical to ``prepared_surface``.
    preparation_diagnostics : SurfacePreparationDiagnostics
        Surface-cleanup and bridge-insertion diagnostics collected across
        preparation, Q-state editing, and optional grafting.
    """

    name: str
    temperature_k: float
    box_nm: list[float]
    slit_width_nm: float
    wall_thickness_nm: float
    site_ex: int
    siloxane_bridges: int
    siloxane_distance_range_nm: tuple[float, float]
    surface_fraction_tolerance: float
    alpha_auto: float
    alpha_effective: float
    used_surface_tolerance: bool
    experimental_target: ExperimentalSiliconStateTarget
    derived_surface_target: SiliconStateFractions
    initial_surface: SiliconStateComposition
    target_surface: SiliconStateComposition
    prepared_surface: SiliconStateComposition
    final_surface: SiliconStateComposition
    preparation_diagnostics: pms.SurfacePreparationDiagnostics


@dataclass
class SlitPreparationResult:
    """Prepared bare slit system together with its report.

    Parameters
    ----------
    system : PoreKit
        Bare slit system associated with the preparation report.
    report : SlitPreparationReport
        Summary of the generated slit geometry and surface composition.
    """

    system: pms.PoreKit
    report: SlitPreparationReport


@dataclass(frozen=True)
class SilaneAttachmentConfig:
    """Attachment settings for one silane family.

    Parameters
    ----------
    molecule : Molecule
        Base post-condensation ligand fragment used for both single and
        geminal attachment.
    mount : int
        Atom id placed onto the selected silicon surface site.
    axis : tuple[int, int]
        Two atom ids defining the molecular attachment axis.
    """

    molecule: pms.Molecule
    mount: int
    axis: tuple[int, int]


@dataclass(frozen=True)
class FunctionalizedAmorphousSlitConfig:
    """Configuration for an exactly targeted functionalized amorphous slit.

    Parameters
    ----------
    slit_config : AmorphousSlitConfig
        Base slit configuration, including the unified experimental target.
    ligand : SilaneAttachmentConfig
        Silane attachment definition used to realize ``T2`` and ``T3``.
    """

    slit_config: AmorphousSlitConfig
    ligand: SilaneAttachmentConfig


@dataclass
class FunctionalizedSlitResult:
    """Prepared functionalized slit system together with its report.

    Parameters
    ----------
    system : PoreKit
        Functionalized slit system associated with the preparation report.
    report : SlitPreparationReport
        Summary of the generated slit geometry and surface composition.
    """

    system: pms.PoreKit
    report: SlitPreparationReport


@dataclass(frozen=True)
class _SurfaceTargetCandidate:
    """Candidate final surface composition ranked against target fractions.

    Parameters
    ----------
    composition : SiliconStateComposition
        Candidate integer five-state surface composition.
    total_fraction_error : float
        Sum of the absolute fraction deviations from the requested five-state
        target.
    """

    composition: SiliconStateComposition
    total_fraction_error: float


@dataclass
class _SurfaceTargetAttempt:
    """Successful realization of one final slit-surface composition.

    Parameters
    ----------
    system : PoreKit
        Edited slit system that satisfies the selected final target.
    target_surface : SiliconStateComposition
        Selected integer final target.
    prepared_surface : SiliconStateComposition
        Intermediate prepared bare surface before optional grafting.
    final_surface : SiliconStateComposition
        Final post-grafting surface composition.
    siloxane_bridges : int
        Number of siloxane bridges introduced during Q-state preparation.
    used_surface_tolerance : bool
        Whether the selected target came from the tolerance fallback rather
        than the exact integer target.
    """

    system: pms.PoreKit
    target_surface: SiliconStateComposition
    prepared_surface: SiliconStateComposition
    final_surface: SiliconStateComposition
    siloxane_bridges: int
    used_surface_tolerance: bool


@dataclass
class _BaseSlitBuild:
    """Base slit system prepared before target realization.

    Parameters
    ----------
    system : PoreKit
        Prepared slit system before custom siloxane formation.
    total_surface_si : int
        Number of tracked surface silicon sites.
    total_active_si : int
        Number of active silicon atoms in the current slit model.
    initial_surface : SiliconStateComposition
        Initial surface composition before custom siloxane formation.
    """

    system: pms.PoreKit
    total_surface_si: int
    total_active_si: int
    initial_surface: SiliconStateComposition


def _amorphous_template_path():
    """Return the built-in amorphous silica template path.

    Returns
    -------
    template_path : str
        Absolute path to the amorphous silica template.
    """
    return os.path.join(os.path.dirname(__file__), "templates", "amorph.gro")


def _replicate_along_y(base, repeat_y):
    """Replicate the amorphous template along the ``y`` axis.

    Parameters
    ----------
    base : Molecule
        Base amorphous silica template.
    repeat_y : int
        Number of copies along ``y``.

    Returns
    -------
    replicated : Molecule
        Replicated amorphous structure.
    """
    box = base.get_box()
    copies = []
    for copy_id in range(repeat_y):
        block = copy.deepcopy(base)
        block.translate([0, box[1] * copy_id, 0])
        copies.append(block)

    replicated = pms.Molecule(inp=copies)
    replicated.set_box([box[0], box[1] * repeat_y, box[2]])
    replicated.set_name("replicated_amorphous_silica")
    replicated.set_short("AMO")

    return replicated


def _duplicate_template_splits(matrix, atoms_per_copy, repeat_y, split_pairs):
    """Apply template-specific bond removals to each replicated copy.

    Parameters
    ----------
    matrix : Matrix
        Connectivity matrix for the replicated amorphous structure.
    atoms_per_copy : int
        Number of atoms in one amorphous template copy.
    repeat_y : int
        Number of copies stacked along ``y``.
    split_pairs : tuple[tuple[int, int], ...]
        Pairwise atom indices that must be disconnected in each copy.
    """
    for copy_id in range(repeat_y):
        offset = copy_id * atoms_per_copy
        for atom_a, atom_b in split_pairs:
            matrix.split(atom_a + offset, atom_b + offset)


def _site_distance(pos_a, pos_b):
    """Calculate the direct distance between two silicon positions.

    Parameters
    ----------
    pos_a : list[float]
        First position vector.
    pos_b : list[float]
        Second position vector.

    Returns
    -------
    distance : float
        Euclidean distance.
    """
    return pms.geom.length(pms.geom.vector(pos_a, pos_b))


def _minimum_image_vector(pos_a, pos_b, box):
    """Return the minimum-image vector between two positions.

    Parameters
    ----------
    pos_a : list[float]
        First Cartesian position.
    pos_b : list[float]
        Second Cartesian position.
    box : list[float]
        Periodic box lengths.

    Returns
    -------
    vector : list[float]
        Minimum-image vector from ``pos_a`` to ``pos_b``.
    """
    vector = pms.geom.vector(pos_a, pos_b)
    for dim, length in enumerate(box):
        if length > 0:
            vector[dim] -= length * round(vector[dim] / length)
    return vector


def _wrap_position(pos, box):
    """Wrap a Cartesian position back into the periodic box.

    Parameters
    ----------
    pos : list[float]
        Cartesian position.
    box : list[float]
        Periodic box lengths.

    Returns
    -------
    wrapped_pos : list[float]
        Box-wrapped Cartesian position.
    """
    wrapped_pos = pos[:]
    for dim, length in enumerate(box):
        if length > 0:
            wrapped_pos[dim] %= length
    return wrapped_pos


def _active_silicon_count(system):
    """Count active silicon atoms in the current slit model.

    Parameters
    ----------
    system : PoreKit
        Slit system whose active connectivity matrix should be inspected.

    Returns
    -------
    count : int
        Number of silicon atoms still present in the active slit model.
    """
    return sum(
        1
        for atom_id in system._matrix.get_matrix()
        if system._block.get_atom_type(atom_id) == "Si"
    )


def _attached_state_counts(system, ligand):
    """Count attached ``T2`` and ``T3`` sites for one silane family.

    Parameters
    ----------
    system : PoreKit
        Current slit system.
    ligand : SilaneAttachmentConfig or None
        Silane family tracked in the current build. ``None`` means no grafted
        silicon states are present.

    Returns
    -------
    counts : tuple[int, int]
        Attached ``T2`` and ``T3`` counts in that order.
    """
    if ligand is None:
        return (0, 0)

    site_dict = system._pore.get_site_dict()["in"]
    base_short = ligand.molecule.get_short()
    return (len(site_dict.get(base_short + "G", [])), len(site_dict.get(base_short, [])))


def _interior_attached_molecule_counts(system):
    """Return non-silanol interior molecule counts.

    Parameters
    ----------
    system : PoreKit
        Current slit system.

    Returns
    -------
    counts : dict[str, int]
        Attached interior molecule counts keyed by residue short name.
    """
    counts = {}
    for short_name, mols in system._pore.get_site_dict()["in"].items():
        if short_name in {"SL", "SLG", "SLX"}:
            continue
        counts[short_name] = len(mols)
    return counts


def _surface_composition(total_surface_si, sites, t2_sites=0, t3_sites=0):
    """Summarize the current five-state surface composition.

    Parameters
    ----------
    total_surface_si : int
        Total number of tracked surface silicon atoms.
    sites : dict[int, pms.BindingSite]
        Current slit binding sites keyed by silicon identifier.
    t2_sites : int, optional
        Number of attached ``T2`` states already realized on the slit.
    t3_sites : int, optional
        Number of attached ``T3`` states already realized on the slit.

    Returns
    -------
    composition : SiliconStateComposition
        Five-state surface composition.
    """
    raw_q2_sites = sum(
        1 for site in sites.values() if site.site_type == "in" and site.is_geminal
    )
    raw_q3_sites = sum(
        1 for site in sites.values() if site.site_type == "in" and site.oxygen_count == 1
    )
    q2_sites = raw_q2_sites - t2_sites
    q3_sites = raw_q3_sites - t3_sites
    q4_sites = total_surface_si - q2_sites - q3_sites - t2_sites - t3_sites

    return SiliconStateComposition(
        total_surface_si=total_surface_si,
        q2_sites=q2_sites,
        q3_sites=q3_sites,
        q4_sites=q4_sites,
        t2_sites=t2_sites,
        t3_sites=t3_sites,
    )


def _surface_fraction_errors(composition, target):
    """Return absolute fraction deviations from a requested five-state target.

    Parameters
    ----------
    composition : SiliconStateComposition
        Candidate or realized surface composition.
    target : SiliconStateFractions
        Requested surface-only five-state fractions.

    Returns
    -------
    errors : tuple[float, float, float, float, float]
        Absolute fraction deviations for ``Q2``, ``Q3``, ``Q4``, ``T2``, and
        ``T3``.
    """
    return (
        abs(composition.q2_fraction - target.q2_fraction),
        abs(composition.q3_fraction - target.q3_fraction),
        abs(composition.q4_fraction - target.q4_fraction),
        abs(composition.t2_fraction - target.t2_fraction),
        abs(composition.t3_fraction - target.t3_fraction),
    )


def _effective_alpha(total_surface_si, total_active_si, target):
    """Resolve the automatic and effective alpha values.

    Parameters
    ----------
    total_surface_si : int
        Number of tracked surface silicon sites in the slit.
    total_active_si : int
        Number of active silicon atoms in the current slit model.
    target : ExperimentalSiliconStateTarget
        Experimental all-silicon target supplied by the caller.

    Returns
    -------
    alpha_values : tuple[float, float]
        Auto-derived alpha and the effective alpha used for conversion.

    Raises
    ------
    ValueError
        Raised when the slit does not contain any active silicon atoms or when
        the effective alpha falls outside ``(0, 1]``.
    """
    if total_active_si <= 0:
        raise ValueError("Cannot derive alpha from a slit without active silicon atoms.")

    alpha_auto = total_surface_si / total_active_si
    alpha_effective = target.alpha_override if target.alpha_override is not None else alpha_auto
    if not (0.0 < alpha_effective <= 1.0):
        raise ValueError("The effective alpha must be in the interval (0, 1].")

    return alpha_auto, alpha_effective


def _surface_target_from_experimental(target, alpha):
    """Convert experimental all-silicon ratios into surface-only fractions.

    Parameters
    ----------
    target : ExperimentalSiliconStateTarget
        Experimental all-silicon ratios supplied by the caller.
    alpha : float
        Effective surface-to-total silicon ratio used for the conversion.

    Returns
    -------
    surface_target : SiliconStateFractions
        Modeled surface-only fractions.

    Raises
    ------
    ValueError
        Raised when the experimental ratios and alpha are incompatible with a
        surface-only interpretation.
    """
    surface_target = SiliconStateFractions(
        q2_fraction=target.q2_fraction / alpha,
        q3_fraction=target.q3_fraction / alpha,
        q4_fraction=(target.q4_fraction - (1.0 - alpha)) / alpha,
        t2_fraction=target.t2_fraction / alpha,
        t3_fraction=target.t3_fraction / alpha,
    )

    non_q4_fraction = (
        target.q2_fraction
        + target.q3_fraction
        + target.t2_fraction
        + target.t3_fraction
    )
    if alpha + 1e-9 < non_q4_fraction:
        raise ValueError(
            "The effective alpha is too small for the requested experimental "
            "non-Q4 silicon-state fractions."
        )

    return surface_target


def _nearest_integer_composition(total_surface_si, target):
    """Convert target fractions into one nearest integer surface composition.

    Parameters
    ----------
    total_surface_si : int
        Number of tracked surface silicon atoms.
    target : SiliconStateFractions
        Requested surface-only target fractions.

    Returns
    -------
    composition : SiliconStateComposition
        Nearest integer surface composition whose counts add up to
        ``total_surface_si``.
    """
    ideals = {
        "q2_sites": total_surface_si * target.q2_fraction,
        "q3_sites": total_surface_si * target.q3_fraction,
        "q4_sites": total_surface_si * target.q4_fraction,
        "t2_sites": total_surface_si * target.t2_fraction,
        "t3_sites": total_surface_si * target.t3_fraction,
    }
    counts = {key: math.floor(value) for key, value in ideals.items()}
    missing = total_surface_si - sum(counts.values())
    ranked = sorted(
        ideals,
        key=lambda key: (ideals[key] - counts[key], ideals[key], key),
        reverse=True,
    )
    for key in ranked[:missing]:
        counts[key] += 1

    return SiliconStateComposition(
        total_surface_si=total_surface_si,
        **counts,
    )


def _surface_target_candidates(total_surface_si, target, exact_target, tolerance):
    """Enumerate nearby five-state targets inside the allowed tolerance.

    Parameters
    ----------
    total_surface_si : int
        Number of tracked surface silicon atoms.
    target : SiliconStateFractions
        Requested surface-only target fractions.
    exact_target : SiliconStateComposition
        Preferred nearest-integer target.
    tolerance : float
        Allowed absolute fraction deviation per silicon state.

    Returns
    -------
    candidates : list[_SurfaceTargetCandidate]
        Nearby candidate targets sorted by increasing total fraction error.
    """
    ranges = {}
    for state_name, state_fraction in (
        ("q2_sites", target.q2_fraction),
        ("q3_sites", target.q3_fraction),
        ("t2_sites", target.t2_fraction),
        ("t3_sites", target.t3_fraction),
    ):
        lower = max(
            0,
            math.ceil(max(0.0, state_fraction - tolerance) * total_surface_si - 1e-9),
        )
        upper = min(
            total_surface_si,
            math.floor(min(1.0, state_fraction + tolerance) * total_surface_si + 1e-9),
        )
        ranges[state_name] = (lower, upper)

    q4_lower = max(
        0,
        math.ceil(max(0.0, target.q4_fraction - tolerance) * total_surface_si - 1e-9),
    )
    q4_upper = min(
        total_surface_si,
        math.floor(min(1.0, target.q4_fraction + tolerance) * total_surface_si + 1e-9),
    )

    seen = {exact_target}
    candidates = []
    for q2_sites in range(ranges["q2_sites"][0], ranges["q2_sites"][1] + 1):
        for q3_sites in range(ranges["q3_sites"][0], ranges["q3_sites"][1] + 1):
            for t2_sites in range(ranges["t2_sites"][0], ranges["t2_sites"][1] + 1):
                for t3_sites in range(ranges["t3_sites"][0], ranges["t3_sites"][1] + 1):
                    q4_sites = total_surface_si - q2_sites - q3_sites - t2_sites - t3_sites
                    if not (q4_lower <= q4_sites <= q4_upper):
                        continue

                    composition = SiliconStateComposition(
                        total_surface_si=total_surface_si,
                        q2_sites=q2_sites,
                        q3_sites=q3_sites,
                        q4_sites=q4_sites,
                        t2_sites=t2_sites,
                        t3_sites=t3_sites,
                    )
                    if composition in seen:
                        continue

                    errors = _surface_fraction_errors(composition, target)
                    if any(error > tolerance for error in errors):
                        continue

                    candidates.append(
                        _SurfaceTargetCandidate(
                            composition=composition,
                            total_fraction_error=sum(errors),
                        )
                    )
                    seen.add(composition)

    candidates.sort(
        key=lambda candidate: (
            candidate.total_fraction_error,
            candidate.composition.q4_sites,
            candidate.composition.q3_sites,
            candidate.composition.q2_sites,
            candidate.composition.t3_sites,
            candidate.composition.t2_sites,
        )
    )
    return candidates


def _prepared_target_from_final(final_surface):
    """Return the bare pre-grafting target required by a final five-state goal.

    Parameters
    ----------
    final_surface : SiliconStateComposition
        Selected final five-state surface composition.

    Returns
    -------
    prepared_surface : SiliconStateComposition
        Bare pre-grafting Q-state composition required to realize the final
        surface after exact ``T2/T3`` attachment.
    """
    return SiliconStateComposition(
        total_surface_si=final_surface.total_surface_si,
        q2_sites=final_surface.q2_sites + final_surface.t2_sites,
        q3_sites=final_surface.q3_sites + final_surface.t3_sites,
        q4_sites=final_surface.q4_sites,
    )


def _prepared_target_is_compatible(initial_surface, prepared_surface):
    """Return True when a prepared Q-state target is chemically reachable.

    Parameters
    ----------
    initial_surface : SiliconStateComposition
        Initial surface composition before custom condensation.
    prepared_surface : SiliconStateComposition
        Candidate pre-grafting Q-state composition.

    Returns
    -------
    is_compatible : bool
        True when the current siloxane-editing algebra can in principle reach
        the requested prepared surface.
    """
    return (
        prepared_surface.t2_sites == 0
        and prepared_surface.t3_sites == 0
        and prepared_surface.q2_sites <= initial_surface.q2_sites
        and prepared_surface.q4_sites >= initial_surface.q4_sites
        and prepared_surface.q3_sites % 2 == initial_surface.q3_sites % 2
    )


def _are_sites_directly_connected(matrix, site_a, site_b):
    """Check whether two surface silicon atoms already share a bridge oxygen.

    Parameters
    ----------
    matrix : Matrix
        Current connectivity matrix.
    site_a : int
        First silicon site identifier.
    site_b : int
        Second silicon site identifier.

    Returns
    -------
    is_connected : bool
        True if the sites already share a bonded oxygen atom.
    """
    atoms_a = matrix.get_matrix()[site_a]["atoms"]
    atoms_b = matrix.get_matrix()[site_b]["atoms"]
    return any(atom_o in atoms_b for atom_o in atoms_a)


def _build_slit_site_adjacency(kit, site_ids, distance_range):
    """Build a static neighbor graph for potential siloxane formation.

    Parameters
    ----------
    kit : PoreKit
        Slit system under preparation.
    site_ids : list[int]
        Surface silicon identifiers.
    distance_range : tuple[float, float]
        Accepted ``Si-Si`` distance range for siloxane formation.

    Returns
    -------
    adjacency : dict[int, list[tuple[int, float]]]
        Mapping of surface silicon identifiers to sorted neighbor lists.
    """
    adjacency = {site: [] for site in site_ids}
    positions = {site: kit._pore.get_block().pos(site) for site in site_ids}

    for site_index, site_a in enumerate(site_ids):
        for site_b in site_ids[site_index + 1:]:
            if _are_sites_directly_connected(kit._matrix, site_a, site_b):
                continue

            distance = _site_distance(positions[site_a], positions[site_b])
            if distance_range[0] <= distance <= distance_range[1]:
                adjacency[site_a].append((site_b, distance))
                adjacency[site_b].append((site_a, distance))

    for site in adjacency:
        adjacency[site].sort(key=lambda item: (item[1], item[0]))

    return adjacency


def _find_pair(sites, adjacency, first_count, second_count):
    """Find the next eligible siloxane pair for the requested site types.

    Parameters
    ----------
    sites : dict[int, pms.BindingSite]
        Current binding sites keyed by silicon identifier.
    adjacency : dict[int, list[tuple[int, float]]]
        Precomputed slit neighbor graph.
    first_count : int
        Required number of free oxygen atoms on the first site.
    second_count : int
        Required number of free oxygen atoms on the second site.

    Returns
    -------
    pair : tuple[int, int] or None
        Pair of silicon site identifiers, or ``None`` if no eligible pair was
        found.
    """
    for site_a in sorted(sites):
        if sites[site_a].site_type != "in" or sites[site_a].oxygen_count != first_count:
            continue

        for site_b, _distance in adjacency.get(site_a, []):
            if site_b not in sites:
                continue
            if sites[site_b].site_type != "in" or sites[site_b].oxygen_count != second_count:
                continue
            if first_count == second_count and site_b < site_a:
                continue

            return (site_a, site_b)

    return None


def _find_placeable_pair(kit, sites, adjacency, first_count, second_count):
    """Find the next eligible siloxane pair with a valid bridge placement.

    Parameters
    ----------
    kit : PoreKit
        Slit system under preparation.
    sites : dict[int, pms.BindingSite]
        Current binding sites keyed by silicon identifier.
    adjacency : dict[int, list[tuple[int, float]]]
        Precomputed slit neighbor graph.
    first_count : int
        Required number of free oxygen atoms on the first site.
    second_count : int
        Required number of free oxygen atoms on the second site.

    Returns
    -------
    result : tuple[tuple[int, int], list[float]] or tuple[None, None]
        Pair of silicon identifiers and the selected bridge position, or
        ``(None, None)`` when no currently placeable pair exists.
    """
    for site_a in sorted(sites):
        if sites[site_a].site_type != "in" or sites[site_a].oxygen_count != first_count:
            continue

        for site_b, _distance in adjacency.get(site_a, []):
            if site_b not in sites:
                continue
            if sites[site_b].site_type != "in" or sites[site_b].oxygen_count != second_count:
                continue
            if first_count == second_count and site_b < site_a:
                continue

            pair = (site_a, site_b)
            bridge_position = _siloxane_bridge_position(kit, pair)
            if bridge_position is not None:
                return pair, bridge_position

    return None, None


def _bridge_base_direction(kit, pair, center_pos, axis_unit):
    """Return a surface-guided transverse direction for bridge placement.

    Parameters
    ----------
    kit : PoreKit
        Slit system under preparation.
    pair : tuple[int, int]
        Pair of silicon site identifiers.
    center_pos : list[float]
        Minimum-image midpoint between the two silicon atoms.
    axis_unit : list[float]
        Unit vector along the silicon-silicon axis.

    Returns
    -------
    direction : list[float]
        Unit vector perpendicular to the silicon-silicon axis, biased towards
        the local pore-facing surface normal.
    """
    site_dict = kit._pore.get_sites()
    normal_a = site_dict[pair[0]].normal(center_pos)
    normal_b = site_dict[pair[1]].normal(center_pos)
    surface_axis = [normal_a[dim] + normal_b[dim] for dim in range(3)]
    axis_projection = pms.geom.dot_product(surface_axis, axis_unit)
    transverse = [
        surface_axis[dim] - axis_projection * axis_unit[dim]
        for dim in range(3)
    ]

    if pms.geom.length(transverse) < 1e-8:
        fallback_axis = [1.0, 0.0, 0.0] if abs(axis_unit[0]) < 0.9 else [0.0, 1.0, 0.0]
        transverse = pms.geom.cross_product(axis_unit, fallback_axis)

    return pms.geom.unit(transverse)


def _bridge_candidate_positions(kit, pair):
    """Generate bridge-oxygen candidate positions for one silicon pair.

    Parameters
    ----------
    kit : PoreKit
        Slit system under preparation.
    pair : tuple[int, int]
        Pair of silicon site identifiers.

    Returns
    -------
    positions : list[list[float]]
        Box-wrapped candidate positions for the bridging oxygen.
    """
    block = kit._pore.get_block()
    box = block.get_box()
    pos_a = block.pos(pair[0])
    pos_b = block.pos(pair[1])
    pair_vector = _minimum_image_vector(pos_a, pos_b, box)
    center_pos = _wrap_position(
        [pos_a[dim] + 0.5 * pair_vector[dim] for dim in range(3)],
        box,
    )
    axis_unit = pms.geom.unit(pair_vector)
    base_direction = _bridge_base_direction(kit, pair, center_pos, axis_unit)

    positions = []
    for angle in _BRIDGE_CANDIDATE_ROTATIONS_DEG:
        direction = pms.geom.rotate(base_direction, axis_unit, angle, True)
        positions.append(
            _wrap_position(
                [center_pos[dim] + _BRIDGE_OFFSET_NM * direction[dim] for dim in range(3)],
                box,
            )
        )

    return positions


def _bridge_steric_score(kit, pair, bridge_position):
    """Score one bridge-oxygen candidate against nearby active scaffold atoms.

    Parameters
    ----------
    kit : PoreKit
        Slit system under preparation.
    pair : tuple[int, int]
        Pair of silicon site identifiers.
    bridge_position : list[float]
        Candidate bridge-oxygen position.

    Returns
    -------
    score : float
        Minimum steric clearance in nanometers. Negative values indicate that
        at least one nonbonded atom overlaps the candidate more closely than
        allowed.
    """
    block = kit._pore.get_block()
    box = block.get_box()
    matrix = kit._matrix.get_matrix()
    sites = kit._pore.get_sites()
    consumed_oxygen_ids = {
        sites[pair[0]].oxygen_ids[0],
        sites[pair[1]].oxygen_ids[0],
    }
    excluded_ids = {
        pair[0],
        pair[1],
        *consumed_oxygen_ids,
    }

    frontier = list(pair)
    local_ids = set(pair)
    for _depth in range(_BRIDGE_STERIC_GRAPH_DEPTH):
        next_frontier = []
        for atom_id in frontier:
            for neighbor_id in matrix[atom_id]["atoms"]:
                if neighbor_id not in local_ids:
                    local_ids.add(neighbor_id)
                    next_frontier.append(neighbor_id)
        if not next_frontier:
            break
        frontier = next_frontier

    min_clearance = float("inf")
    for atom_id in local_ids:
        if atom_id in excluded_ids:
            continue

        atom_type = block.get_atom_type(atom_id)
        min_distance = _BRIDGE_MIN_CLEARANCE_BY_TYPE_NM.get(atom_type, 0.18)
        delta = _minimum_image_vector(bridge_position, block.pos(atom_id), box)

        if any(abs(component) > _BRIDGE_STERIC_DISTANCE_CUTOFF_NM for component in delta):
            continue

        clearance = pms.geom.length(delta) - min_distance
        if clearance < min_clearance:
            min_clearance = clearance
            if min_clearance < 0:
                return min_clearance

    return min_clearance if min_clearance != float("inf") else _BRIDGE_STERIC_DISTANCE_CUTOFF_NM


def _bridge_global_clearance(kit, pair, bridge_position):
    """Return the full-structure steric clearance for one bridge candidate.

    Parameters
    ----------
    kit : PoreKit
        Slit system under preparation.
    pair : tuple[int, int]
        Pair of silicon site identifiers.
    bridge_position : list[float]
        Candidate bridge-oxygen position.

    Returns
    -------
    score : float
        Minimum steric clearance in nanometers over all active scaffold atoms.
    """
    block = kit._pore.get_block()
    box = block.get_box()
    matrix = kit._matrix.get_matrix()
    sites = kit._pore.get_sites()
    consumed_oxygen_ids = {
        sites[pair[0]].oxygen_ids[0],
        sites[pair[1]].oxygen_ids[0],
    }
    excluded_ids = {
        pair[0],
        pair[1],
        *consumed_oxygen_ids,
    }

    min_clearance = float("inf")
    for atom_id in matrix:
        if atom_id in excluded_ids:
            continue

        atom_type = block.get_atom_type(atom_id)
        min_distance = _BRIDGE_MIN_CLEARANCE_BY_TYPE_NM.get(atom_type, 0.18)
        delta = _minimum_image_vector(bridge_position, block.pos(atom_id), box)

        if any(abs(component) > _BRIDGE_STERIC_DISTANCE_CUTOFF_NM for component in delta):
            continue

        clearance = pms.geom.length(delta) - min_distance
        if clearance < min_clearance:
            min_clearance = clearance
            if min_clearance < 0:
                return min_clearance

    return min_clearance if min_clearance != float("inf") else _BRIDGE_STERIC_DISTANCE_CUTOFF_NM


def _siloxane_bridge_position(kit, pair):
    """Return the least crowded bridge-oxygen position for one silicon pair.

    Parameters
    ----------
    kit : PoreKit
        Slit system under preparation.
    pair : tuple[int, int]
        Pair of silicon site identifiers.

    Returns
    -------
    position : list[float] or None
        Bridging oxygen position, or ``None`` when no sterically acceptable
        candidate was found for the pair.
    """
    candidate_scores = []
    for candidate_position in _bridge_candidate_positions(kit, pair):
        local_score = _bridge_steric_score(kit, pair, candidate_position)
        if local_score >= 0:
            candidate_scores.append((local_score, candidate_position))

    for _local_score, candidate_position in sorted(
        candidate_scores,
        key=lambda item: item[0],
        reverse=True,
    ):
        if _bridge_global_clearance(kit, pair, candidate_position) >= 0:
            return candidate_position

    return None


def _bridge_pair(kit, pair, bridge_position=None):
    """Create one siloxane bridge between two specific surface silicon sites.

    Parameters
    ----------
    kit : PoreKit
        Slit system under preparation.
    pair : tuple[int, int]
        Pair of silicon site identifiers.
    bridge_position : list[float] or None, optional
        Optional preselected bridge-oxygen position. When omitted, the helper
        searches for the least crowded valid bridge position automatically.

    Returns
    -------
    bridge_count : int
        Number of siloxane bridges created, always one on success.
    """
    sites = kit._pore.get_sites()
    if pair[0] not in sites or pair[1] not in sites:
        raise ValueError("Cannot bridge a silicon pair that is no longer present in the site dictionary.")

    bridge_position = _siloxane_bridge_position(kit, pair) if bridge_position is None else bridge_position
    if bridge_position is None:
        raise ValueError("Cannot bridge a silicon pair without a sterically acceptable bridge-oxygen position.")
    block = kit._pore.get_block()
    bridge_atom_id = block.get_num()
    block.add("O", bridge_position, name="OM1")
    kit._matrix.add(pair[0], bridge_atom_id)
    kit._matrix.add(pair[1], bridge_atom_id)
    kit._matrix.get_matrix()[bridge_atom_id]["bonds"] = 2
    kit._pore._record_surface_edit(bridge_atom_id, "inserted_bridge_oxygen")
    kit._pore.objectify([bridge_atom_id])

    newly_condensed_si = []
    for site_id in pair:
        oxygen_id = sites[site_id].oxygen_ids[0]
        kit._matrix.remove(oxygen_id)
        if sites[site_id].is_geminal:
            sites[site_id].oxygen_ids.pop(0)
        else:
            newly_condensed_si.append(site_id)
            del sites[site_id]

    if newly_condensed_si:
        kit._pore.objectify(newly_condensed_si)

    return 1


def _consume_pair(adjacency, pair):
    """Remove a bridged silicon pair from the static slit adjacency graph.

    Parameters
    ----------
    adjacency : dict[int, list[tuple[int, float]]]
        Precomputed slit neighbor graph.
    pair : tuple[int, int]
        Silicon identifiers that have already been bridged once.
    """
    site_a, site_b = pair
    adjacency[site_a] = [item for item in adjacency.get(site_a, []) if item[0] != site_b]
    adjacency[site_b] = [item for item in adjacency.get(site_b, []) if item[0] != site_a]


def _refresh_single_slit_tracking(kit, total_surface_si, composition):
    """Refresh slit-only site tracking after Q-state editing or attachment.

    Parameters
    ----------
    kit : PoreKit
        Slit system under preparation.
    total_surface_si : int
        Total number of tracked surface silicon atoms.
    composition : SiliconStateComposition
        Current five-state slit surface composition.
    """
    sites = kit._pore.get_sites()
    kit._pore.refresh_surface_preparation_diagnostics()
    available_site_in = sorted(
        site for site, data in sites.items() if data.site_type == "in" and data.is_available
    )

    kit._site_in = available_site_in
    kit._site_ex = []
    kit._si_pos_in = [[kit._pore.get_block().pos(site) for site in available_site_in]]
    kit._si_pos_ex = []
    kit.sites_shape = {0: available_site_in}
    kit._pore.sites_sl_shape = {0: available_site_in}

    siloxane_num = kit._pore.get_surface_preparation_diagnostics().inserted_bridge_oxygen
    kit._pore.sites_attach_mol = {
        0: pms.ShapeAttachmentSummary(
            single_silanol_sites=composition.q3_sites,
            geminal_silanol_sites=composition.q2_sites,
            siloxane_bridges=siloxane_num,
            attached_molecules=_interior_attached_molecule_counts(kit),
        )
    }

    # Keep the internal surface-silicon count available for downstream helpers.
    kit._slit_total_surface_si = total_surface_si


def _enforce_surface_target(kit, total_surface_si, target_surface, distance_range):
    """Condense the slit surface until the prepared ``Q`` counts are met.

    Parameters
    ----------
    kit : PoreKit
        Slit system under preparation.
    total_surface_si : int
        Total number of tracked surface silicon atoms.
    target_surface : SiliconStateComposition
        Bare pre-grafting target surface composition. ``T2/T3`` must be zero.
    distance_range : tuple[float, float]
        Accepted ``Si-Si`` distance range for siloxane formation.

    Returns
    -------
    bridge_count : int
        Number of siloxane bridges introduced.
    """
    adjacency = _build_slit_site_adjacency(kit, sorted(kit._site_in), distance_range)
    bridge_count = 0
    sites = kit._pore.get_sites()
    current_surface = _surface_composition(total_surface_si, sites)

    while current_surface.q3_sites < target_surface.q3_sites:
        pair, bridge_position = _find_placeable_pair(kit, sites, adjacency, 2, 2)
        if pair is None:
            raise ValueError("No remaining Q2/Q2 siloxane pair is available to increase the Q3 population.")

        bridge_count += _bridge_pair(kit, pair, bridge_position=bridge_position)
        _consume_pair(adjacency, pair)
        current_surface = _surface_composition(total_surface_si, sites)

    if current_surface.q2_sites < target_surface.q2_sites:
        raise ValueError("The slit surface cannot increase Q3 to the requested value without undershooting the requested Q2 count.")

    while current_surface.q2_sites > target_surface.q2_sites:
        q2_delta = current_surface.q2_sites - target_surface.q2_sites
        pair, bridge_position = _find_placeable_pair(kit, sites, adjacency, 2, 1)

        if pair is None:
            if q2_delta < 2:
                raise ValueError("The slit surface cannot reach the requested Q2 count with the available siloxane pairs.")
            pair, bridge_position = _find_placeable_pair(kit, sites, adjacency, 2, 2)
            if pair is None:
                raise ValueError("No remaining Q2/Q2 siloxane pair is available to reduce the Q2 population.")

        bridge_count += _bridge_pair(kit, pair, bridge_position=bridge_position)
        _consume_pair(adjacency, pair)
        current_surface = _surface_composition(total_surface_si, sites)

    while current_surface.q3_sites > target_surface.q3_sites:
        if (current_surface.q3_sites - target_surface.q3_sites) < 2:
            raise ValueError("The requested Q3 count is incompatible with the siloxane editing parity constraints.")

        pair, bridge_position = _find_placeable_pair(kit, sites, adjacency, 1, 1)
        if pair is None:
            raise ValueError("No remaining Q3/Q3 siloxane pair is available to reach the requested Q3 count.")

        bridge_count += _bridge_pair(kit, pair, bridge_position=bridge_position)
        _consume_pair(adjacency, pair)
        current_surface = _surface_composition(total_surface_si, sites)

    if current_surface.q2_sites != target_surface.q2_sites or current_surface.q3_sites != target_surface.q3_sites:
        raise ValueError("The slit surface could not be edited to the requested prepared Q-state composition.")

    _refresh_single_slit_tracking(kit, total_surface_si, current_surface)

    return bridge_count


def _attach_to_specific_sites(kit, ligand, site_ids, allow_geminal):
    """Attach one silane family to a deterministic list of specific sites.

    Parameters
    ----------
    kit : PoreKit
        Slit system under preparation.
    ligand : SilaneAttachmentConfig
        Silane attachment settings.
    site_ids : list[int]
        Specific site ids that must be consumed in order.
    allow_geminal : bool
        Forwarded geminal-allowance flag for the internal attachment helper.

    Returns
    -------
    mols : list
        Attached molecules returned by :meth:`porems.pore.Pore.attach`.
    """
    if not site_ids:
        return []

    mols = kit._pore.attach(
        copy.deepcopy(ligand.molecule),
        ligand.mount,
        list(ligand.axis),
        site_ids,
        len(site_ids),
        pos_list=[],
        site_type="in",
        is_proxi=False,
        is_random=False,
        is_rotate=False,
        is_g=allow_geminal,
    )
    for mol in mols:
        if mol.get_short() not in kit._sort_list:
            kit._sort_list.append(mol.get_short())
    return mols


def _available_site_ids(system, oxygen_count):
    """Return sorted available interior site ids for one oxygen-count class.

    Parameters
    ----------
    system : PoreKit
        Current slit system.
    oxygen_count : int
        Required number of oxygen handles on the surface site.

    Returns
    -------
    site_ids : list[int]
        Sorted interior site ids matching the requested oxygen count and still
        available for attachment.
    """
    return sorted(
        site_id
        for site_id, site in system._pore.get_sites().items()
        if site.site_type == "in" and site.is_available and site.oxygen_count == oxygen_count
    )


def _realize_surface_target(
    base_system,
    total_surface_si,
    initial_surface,
    target,
    exact_target,
    tolerance,
    distance_range,
    ligand=None,
):
    """Select and realize a compatible final slit-surface composition.

    Parameters
    ----------
    base_system : PoreKit
        Prepared slit system before custom siloxane formation.
    total_surface_si : int
        Total number of tracked surface silicon atoms.
    initial_surface : SiliconStateComposition
        Initial surface composition before custom condensation.
    target : SiliconStateFractions
        Requested surface-only five-state fractions.
    exact_target : SiliconStateComposition
        Preferred exact integer target derived from the requested fractions.
    tolerance : float
        Allowed absolute fraction deviation per silicon state for fallback
        target selection.
    distance_range : tuple[float, float]
        Accepted ``Si-Si`` distance range for siloxane formation.
    ligand : SilaneAttachmentConfig or None, optional
        Optional silane attachment definition used to realize ``T2`` and
        ``T3``.

    Returns
    -------
    attempt : _SurfaceTargetAttempt
        Successfully realized slit surface and selected target metadata.

    Raises
    ------
    ValueError
        Raised when no exact or tolerance-compatible target can be realized on
        the current slit.
    """
    candidate_specs = [(exact_target, False)]
    candidate_specs.extend(
        (candidate.composition, True)
        for candidate in _surface_target_candidates(
            total_surface_si,
            target,
            exact_target,
            tolerance,
        )
    )

    for candidate_surface, used_tolerance in candidate_specs:
        prepared_target = _prepared_target_from_final(candidate_surface)
        if not _prepared_target_is_compatible(initial_surface, prepared_target):
            continue

        trial_system = copy.deepcopy(base_system)
        try:
            bridge_count = _enforce_surface_target(
                trial_system,
                total_surface_si,
                prepared_target,
                distance_range,
            )
        except ValueError:
            continue

        prepared_surface = _surface_composition(total_surface_si, trial_system._pore.get_sites())

        if ligand is not None:
            geminal_sites = _available_site_ids(trial_system, oxygen_count=2)
            if len(geminal_sites) < candidate_surface.t2_sites:
                continue
            _attach_to_specific_sites(
                trial_system,
                ligand,
                geminal_sites[:candidate_surface.t2_sites],
                allow_geminal=True,
            )

            single_sites = _available_site_ids(trial_system, oxygen_count=1)
            if len(single_sites) < candidate_surface.t3_sites:
                continue
            _attach_to_specific_sites(
                trial_system,
                ligand,
                single_sites[:candidate_surface.t3_sites],
                allow_geminal=False,
            )

        attached_t2, attached_t3 = _attached_state_counts(trial_system, ligand)
        final_surface = _surface_composition(
            total_surface_si,
            trial_system._pore.get_sites(),
            t2_sites=attached_t2,
            t3_sites=attached_t3,
        )
        if final_surface != candidate_surface:
            continue

        _refresh_single_slit_tracking(trial_system, total_surface_si, final_surface)
        return _SurfaceTargetAttempt(
            system=trial_system,
            target_surface=candidate_surface,
            prepared_surface=prepared_surface,
            final_surface=final_surface,
            siloxane_bridges=bridge_count,
            used_surface_tolerance=used_tolerance,
        )

    raise ValueError(
        "The slit surface could not be edited to the requested silicon-state composition within the allowed tolerance."
    )


def _build_base_slit_system(config):
    """Build the base amorphous slit before target realization.

    Parameters
    ----------
    config : AmorphousSlitConfig
        Slit preparation configuration.

    Returns
    -------
    build : _BaseSlitBuild
        Base slit system together with the initial surface metadata.

    Raises
    ------
    ValueError
        Raised when the generated slit unexpectedly contains exterior sites.
    """
    base = pms.Molecule(inp=_amorphous_template_path())
    replicated = _replicate_along_y(base, config.repeat_y)

    system = pms.PoreKit()
    system.structure(replicated)
    system.build(bonds=list(config.amorph_bond_range_nm))
    _duplicate_template_splits(system._matrix, base.get_num(), config.repeat_y, config.template_split_pairs)

    system.add_shape(system.shape_slit(config.slit_width_nm, centroid=system.centroid()), hydro=0)
    system.prepare()

    if system._site_ex:
        raise ValueError("The periodic slit preparation requires zero exterior sites.")

    total_surface_si = len(system._site_in)
    total_active_si = _active_silicon_count(system)
    initial_surface = _surface_composition(total_surface_si, system._pore.get_sites())
    _refresh_single_slit_tracking(system, total_surface_si, initial_surface)

    return _BaseSlitBuild(
        system=system,
        total_surface_si=total_surface_si,
        total_active_si=total_active_si,
        initial_surface=initial_surface,
    )


def _build_report(
    config,
    alpha_auto,
    alpha_effective,
    derived_surface_target,
    target_attempt,
    initial_surface,
):
    """Create a slit preparation report for a bare or functionalized build.

    Parameters
    ----------
    config : AmorphousSlitConfig
        Base slit configuration.
    alpha_auto : float
        Alpha derived from the current slit geometry.
    alpha_effective : float
        Alpha value actually used for target conversion.
    derived_surface_target : SiliconStateFractions
        Surface-only fractions derived from the experimental target.
    target_attempt : _SurfaceTargetAttempt
        Successful target realization payload.
    initial_surface : SiliconStateComposition
        Surface composition before custom condensation.

    Returns
    -------
    report : SlitPreparationReport
        Report summarizing the slit build.
    """
    system = target_attempt.system
    system._pore.set_name(config.name)
    wall_thickness = (system.box()[1] - config.slit_width_nm) / 2
    diagnostics = system._pore.get_surface_preparation_diagnostics()

    return SlitPreparationReport(
        name=config.name,
        temperature_k=config.temperature_k,
        box_nm=system.box(),
        slit_width_nm=config.slit_width_nm,
        wall_thickness_nm=wall_thickness,
        site_ex=len(system._site_ex),
        siloxane_bridges=target_attempt.siloxane_bridges,
        siloxane_distance_range_nm=tuple(config.siloxane_distance_range_nm),
        surface_fraction_tolerance=config.surface_fraction_tolerance,
        alpha_auto=alpha_auto,
        alpha_effective=alpha_effective,
        used_surface_tolerance=target_attempt.used_surface_tolerance,
        experimental_target=config.surface_target,
        derived_surface_target=derived_surface_target,
        initial_surface=initial_surface,
        target_surface=target_attempt.target_surface,
        prepared_surface=target_attempt.prepared_surface,
        final_surface=target_attempt.final_surface,
        preparation_diagnostics=diagnostics,
    )


def prepare_amorphous_slit_surface(config=None):
    """Prepare a bare amorphous slit surface from alpha-aware experimental data.

    Parameters
    ----------
    config : AmorphousSlitConfig, optional
        Bare slit preparation configuration.

    Returns
    -------
    result : SlitPreparationResult
        Attach-ready bare slit system and its preparation report.

    Raises
    ------
    ValueError
        Raised when the provided target contains non-zero ``T2/T3`` fractions
        or when the slit cannot realize the requested bare surface.
    """
    config = config if config is not None else AmorphousSlitConfig()
    if config.surface_target.t2_fraction or config.surface_target.t3_fraction:
        raise ValueError("Bare slit preparation requires t2_fraction == 0 and t3_fraction == 0.")

    build = _build_base_slit_system(config)
    alpha_auto, alpha_effective = _effective_alpha(
        build.total_surface_si,
        build.total_active_si,
        config.surface_target,
    )
    derived_surface_target = _surface_target_from_experimental(
        config.surface_target,
        alpha_effective,
    )
    exact_target = _nearest_integer_composition(build.total_surface_si, derived_surface_target)
    target_attempt = _realize_surface_target(
        build.system,
        build.total_surface_si,
        build.initial_surface,
        derived_surface_target,
        exact_target,
        config.surface_fraction_tolerance,
        tuple(config.siloxane_distance_range_nm),
        ligand=None,
    )
    report = _build_report(
        config,
        alpha_auto,
        alpha_effective,
        derived_surface_target,
        target_attempt,
        build.initial_surface,
    )
    return SlitPreparationResult(system=target_attempt.system, report=report)


def prepare_functionalized_amorphous_slit_surface(config):
    """Prepare an exactly targeted functionalized amorphous slit surface.

    Parameters
    ----------
    config : FunctionalizedAmorphousSlitConfig
        Functionalized slit configuration.

    Returns
    -------
    result : FunctionalizedSlitResult
        Attach-ready functionalized slit system and its preparation report.
    """
    slit_config = config.slit_config
    build = _build_base_slit_system(slit_config)
    alpha_auto, alpha_effective = _effective_alpha(
        build.total_surface_si,
        build.total_active_si,
        slit_config.surface_target,
    )
    derived_surface_target = _surface_target_from_experimental(
        slit_config.surface_target,
        alpha_effective,
    )
    exact_target = _nearest_integer_composition(build.total_surface_si, derived_surface_target)
    target_attempt = _realize_surface_target(
        build.system,
        build.total_surface_si,
        build.initial_surface,
        derived_surface_target,
        exact_target,
        slit_config.surface_fraction_tolerance,
        tuple(slit_config.siloxane_distance_range_nm),
        ligand=config.ligand,
    )
    report = _build_report(
        slit_config,
        alpha_auto,
        alpha_effective,
        derived_surface_target,
        target_attempt,
        build.initial_surface,
    )
    return FunctionalizedSlitResult(system=target_attempt.system, report=report)


def write_bare_amorphous_slit(
    output_dir,
    config=None,
    write_object_files=False,
    write_pdb=False,
    write_pdb_conect=False,
):
    """Prepare, finalize, and store a bare amorphous silica slit.

    Parameters
    ----------
    output_dir : str
        Output directory for the generated slit files and JSON report.
    config : AmorphousSlitConfig, optional
        Bare slit preparation configuration.
    write_object_files : bool, optional
        When ``True``, also serialize the finalized pore structure and full
        :class:`porems.system.PoreKit` state as ``.obj`` files. The default is
        ``False`` so object exports remain an explicit opt-in.
    write_pdb : bool, optional
        When ``True``, also write a PDB structure file for inspection.
    write_pdb_conect : bool, optional
        When ``True``, emit inspection-oriented ``CONECT`` records in the
        written PDB file. This flag implies PDB output.

    Returns
    -------
    result : SlitPreparationResult
        Finalized bare slit system and its preparation report.
    """
    result = prepare_amorphous_slit_surface(config=config)
    pms.utils.mkdirp(output_dir)

    result.system.finalize()
    result.system.store(
        output_dir,
        write_object_files=write_object_files,
        write_pdb=write_pdb,
        write_pdb_conect=write_pdb_conect,
    )

    report_path = os.path.join(output_dir, f"{result.report.name}_report.json")
    with open(report_path, "w") as file_out:
        json.dump(asdict(result.report), file_out, indent=2)

    return result


def write_functionalized_amorphous_slit(
    output_dir,
    config,
    write_object_files=False,
    write_pdb=False,
    write_pdb_conect=False,
):
    """Prepare, finalize, and store a functionalized amorphous silica slit.

    Parameters
    ----------
    output_dir : str
        Output directory for the generated slit files and JSON report.
    config : FunctionalizedAmorphousSlitConfig
        Functionalized slit preparation configuration.
    write_object_files : bool, optional
        When ``True``, also serialize the finalized pore structure and full
        :class:`porems.system.PoreKit` state as ``.obj`` files. The default is
        ``False`` so object exports remain an explicit opt-in.
    write_pdb : bool, optional
        When ``True``, also write a PDB structure file for inspection.
    write_pdb_conect : bool, optional
        When ``True``, emit inspection-oriented ``CONECT`` records in the
        written PDB file. This flag implies PDB output.

    Returns
    -------
    result : FunctionalizedSlitResult
        Finalized functionalized slit system and its preparation report.
    """
    result = prepare_functionalized_amorphous_slit_surface(config)
    pms.utils.mkdirp(output_dir)

    result.system.finalize()
    result.system.store(
        output_dir,
        write_object_files=write_object_files,
        write_pdb=write_pdb,
        write_pdb_conect=write_pdb_conect,
    )

    report_path = os.path.join(output_dir, f"{result.report.name}_report.json")
    with open(report_path, "w") as file_out:
        json.dump(asdict(result.report), file_out, indent=2)

    return result
