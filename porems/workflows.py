################################################################################
# Workflow Helpers                                                             #
#                                                                              #
"""High-level helpers for building reproducible PoreMS study systems."""
################################################################################


import copy
import json
import os

from dataclasses import asdict, dataclass, field

import porems as pms


@dataclass(frozen=True)
class SurfaceCompositionTarget:
    """Target fractions for the exposed surface silicon population.

    Parameters
    ----------
    q2_fraction : float
        Fraction of geminal silanol sites ``Q2`` over all exposed surface
        silicon atoms.
    q3_fraction : float
        Fraction of single silanol sites ``Q3`` over all exposed surface
        silicon atoms.
    q4_fraction : float
        Fraction of fully condensed surface silicon sites ``Q4`` over all
        exposed surface silicon atoms.
    """

    q2_fraction: float
    q3_fraction: float
    q4_fraction: float

    def __post_init__(self):
        total = self.q2_fraction + self.q3_fraction + self.q4_fraction
        if min(self.q2_fraction, self.q3_fraction, self.q4_fraction) < 0:
            raise ValueError("Surface fractions must be non-negative.")
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Surface fractions must add up to 1.0.")


@dataclass(frozen=True)
class BareAmorphousSlitConfig:
    """Configuration for a periodic bare amorphous silica slit.

    Parameters
    ----------
    name : str, optional
        Base name used for stored study files.
    slit_width_nm : float, optional
        Width of the slit in nanometers.
    repeat_y : int, optional
        Number of amorphous template copies stacked along the slit-normal
        direction.
    temperature_k : float, optional
        Target simulation temperature in Kelvin.
    surface_target : SurfaceCompositionTarget, optional
        Target exposed-surface composition.
    amorph_bond_range_nm : tuple, optional
        Accepted ``Si-O`` bond-length range for the amorphous template.
    siloxane_distance_range_nm : tuple, optional
        Accepted ``Si-Si`` distance range for custom siloxane formation.
    template_split_pairs : tuple, optional
        Template-specific bond pairs that must be disconnected after
        reconstructing the amorphous connectivity matrix.
    """

    name: str = "bare_amorphous_silica_slit"
    slit_width_nm: float = 7.0
    repeat_y: int = 2
    temperature_k: float = 300.0
    surface_target: SurfaceCompositionTarget = field(
        default_factory=lambda: SurfaceCompositionTarget(0.069, 0.681, 0.25)
    )
    amorph_bond_range_nm: tuple[float, float] = (0.160 - 0.02, 0.160 + 0.02)
    siloxane_distance_range_nm: tuple[float, float] = (0.40, 0.65)
    template_split_pairs: tuple[tuple[int, int], ...] = ((57790, 2524),)

    def __post_init__(self):
        if self.slit_width_nm <= 0:
            raise ValueError("The slit width must be positive.")
        if self.repeat_y < 2:
            raise ValueError("The amorphous slit requires at least two y-repeats.")
        if self.temperature_k <= 0:
            raise ValueError("The temperature must be positive.")


@dataclass(frozen=True)
class SurfaceComposition:
    """Surface-site counts for a slit model.

    Parameters
    ----------
    total_surface_si : int
        Total number of exposed surface silicon atoms tracked by the workflow.
    q2_sites : int
        Number of geminal silanol surface silicon sites.
    q3_sites : int
        Number of single silanol surface silicon sites.
    q4_sites : int
        Number of fully condensed surface silicon sites.
    """

    total_surface_si: int
    q2_sites: int
    q3_sites: int
    q4_sites: int

    def __post_init__(self):
        if min(self.total_surface_si, self.q2_sites, self.q3_sites, self.q4_sites) < 0:
            raise ValueError("Surface-site counts must be non-negative.")
        if self.q2_sites + self.q3_sites + self.q4_sites != self.total_surface_si:
            raise ValueError("Surface-site counts must add up to the total surface silicon count.")

    @property
    def q2_fraction(self):
        """Return the ``Q2`` fraction."""
        return self.q2_sites / self.total_surface_si if self.total_surface_si else 0.0

    @property
    def q3_fraction(self):
        """Return the ``Q3`` fraction."""
        return self.q3_sites / self.total_surface_si if self.total_surface_si else 0.0

    @property
    def q4_fraction(self):
        """Return the ``Q4`` fraction."""
        return self.q4_sites / self.total_surface_si if self.total_surface_si else 0.0


@dataclass(frozen=True)
class SlitBuildReport:
    """Summary of the generated bare amorphous slit system.

    Parameters
    ----------
    name : str
        Study-system name.
    temperature_k : float
        Target simulation temperature in Kelvin.
    box_nm : list
        Final periodic simulation box in nanometers.
    slit_width_nm : float
        Requested slit width.
    wall_thickness_nm : float
        Silica wall thickness on each side of the slit.
    site_ex : int
        Number of exterior surface sites.
    siloxane_bridges : int
        Number of siloxane bridges introduced during surface editing.
    siloxane_distance_range_nm : tuple
        Accepted ``Si-Si`` distance range used during custom surface editing.
    initial_surface : SurfaceComposition
        Surface composition before custom condensation.
    target_surface : SurfaceComposition
        Requested final surface composition.
    final_surface : SurfaceComposition
        Final surface composition after custom condensation.
    """

    name: str
    temperature_k: float
    box_nm: list[float]
    slit_width_nm: float
    wall_thickness_nm: float
    site_ex: int
    siloxane_bridges: int
    siloxane_distance_range_nm: tuple[float, float]
    initial_surface: SurfaceComposition
    target_surface: SurfaceComposition
    final_surface: SurfaceComposition


@dataclass
class SlitBuildResult:
    """Build result containing the generated PoreMS system and summary report.

    Parameters
    ----------
    system : PoreKit
        Generated slit system.
    report : SlitBuildReport
        Summary of the generated geometry and surface composition.
    """

    system: pms.PoreKit
    report: SlitBuildReport


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
    split_pairs : tuple
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
    pos_a : list
        First position vector.
    pos_b : list
        Second position vector.

    Returns
    -------
    distance : float
        Euclidean distance.
    """
    return pms.geom.length(pms.geom.vector(pos_a, pos_b))


def _surface_composition(total_surface_si, sites):
    """Summarize the current surface ``Q`` populations.

    Parameters
    ----------
    total_surface_si : int
        Total number of exposed surface silicon atoms tracked by the workflow.
    sites : dict
        Current PoreMS binding-site dictionary.

    Returns
    -------
    composition : SurfaceComposition
        Surface-site counts.
    """
    q2_sites = sum(1 for site in sites.values() if site["type"] == "in" and len(site["o"]) == 2)
    q3_sites = sum(1 for site in sites.values() if site["type"] == "in" and len(site["o"]) == 1)

    return SurfaceComposition(
        total_surface_si=total_surface_si,
        q2_sites=q2_sites,
        q3_sites=q3_sites,
        q4_sites=total_surface_si - q2_sites - q3_sites,
    )


def _target_surface_counts(total_surface_si, initial_surface, target):
    """Convert target fractions into integer site counts.

    Parameters
    ----------
    total_surface_si : int
        Total number of exposed surface silicon atoms.
    initial_surface : SurfaceComposition
        Initial surface-site counts before custom condensation.
    target : SurfaceCompositionTarget
        Target surface fractions.

    Returns
    -------
    composition : SurfaceComposition
        Integer target counts compatible with the current slit.
    """
    ideal_q2 = total_surface_si * target.q2_fraction
    ideal_q3 = total_surface_si * target.q3_fraction

    best = None
    q2_start = max(0, int(round(ideal_q2)) - 3)
    q2_stop = min(total_surface_si, int(round(ideal_q2)) + 3) + 1
    q3_start = max(0, int(round(ideal_q3)) - 3)
    q3_stop = min(total_surface_si, int(round(ideal_q3)) + 3) + 1

    for q2_sites in range(q2_start, q2_stop):
        for q3_sites in range(q3_start, q3_stop):
            q4_sites = total_surface_si - q2_sites - q3_sites
            if q4_sites < 0:
                continue
            if q2_sites > initial_surface.q2_sites:
                continue
            if q3_sites % 2 != initial_surface.q3_sites % 2:
                continue

            error = (
                abs(q2_sites / total_surface_si - target.q2_fraction)
                + abs(q3_sites / total_surface_si - target.q3_fraction)
                + abs(q4_sites / total_surface_si - target.q4_fraction)
            )
            candidate = SurfaceComposition(total_surface_si, q2_sites, q3_sites, q4_sites)

            if best is None or error < best[0]:
                best = (error, candidate)

    if best is None:
        raise ValueError("Could not determine target surface counts compatible with the current slit.")

    return best[1]


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
        Slit system under construction.
    site_ids : list
        Surface silicon identifiers.
    distance_range : tuple
        Accepted ``Si-Si`` distance range for siloxane formation.

    Returns
    -------
    adjacency : dict
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
    sites : dict
        Current binding-site dictionary.
    adjacency : dict
        Precomputed slit neighbor graph.
    first_count : int
        Required number of free oxygen atoms on the first site.
    second_count : int
        Required number of free oxygen atoms on the second site.

    Returns
    -------
    pair : tuple or None
        Pair of silicon site identifiers, or ``None`` if no eligible pair was
        found.
    """
    for site_a in sorted(sites):
        if sites[site_a]["type"] != "in" or len(sites[site_a]["o"]) != first_count:
            continue

        for site_b, distance in adjacency.get(site_a, []):
            if site_b not in sites:
                continue
            if sites[site_b]["type"] != "in" or len(sites[site_b]["o"]) != second_count:
                continue
            if first_count == second_count and site_b < site_a:
                continue

            return (site_a, site_b)

    return None


def _siloxane_bridge_molecule(kit, pair):
    """Create the siloxane bridge molecule for a specific silicon pair.

    Parameters
    ----------
    kit : PoreKit
        Slit system under construction.
    pair : tuple
        Pair of silicon site identifiers.

    Returns
    -------
    molecule : Molecule
        Positioned siloxane bridge molecule.
    """
    molecule = pms.Molecule("siloxane", "SLX")
    molecule.add("O", [0, 0, 0], name="OM1")
    molecule.add("O", 0, r=0.09, name="OM1")

    molecule_axis = molecule.bond(0, 1)
    molecule.rotate(
        pms.geom.cross_product(molecule_axis, [0, 0, 1]),
        pms.geom.angle(molecule_axis, [0, 0, 1]),
    )
    molecule.zero()

    pos_a = kit._pore.get_block().pos(pair[0])
    pos_b = kit._pore.get_block().pos(pair[1])
    center_pos = [
        pos_a[dim] + (pos_b[dim] - pos_a[dim]) / 2 for dim in range(3)
    ]

    surface_axis = kit._pore.get_sites()[pair[0]]["normal"](center_pos)
    molecule.rotate(
        pms.geom.cross_product([0, 0, 1], surface_axis),
        -pms.geom.angle([0, 0, 1], surface_axis),
    )
    molecule.move(0, center_pos)
    molecule.delete(0)

    return molecule


def _bridge_pair(kit, pair, distance_range):
    """Create one siloxane bridge between two specific surface silicon sites.

    Parameters
    ----------
    kit : PoreKit
        Slit system under construction.
    pair : tuple
        Pair of silicon site identifiers.
    distance_range : tuple
        Accepted ``Si-Si`` distance range for siloxane formation.

    Returns
    -------
    bridge_count : int
        Number of siloxane bridges created, always one on success.
    """
    sites = kit._pore.get_sites()
    if pair[0] not in sites or pair[1] not in sites:
        raise ValueError("Cannot bridge a silicon pair that is no longer present in the site dictionary.")

    molecule = _siloxane_bridge_molecule(kit, pair)
    if "SLX" not in kit._pore._mol_dict["in"]:
        kit._pore._mol_dict["in"]["SLX"] = []
    kit._pore._mol_dict["in"]["SLX"].append(molecule)
    if molecule.get_short() not in kit._sort_list:
        kit._sort_list.append(molecule.get_short())

    for site_id in pair:
        oxygen_id = sites[site_id]["o"][0]
        kit._matrix.strip(oxygen_id)
        if len(sites[site_id]["o"]) == 2:
            sites[site_id]["o"].pop(0)
        else:
            del sites[site_id]

    return 1


def _consume_pair(adjacency, pair):
    """Remove a bridged silicon pair from the static slit adjacency graph.

    Parameters
    ----------
    adjacency : dict
        Precomputed slit neighbor graph.
    pair : tuple
        Silicon identifiers that have already been bridged once.
    """
    site_a, site_b = pair
    adjacency[site_a] = [item for item in adjacency.get(site_a, []) if item[0] != site_b]
    adjacency[site_b] = [item for item in adjacency.get(site_b, []) if item[0] != site_a]


def _refresh_single_slit_tracking(kit, total_surface_si):
    """Refresh slit-only site tracking after custom condensation.

    Parameters
    ----------
    kit : PoreKit
        Slit system under construction.
    total_surface_si : int
        Total number of exposed surface silicon atoms tracked by the workflow.
    """
    sites = kit._pore.get_sites()
    site_in = sorted(site for site, data in sites.items() if data["type"] == "in")

    kit._site_in = site_in
    kit._site_ex = []
    kit._si_pos_in = [[kit._pore.get_block().pos(site) for site in site_in]]
    kit._si_pos_ex = []
    kit.sites_shape = {0: site_in}
    kit._pore.sites_sl_shape = {0: site_in}

    composition = _surface_composition(total_surface_si, sites)
    siloxane_num = len(kit._pore.get_site_dict()["in"].get("SLX", []))
    kit._pore.sites_attach_mol = {
        0: {
            "SL": composition.q3_sites,
            "SLG": composition.q2_sites,
            "SLX": siloxane_num,
        }
    }


def _enforce_surface_target(kit, total_surface_si, target_surface, distance_range):
    """Condense the slit surface until the target ``Q2/Q3/Q4`` counts are met.

    Parameters
    ----------
    kit : PoreKit
        Slit system under construction.
    total_surface_si : int
        Total number of exposed surface silicon atoms tracked by the workflow.
    target_surface : SurfaceComposition
        Target surface-site counts.
    distance_range : tuple
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

    while current_surface.q2_sites > target_surface.q2_sites:
        q2_delta = current_surface.q2_sites - target_surface.q2_sites
        pair = _find_pair(sites, adjacency, 2, 1)

        if pair is None:
            if q2_delta < 2:
                raise ValueError("The slit surface cannot reach the requested Q2 count with the available siloxane pairs.")
            pair = _find_pair(sites, adjacency, 2, 2)
            if pair is None:
                raise ValueError("No remaining Q2/Q2 siloxane pair is available to reduce the Q2 population.")

        bridge_count += _bridge_pair(kit, pair, distance_range)
        _consume_pair(adjacency, pair)
        current_surface = _surface_composition(total_surface_si, sites)

    if current_surface.q3_sites < target_surface.q3_sites:
        raise ValueError("The slit surface contains fewer Q3 sites than requested after Q2 reduction.")

    while current_surface.q3_sites > target_surface.q3_sites:
        if (current_surface.q3_sites - target_surface.q3_sites) < 2:
            raise ValueError("The requested Q3 count is incompatible with the siloxane editing parity constraints.")

        pair = _find_pair(sites, adjacency, 1, 1)
        if pair is None:
            raise ValueError("No remaining Q3/Q3 siloxane pair is available to reach the requested Q3 count.")

        bridge_count += _bridge_pair(kit, pair, distance_range)
        _consume_pair(adjacency, pair)
        current_surface = _surface_composition(total_surface_si, sites)

    if current_surface != target_surface:
        raise ValueError("The slit surface could not be edited to the requested Q2/Q3/Q4 composition.")

    _refresh_single_slit_tracking(kit, total_surface_si)

    return bridge_count


def _workflow_notes(report):
    """Create Markdown notes for the external thymol setup steps.

    Parameters
    ----------
    report : SlitBuildReport
        Summary of the generated slit system.

    Returns
    -------
    notes : str
        Markdown notes for the next simulation steps.
    """
    return f"""# Bare Amorphous Silica / Thymol Workflow

## Silica Slit
- Generated fully periodic amorphous silica slit named `{report.name}`.
- Box: `{report.box_nm[0]:.3f} x {report.box_nm[1]:.3f} x {report.box_nm[2]:.3f} nm`
- Slit width: `{report.slit_width_nm:.3f} nm`
- Silica wall thickness on each side: `{report.wall_thickness_nm:.3f} nm`
- Exterior sites: `{report.site_ex}`
- Custom siloxane search window: `{report.siloxane_distance_range_nm[0]:.3f} - {report.siloxane_distance_range_nm[1]:.3f} nm`
- Surface target: `Q2={report.target_surface.q2_fraction:.3f}`, `Q3={report.target_surface.q3_fraction:.3f}`, `Q4={report.target_surface.q4_fraction:.3f}`
- Final surface counts: `Q2={report.final_surface.q2_sites}`, `Q3={report.final_surface.q3_sites}`, `Q4={report.final_surface.q4_sites}`
- Siloxane bridges introduced during editing: `{report.siloxane_bridges}`

## External Thymol Parametrization
- Parametrize thymol with `GAFF2 + AM1-BCC` using AmberTools.
- Convert the resulting topology to GROMACS format outside PoreMS.
- Do not use the incomplete bundled `antechamber.job` helper as an end-to-end workflow.

## Bulk Calibration
- Build a separate bulk thymol box.
- Run `NPT` at `{report.temperature_k:.1f} K` to determine the target liquid density.
- Use the bulk density when packing thymol into the slit.

## Slit MD
- Keep the silica framework rigid for the first campaign.
- Pack thymol into the slit only, not into the silica wall.
- Treat the two equivalent silica/thymol interfaces under PBC as intentional.
- Run:
  - energy minimization
  - short restrained equilibration
  - fixed-box `NVT` production at `{report.temperature_k:.1f} K`

## Analysis Plan
- Compute the thymol density profile along the slit normal.
- Compute the fraction of thymol in the first adsorption layer.
- Measure residence times in the first adsorption layer.
- Measure thymol orientation relative to the silica surface.
- Measure silica-thymol contacts and hydrogen-bond statistics.
- Measure thymol diffusion parallel to the slit plane.
- Treat this study as a relative adsorption-strength benchmark, not an evaporation-rate prediction.

## Validation Checks
- Confirm `site_ex = 0` and that no exterior molecules were generated.
- Confirm the slit width and wall thickness from the stored box dimensions.
- Confirm the final exposed-surface `Q2/Q3/Q4` counts match the requested target.
- Confirm thymol remains in the slit and does not penetrate the rigid silica wall during MD.
"""


def build_periodic_amorphous_slit(config=None):
    """Build a periodic bare amorphous silica slit with a target surface state.

    Parameters
    ----------
    config : BareAmorphousSlitConfig, optional
        Workflow configuration. Defaults to the study settings used in the
        first bare-silica thymol campaign.

    Returns
    -------
    result : SlitBuildResult
        Generated slit system and summary report.

    Raises
    ------
    ValueError
        Raised when the generated slit unexpectedly contains exterior surface
        sites or when the provided configuration is internally inconsistent.
    """
    config = config if config is not None else BareAmorphousSlitConfig()

    base = pms.Molecule(inp=_amorphous_template_path())
    replicated = _replicate_along_y(base, config.repeat_y)

    system = pms.PoreKit()
    system.structure(replicated)
    system.build(bonds=list(config.amorph_bond_range_nm))
    _duplicate_template_splits(system._matrix, base.get_num(), config.repeat_y, config.template_split_pairs)

    system.add_shape(system.shape_slit(config.slit_width_nm, centroid=system.centroid()), hydro=0)
    system.prepare()

    if system._site_ex:
        raise ValueError("The periodic slit workflow requires zero exterior sites.")

    total_surface_si = len(system._site_in)
    initial_surface = _surface_composition(total_surface_si, system._pore.get_sites())
    target_surface = _target_surface_counts(total_surface_si, initial_surface, config.surface_target)
    siloxane_bridges = _enforce_surface_target(
        system,
        total_surface_si,
        target_surface,
        tuple(config.siloxane_distance_range_nm),
    )

    system._pore.set_name(config.name)
    system.finalize()
    _refresh_single_slit_tracking(system, total_surface_si)

    final_surface = _surface_composition(total_surface_si, system._pore.get_sites())
    wall_thickness = (system.box()[1] - config.slit_width_nm) / 2
    report = SlitBuildReport(
        name=config.name,
        temperature_k=config.temperature_k,
        box_nm=system.box(),
        slit_width_nm=config.slit_width_nm,
        wall_thickness_nm=wall_thickness,
        site_ex=len(system._site_ex),
        siloxane_bridges=siloxane_bridges,
        siloxane_distance_range_nm=tuple(config.siloxane_distance_range_nm),
        initial_surface=initial_surface,
        target_surface=target_surface,
        final_surface=final_surface,
    )

    return SlitBuildResult(system=system, report=report)


def write_bare_amorphous_slit_study(output_dir, config=None):
    """Build and store a bare amorphous silica slit study.

    Parameters
    ----------
    output_dir : str
        Output directory for the generated slit files and workflow metadata.
    config : BareAmorphousSlitConfig, optional
        Workflow configuration.

    Returns
    -------
    result : SlitBuildResult
        Generated slit system and summary report.

    Raises
    ------
    ValueError
        Raised when :func:`build_periodic_amorphous_slit` cannot generate a
        valid periodic slit for the requested configuration.
    """
    result = build_periodic_amorphous_slit(config=config)
    pms.utils.mkdirp(output_dir)

    result.system.store(output_dir)

    report_path = os.path.join(output_dir, f"{result.report.name}_study.json")
    with open(report_path, "w") as file_out:
        json.dump(asdict(result.report), file_out, indent=2)

    notes_path = os.path.join(output_dir, f"{result.report.name}_next_steps.md")
    with open(notes_path, "w") as file_out:
        file_out.write(_workflow_notes(result.report))

    return result
