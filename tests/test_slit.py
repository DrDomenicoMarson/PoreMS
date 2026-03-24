from dataclasses import asdict, replace
import inspect
import json
import os
from pathlib import Path
import numpy as np
import pytest
import yaml

import porems as pms
import porems.slit as slit_mod
import porems.store as store_mod
import porems.topology as topo_mod
from porems._version import __version__ as EXPECTED_VERSION


def experimental_target_from_surface(surface_target, alpha, alpha_override=None):
    """Build an experimental all-silicon target from surface-only fractions.

    Parameters
    ----------
    surface_target : SiliconStateFractions
        Desired surface-only silicon-state fractions.
    alpha : float
        Surface-to-total silicon fraction used for the back-conversion.
    alpha_override : float or None, optional
        Explicit alpha override stored on the resulting experimental target.

    Returns
    -------
    target : ExperimentalSiliconStateTarget
        Experimental all-silicon target that maps back to ``surface_target``
        when the same alpha value is applied.
    """
    return pms.ExperimentalSiliconStateTarget(
        q2_fraction=alpha * surface_target.q2_fraction,
        q3_fraction=alpha * surface_target.q3_fraction,
        q4_fraction=alpha * surface_target.q4_fraction + (1.0 - alpha),
        t2_fraction=alpha * surface_target.t2_fraction,
        t3_fraction=alpha * surface_target.t3_fraction,
        alpha_override=alpha_override,
    )


def test_topology_parameter_helpers_are_exported_from_package_root():
    assert pms.GromacsAngleParameters is topo_mod.GromacsAngleParameters
    assert pms.GromacsBondParameters is topo_mod.GromacsBondParameters


def itp_atom_rows(itp_path):
    """Return parsed ``[ atoms ]`` rows from one topology file.

    Parameters
    ----------
    itp_path : str or Path
        Path to the generated ``.itp`` file.

    Returns
    -------
    rows : list[tuple[str, str, str, float]]
        Parsed rows as ``(atom_type, residue_name, atom_name, charge)``.
    """
    rows = []
    section = None
    with open(itp_path, "r") as file_in:
        for line in file_in:
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                section = stripped.strip("[]").strip().lower()
                continue
            if section == "atoms":
                fields = stripped.split()
                if len(fields) >= 7:
                    rows.append((fields[1], fields[3], fields[4], float(fields[6])))
    return rows


def graph_without_bond(graph, atom_a, atom_b):
    """Return a copy of ``graph`` with one bond removed.

    Parameters
    ----------
    graph : AssembledStructureGraph
        Source graph whose bond list should be filtered.
    atom_a : int
        First atom id of the bond to remove.
    atom_b : int
        Second atom id of the bond to remove.

    Returns
    -------
    graph : AssembledStructureGraph
        New graph with every matching bond removed.
    """
    bond_key = tuple(sorted((atom_a, atom_b)))
    filtered_bonds = [
        bond
        for bond in graph.bonds
        if (bond.atom_a, bond.atom_b) != bond_key
    ]
    return pms.AssembledStructureGraph.from_bonds(graph.atom_ids, filtered_bonds)


def teps_ligand(repo_root):
    """Return the local explicit-bond TEPS ligand used for slit smoke tests.

    Parameters
    ----------
    repo_root : Path
        Repository root directory.

    Returns
    -------
    ligand : Molecule
        TEPS ligand loaded from the checked-in PDB fixture.
    """
    teps_path = Path(repo_root) / "scripts" / "TEPS.pdb"
    return pms.Molecule("TEPS", "TEPS", str(teps_path))


def bundled_tms_template_path():
    """Return the checked-in legacy TMS flat topology path.

    Returns
    -------
    itp_path : Path
        Package path to the legacy checked-in TMS flat topology bundle.
    """
    return Path(pms.__file__).resolve().parent / "templates" / "tms_slit.itp"


def explicit_tms_geminal_cross_terms():
    """Return explicit generated geminal cross terms for TMS export tests.

    Returns
    -------
    cross_terms : SilaneGeminalCrossTerms
        Deterministic geminal cross terms used by the functionalized export
        tests.
    """
    return pms.SilaneGeminalCrossTerms(
        first_ligand_atom_name="O1",
        geminal_oxygen_mount_ligand_angle=topo_mod.GromacsAngleParameters.harmonic(
            angle_deg=117.65432,
            force_constant=432.123456,
        ),
        geminal_dihedrals=(
            pms.GeminalMountDihedralSpec(
                fourth_atom_name="Si2",
                function=1,
                parameters=("12.34567", "0.98765", "2"),
            ),
        ),
    )


def explicit_tms_topology_config(
    tmp_path,
    total_charge=0.825,
    include_geminal_terms=True,
    junction_parameters=None,
    source_itp_path=None,
):
    """Return an explicit TMS topology config for full-slab export tests.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory used to materialize corrected flat topology files.
    total_charge : float, optional
        Target total charge written onto the base parsed TMS fragment.
    include_geminal_terms : bool, optional
        True to include explicit generated geminal cross terms on the returned
        config.
    junction_parameters : SlitJunctionParameters or None, optional
        Optional explicit legacy junction-parameter override carried by the
        returned topology config.
    source_itp_path : Path or None, optional
        Optional explicit source ``.itp`` path. When omitted, the checked-in
        legacy TMS bundle is used as the source text.

    Returns
    -------
    topology : SilaneTopologyConfig
        Explicit TMS topology config suitable for the functionalized slit
        exporter.
    """
    source_itp_path = (
        bundled_tms_template_path()
        if source_itp_path is None
        else Path(source_itp_path)
    )
    bundle = topo_mod.parse_flat_itp(
        source_itp_path,
        moleculetype_name="TMS",
    )
    charge_delta = total_charge - bundle.total_charge()
    corrected_atoms = []
    for atom in bundle.moleculetype.atoms:
        charge = float(atom.charge)
        if atom.atom_name == "Si1":
            charge += charge_delta
        corrected_atoms.append(
            replace(
                atom,
                charge=f"{charge:.6f}",
            )
        )

    corrected_bundle_path = Path(tmp_path) / "tms_explicit_charge_target.itp"
    with open(corrected_bundle_path, "w") as file_out:
        file_out.write(
            topo_mod.render_itp(
                bundle.atomtypes,
                replace(bundle.moleculetype, atoms=tuple(corrected_atoms)),
            )
        )

    topology_kwargs = {
        "itp_path": str(corrected_bundle_path),
        "moleculetype_name": "TMS",
        "geminal_cross_terms": (
            explicit_tms_geminal_cross_terms()
            if include_geminal_terms
            else None
        ),
    }
    if junction_parameters is not None:
        topology_kwargs["junction_parameters"] = junction_parameters
    return pms.SilaneTopologyConfig(**topology_kwargs)


def cif_loop_rows(cif_text, first_tag):
    """Return one simple whitespace-delimited mmCIF loop.

    Parameters
    ----------
    cif_text : str
        Full mmCIF document text.
    first_tag : str
        First tag of the loop that should be extracted.

    Returns
    -------
    result : tuple[list[str], list[list[str]]]
        Tuple ``(tags, rows)`` for the requested loop.

    Raises
    ------
    AssertionError
        Raised when the requested loop is not present in ``cif_text``.
    """
    lines = [line.rstrip() for line in cif_text.splitlines()]
    index = 0
    while index < len(lines):
        if lines[index] != "loop_":
            index += 1
            continue

        index += 1
        tags = []
        while index < len(lines) and lines[index].startswith("_"):
            tags.append(lines[index])
            index += 1

        rows = []
        while index < len(lines):
            line = lines[index]
            if not line or line == "#" or line == "loop_":
                break
            rows.append(line.split())
            index += 1

        if tags and tags[0] == first_tag:
            return tags, rows

    raise AssertionError(f"mmCIF loop starting with {first_tag!r} was not found.")


def naive_slit_adjacency(kit, site_ids, distance_range):
    """Return the original loop-based slit adjacency reference."""

    adjacency = {site: [] for site in site_ids}
    positions = {site: kit._pore.get_block().pos(site) for site in site_ids}

    for site_index, site_a in enumerate(site_ids):
        for site_b in site_ids[site_index + 1 :]:
            if slit_mod._are_sites_directly_connected(kit._matrix, site_a, site_b):
                continue

            distance = slit_mod._site_distance(positions[site_a], positions[site_b])
            if distance_range[0] <= distance <= distance_range[1]:
                adjacency[site_a].append((site_b, distance))
                adjacency[site_b].append((site_a, distance))

    for site in adjacency:
        adjacency[site].sort(key=lambda item: (item[1], item[0]))

    return adjacency


def naive_bridge_local_ids(kit, pair):
    """Return the local steric graph used by the original bridge scorer."""

    matrix = kit._matrix.get_matrix()
    frontier = list(pair)
    local_ids = set(pair)
    for _depth in range(slit_mod._BRIDGE_STERIC_GRAPH_DEPTH):
        next_frontier = []
        for atom_id in frontier:
            for neighbor_id in matrix[atom_id]["atoms"]:
                if neighbor_id not in local_ids:
                    local_ids.add(neighbor_id)
                    next_frontier.append(neighbor_id)
        if not next_frontier:
            break
        frontier = next_frontier
    return local_ids


def naive_bridge_clearance(kit, pair, bridge_position, local_only):
    """Return the original loop-based bridge clearance reference."""

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

    atom_ids = naive_bridge_local_ids(kit, pair) if local_only else matrix.keys()
    min_clearance = float("inf")
    for atom_id in atom_ids:
        if atom_id in excluded_ids:
            continue

        atom_type = block.get_atom_type(atom_id)
        min_distance = slit_mod._BRIDGE_MIN_CLEARANCE_BY_TYPE_NM.get(atom_type, 0.18)
        delta = slit_mod._minimum_image_vector(bridge_position, block.pos(atom_id), box)

        if any(abs(component) > slit_mod._BRIDGE_STERIC_DISTANCE_CUTOFF_NM for component in delta):
            continue

        clearance = pms.geom.length(delta) - min_distance
        if clearance < min_clearance:
            min_clearance = clearance
            if min_clearance < 0:
                return min_clearance

    return (
        min_clearance
        if min_clearance != float("inf")
        else slit_mod._BRIDGE_STERIC_DISTANCE_CUTOFF_NM
    )


class RecordingProgressBar:
    """Simple test double for slit progress-bar instrumentation tests."""

    def __init__(self, total, desc, unit, leave):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.leave = leave
        self.n = 0
        self.closed = False
        self.descriptions = [desc]

    def update(self, value=1):
        self.n += value

    def set_description_str(self, desc, refresh=True):
        del refresh
        self.desc = desc
        self.descriptions.append(desc)

    def close(self):
        self.closed = True


@pytest.fixture(scope="class", autouse=True)
def _bare_slit_case_context(request, bare_slit_context):
    """Expose the shared bare-slit context as class attributes."""

    if request.cls is None:
        return

    request.cls.output_dir = str(bare_slit_context.output_dir)
    request.cls.surface_target = bare_slit_context.surface_target
    request.cls.config = bare_slit_context.config
    request.cls.prepared_result = bare_slit_context.prepared_result
    request.cls.prepared_report = bare_slit_context.prepared_report
    request.cls.stored_result = bare_slit_context.stored_result
    request.cls.stored_report = bare_slit_context.stored_report


class TestSurfacePreparationValidation:
    def test_prepare_removes_orphan_oxygen_from_active_matrix(self):
        mol = pms.Molecule("orphan_oxygen")
        mol.add("O", [0.0, 0.0, 0.0], name="OM1")
        matrix = pms.Matrix([[0, []]])
        pore = pms.Pore(mol, matrix)

        pore.prepare()

        assert 0 not in matrix.get_matrix()
        assert pore.get_surface_preparation_diagnostics().removed_orphan_oxygen == 1

    def test_prepare_removes_invalid_oxygen_connectivity(self):
        mol = pms.Molecule("invalid_oxygen")
        mol.add("O", [0.0, 0.0, 0.0], name="OM1")
        mol.add("H", [0.1, 0.0, 0.0], name="H1")
        matrix = pms.Matrix([[0, [1]]])
        pore = pms.Pore(mol, matrix)

        pore.prepare()

        assert 0 not in matrix.get_matrix()
        assert pore.get_surface_preparation_diagnostics().removed_invalid_oxygen == 1

    def test_objectify_accepts_only_valid_framework_oxygen(self):
        valid = pms.Molecule("valid_framework_oxygen")
        valid.add("O", [0.0, 0.0, 0.0], name="OM1")
        valid.add("Si", [0.16, 0.0, 0.0], name="SI1")
        valid.add("Si", [-0.16, 0.0, 0.0], name="SI2")
        valid_matrix = pms.Matrix([[0, [1, 2]]])
        valid_pore = pms.Pore(valid, valid_matrix)

        mols = valid_pore.objectify([0])

        assert len(mols) == 1
        assert mols[0].get_short() == "OM"

        invalid = pms.Molecule("invalid_framework_oxygen")
        invalid.add("O", [0.0, 0.0, 0.0], name="OM1")
        invalid.add("Si", [0.16, 0.0, 0.0], name="SI1")
        invalid_matrix = pms.Matrix([[0, [1]]])
        invalid_pore = pms.Pore(invalid, invalid_matrix)

        with pytest.raises(ValueError):
            invalid_pore.objectify([0])


@pytest.mark.usefixtures("_bare_slit_case_context")
class TestAmorphousSlitPreparation:
    @staticmethod
    def _build_uncondensed_slit(config):
        """Build a prepared slit before custom Q-state enforcement.

        Parameters
        ----------
        config : AmorphousSlitConfig
            Slit preparation settings used to build the uncondensed surface.

        Returns
        -------
        system : PoreKit
            Prepared slit system with the raw silanol surface still intact.
        """
        base = pms.Molecule(inp=slit_mod._amorphous_template_path())
        replicated = slit_mod._replicate_along_y(base, config.repeat_y)

        system = pms.PoreKit()
        system.structure(replicated)
        system.build(bonds=list(config.amorph_bond_range_nm))
        slit_mod._duplicate_template_splits(
            system._matrix,
            base.get_num(),
            config.repeat_y,
            config.template_split_pairs,
        )

        system.add_shape(
            system.shape_slit(config.slit_width_nm, centroid=system.centroid()),
            hydro=0,
        )
        system.prepare()
        return system

    def test_periodic_slit_geometry(self):
        assert self.prepared_report.site_ex == 0
        assert self.prepared_result.system._site_ex == []
        assert self.prepared_result.system._pore.get_site_dict()["ex"] == {}
        assert isinstance(next(iter(self.prepared_result.system._pore.get_sites().values())), pms.BindingSite)

        expected_box = [9.605, 19.210, 9.605]
        for actual, expected in zip(self.prepared_report.box_nm, expected_box):
            assert actual == pytest.approx(expected, abs=10 ** (-(3)))

        assert self.prepared_report.slit_width_nm == pytest.approx(7.0, abs=10 ** (-(3)))
        assert self.prepared_report.wall_thickness_nm == pytest.approx(6.105, abs=10 ** (-(3)))
        assert self.prepared_report.siloxane_distance_range_nm == (0.4, 0.65)
        assert sorted(self.prepared_result.system._pore.sites_sl_shape) == [0]

    def test_default_silica_topology_returns_independent_copies_with_provenance(self):
        model_a = pms.default_silica_topology()
        model_b = pms.default_silica_topology()

        assert isinstance(model_a, pms.SilicaTopologyModel)
        assert isinstance(model_b, pms.SilicaTopologyModel)
        assert model_a is not model_b
        assert model_a.bond_terms.framework_si_o is not model_b.bond_terms.framework_si_o

        model_a.bond_terms.framework_si_o.force_constant = 123456.0

        assert model_b.bond_terms.framework_si_o.force_constant == pytest.approx(119244.0)
        assert model_b.atomtypes.framework_silicon.origin == "doi:10.1021/cm500365c"
        assert model_b.angle_terms.graft_oxygen_mount_oxygen.origin == "doi:10.1021/cm500365c"
        assert model_b.angle_terms.graft_scaffold_si_scaffold_o_mount.angle_deg == pytest.approx(149.0)
        assert model_b.angle_terms.graft_oxygen_mount_oxygen.angle_deg == pytest.approx(109.5)

    def test_default_legacy_slit_junction_parameters_match_default_silica_model(self):
        model = pms.default_silica_topology()
        junctions = pms.SlitJunctionParameters()

        assert junctions.mount_scaffold_bond.parameters == (
            model.bond_terms.graft_mount_scaffold_si_o.to_gromacs_parameters().parameters
        )
        assert junctions.scaffold_si_scaffold_o_mount_angle.parameters == (
            model.angle_terms.graft_scaffold_si_scaffold_o_mount.to_gromacs_parameters().parameters
        )
        assert junctions.oxygen_mount_oxygen_angle.parameters == (
            model.angle_terms.graft_oxygen_mount_oxygen.to_gromacs_parameters().parameters
        )

    def test_silica_topology_serialization_helpers_return_readable_structures(self):
        model = pms.default_silica_topology()

        dict_data = model.to_dict()
        json_data = json.loads(model.to_json())
        yaml_data = yaml.safe_load(model.to_yaml())

        assert dict_data["atomtypes"]["framework_silicon"]["name"] == "SI"
        assert dict_data["bond_terms"]["framework_si_o"]["force_constant"] == pytest.approx(119244.0)
        assert "origin" in dict_data["angle_terms"]["graft_oxygen_mount_oxygen"]
        assert json_data == dict_data
        assert yaml_data == dict_data

    def test_bare_results_expose_resolved_silica_topology(self):
        assert isinstance(self.prepared_result.silica_topology, pms.SilicaTopologyModel)
        assert isinstance(self.stored_result.silica_topology, pms.SilicaTopologyModel)
        assert self.prepared_result.bare_charge_diagnostics is None
        assert isinstance(self.stored_result.bare_charge_diagnostics, pms.BareSilicaChargeDiagnostics)
        assert self.prepared_result.silica_topology.bond_terms.framework_si_o.force_constant == pytest.approx(119244.0)

    def test_bare_charge_diagnostics_match_surface_roles_and_are_neutral(self):
        diagnostics = self.stored_result.bare_charge_diagnostics

        assert diagnostics is not None
        assert diagnostics.is_neutral
        assert diagnostics.coordination_identity_holds
        assert diagnostics.total_charge == pytest.approx(0.0, abs=1e-8)
        assert diagnostics.coordination_identity_delta == 0
        assert diagnostics.silanol_site_count == self.stored_report.final_surface.q3_sites
        assert diagnostics.geminal_site_count == self.stored_report.final_surface.q2_sites
        assert diagnostics.silanol_silicon.atom_count == self.stored_report.final_surface.q3_sites
        assert diagnostics.silanol_oxygen.atom_count == self.stored_report.final_surface.q3_sites
        assert diagnostics.silanol_hydrogen.atom_count == self.stored_report.final_surface.q3_sites
        assert diagnostics.geminal_silicon.atom_count == self.stored_report.final_surface.q2_sites
        assert diagnostics.geminal_oxygen.atom_count == 2 * self.stored_report.final_surface.q2_sites
        assert diagnostics.geminal_hydrogen.atom_count == 2 * self.stored_report.final_surface.q2_sites

    def test_store_bare_charge_diagnostics_matches_stored_result(self):
        diagnostics = pms.Store(
            self.stored_result.system._pore,
            sort_list=self.stored_result.system._sort_list,
        ).bare_slit_charge_diagnostics(
            silica_topology=self.stored_result.silica_topology,
        )

        assert diagnostics == self.stored_result.bare_charge_diagnostics

    def test_bare_charge_diagnostics_satisfy_coordination_identity(self):
        diagnostics = self.stored_result.bare_charge_diagnostics

        assert diagnostics is not None
        assert diagnostics.coordination_identity_left == 4 * diagnostics.total_silicon_count
        assert diagnostics.coordination_identity_right == (
            2 * diagnostics.framework_oxygen.atom_count
            + diagnostics.total_hydroxyl_count
        )
        assert diagnostics.coordination_identity_left == diagnostics.coordination_identity_right

    def test_prepared_surface_composition_matches_target(self):
        expected_alpha_auto = (
            self.prepared_report.prepared_surface.total_surface_si
            / slit_mod._active_silicon_count(self.prepared_result.system)
        )
        assert self.prepared_report.final_surface == self.prepared_report.target_surface
        assert self.prepared_report.prepared_surface == self.prepared_report.target_surface
        assert self.prepared_report.prepared_surface.total_surface_si == 954
        assert self.prepared_report.prepared_surface.q2_sites == 66
        assert self.prepared_report.prepared_surface.q3_sites == 650
        assert self.prepared_report.prepared_surface.q4_sites == 238
        assert self.prepared_report.prepared_surface.t2_sites == 0
        assert self.prepared_report.prepared_surface.t3_sites == 0
        assert not (self.prepared_report.used_surface_tolerance)
        assert self.prepared_report.surface_fraction_tolerance == pytest.approx(0.005, abs=1e-7)
        assert self.prepared_report.alpha_auto == pytest.approx(expected_alpha_auto, abs=10 ** (-(8)))
        assert self.prepared_report.alpha_effective == 1.0
        assert self.prepared_report.derived_surface_target == pms.SiliconStateFractions(0.069, 0.681, 0.25)
        assert self.prepared_report.final_surface.q2_fraction == pytest.approx(self.prepared_report.derived_surface_target.q2_fraction, abs=1e-3)
        assert self.prepared_report.final_surface.q3_fraction == pytest.approx(self.prepared_report.derived_surface_target.q3_fraction, abs=1e-3)
        assert self.prepared_report.final_surface.q4_fraction == pytest.approx(self.prepared_report.derived_surface_target.q4_fraction, abs=1e-3)
        assert (
            self.prepared_report.preparation_diagnostics.final_surface_oxygen_handles
            == self.prepared_report.final_surface.q3_sites
            + 2 * self.prepared_report.final_surface.q2_sites
        )
        assert self.prepared_report.preparation_diagnostics.final_framework_oxygen == len(self.prepared_result.system._pore.get_mol_dict()["OM"])
        assert self.prepared_report.preparation_diagnostics.stripped_silicon_total > 0
        assert self.prepared_report.preparation_diagnostics.removed_orphan_oxygen > 0
        assert self.prepared_report.preparation_diagnostics.inserted_bridge_oxygen == 235
        assert "SLX" not in self.prepared_result.system._pore.get_mol_dict()

    def test_inserted_bridge_oxygen_respects_local_clearance_threshold(self):
        history = self.prepared_result.system._pore.get_surface_edit_history()
        bridge_ids = [
            record.atom_id
            for record in history
            if record.reason == "inserted_bridge_oxygen"
        ]
        matrix = self.prepared_result.system._matrix.get_matrix()
        block = self.prepared_result.system._pore.get_block()
        box = block.get_box()

        assert len(bridge_ids) == self.prepared_report.preparation_diagnostics.inserted_bridge_oxygen

        for bridge_id in bridge_ids:
            bonded_ids = set(matrix[bridge_id]["atoms"]) | {bridge_id}
            for atom_id in matrix:
                if atom_id in bonded_ids:
                    continue

                delta = slit_mod._minimum_image_vector(
                    block.pos(bridge_id),
                    block.pos(atom_id),
                    box,
                )
                if any(
                    abs(component) > slit_mod._BRIDGE_STERIC_DISTANCE_CUTOFF_NM
                    for component in delta
                ):
                    continue

                min_distance = slit_mod._BRIDGE_MIN_CLEARANCE_BY_TYPE_NM.get(
                    block.get_atom_type(atom_id),
                    0.18,
                )
                clearance = pms.geom.length(delta) - min_distance
                assert clearance >= -1e-9, f"Bridge oxygen {bridge_id} is too close to atom {atom_id}."

    def test_slit_adjacency_matches_naive_reference(self):
        site_ids = sorted(self.prepared_result.system._site_in)
        adjacency = slit_mod._build_slit_site_adjacency(
            self.prepared_result.system,
            site_ids,
            self.config.siloxane_distance_range_nm,
        )
        reference = naive_slit_adjacency(
            self.prepared_result.system,
            site_ids,
            self.config.siloxane_distance_range_nm,
        )

        adjacency_pairs = {
            tuple(sorted((site_a, site_b)))
            for site_a, neighbors in adjacency.items()
            for site_b, _distance in neighbors
        }
        reference_pairs = {
            tuple(sorted((site_a, site_b)))
            for site_a, neighbors in reference.items()
            for site_b, _distance in neighbors
        }
        assert adjacency_pairs == reference_pairs

    def test_bridge_clearance_matches_naive_reference_for_valid_candidate(self):
        system = self.prepared_result.system
        adjacency = slit_mod._build_slit_site_adjacency(
            system,
            sorted(system._site_in),
            self.config.siloxane_distance_range_nm,
        )

        pair = None
        bridge_position = None
        for site_a, neighbors in adjacency.items():
            for site_b, _distance in neighbors:
                candidate_pair = (site_a, site_b)
                candidate_position = slit_mod._siloxane_bridge_position(system, candidate_pair)
                if candidate_position is not None:
                    pair = candidate_pair
                    bridge_position = candidate_position
                    break
            if pair is not None:
                break

        assert pair is not None
        assert bridge_position is not None
        assert slit_mod._bridge_steric_score(system, pair, bridge_position) == pytest.approx(
            naive_bridge_clearance(system, pair, bridge_position, local_only=True),
            abs=1e-12,
        )
        assert slit_mod._bridge_global_clearance(system, pair, bridge_position) == pytest.approx(
            naive_bridge_clearance(system, pair, bridge_position, local_only=False),
            abs=1e-12,
        )

    def test_bridge_clearance_matches_naive_reference_for_invalid_candidate(self):
        system = self.prepared_result.system
        adjacency = slit_mod._build_slit_site_adjacency(
            system,
            sorted(system._site_in),
            self.config.siloxane_distance_range_nm,
        )

        pair = None
        for site_a, neighbors in adjacency.items():
            if neighbors:
                pair = (site_a, neighbors[0][0])
                break

        assert pair is not None
        block = system._pore.get_block()
        sites = system._pore.get_sites()
        excluded_ids = {
            pair[0],
            pair[1],
            sites[pair[0]].oxygen_ids[0],
            sites[pair[1]].oxygen_ids[0],
        }
        local_ids = naive_bridge_local_ids(system, pair)
        reference_atom_id = next(
            atom_id
            for atom_id in sorted(local_ids)
            if atom_id not in excluded_ids
        )
        invalid_position = block.pos(reference_atom_id)

        local_score = slit_mod._bridge_steric_score(system, pair, invalid_position)
        global_score = slit_mod._bridge_global_clearance(system, pair, invalid_position)
        assert local_score < 0
        assert global_score < 0
        assert local_score == pytest.approx(
            naive_bridge_clearance(system, pair, invalid_position, local_only=True),
            abs=1e-12,
        )
        assert global_score == pytest.approx(
            naive_bridge_clearance(system, pair, invalid_position, local_only=False),
            abs=1e-12,
        )

    def test_repeat_y_one_reaches_requested_surface_target(self):
        result = pms.prepare_amorphous_slit_surface(
            config=pms.AmorphousSlitConfig(
                name="thin_bare_amorphous_slit",
                repeat_y=1,
                surface_target=self.surface_target,
            )
        )

        assert result.report.site_ex == 0
        assert result.report.wall_thickness_nm == pytest.approx(1.3025, abs=10 ** (-(4)))
        assert not (result.report.used_surface_tolerance)
        assert result.report.prepared_surface == result.report.target_surface
        assert result.report.final_surface == result.report.target_surface
        assert result.report.prepared_surface.total_surface_si == 957
        assert result.report.prepared_surface.q2_sites == 66
        assert result.report.prepared_surface.q3_sites == 652
        assert result.report.prepared_surface.q4_sites == 239

    def test_auto_alpha_q_only_target_uses_unified_conversion(self):
        base_config = pms.AmorphousSlitConfig(
            name="auto_alpha_reference",
            repeat_y=1,
            surface_target=self.surface_target,
        )
        base_result = pms.prepare_amorphous_slit_surface(config=base_config)
        alpha_auto = base_result.report.alpha_auto
        reference_surface = pms.SiliconStateFractions(0.069, 0.681, 0.25)
        experimental_target = experimental_target_from_surface(
            reference_surface,
            alpha_auto,
            alpha_override=None,
        )

        result = pms.prepare_amorphous_slit_surface(
            config=pms.AmorphousSlitConfig(
                name="auto_alpha_case",
                repeat_y=1,
                surface_target=experimental_target,
            )
        )

        assert result.report.alpha_effective == pytest.approx(alpha_auto, abs=10 ** (-(8)))
        assert result.report.derived_surface_target.q2_fraction == pytest.approx(reference_surface.q2_fraction, abs=10 ** (-(12)))
        assert result.report.derived_surface_target.q3_fraction == pytest.approx(reference_surface.q3_fraction, abs=10 ** (-(12)))
        assert result.report.derived_surface_target.q4_fraction == pytest.approx(reference_surface.q4_fraction, abs=10 ** (-(12)))
        assert result.report.final_surface.q2_sites == 66
        assert result.report.final_surface.q3_sites == 652
        assert result.report.final_surface.q4_sites == 239

    def test_surface_conversion_example_matches_expected_q2_enrichment(self):
        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=0.5,
            q3_fraction=0.0,
            alpha_override=0.5,
        )
        surface_target = slit_mod._surface_target_from_experimental(target, 0.5)

        assert surface_target == pms.SiliconStateFractions(1.0, 0.0, 0.0)

    def test_alpha_override_precedence(self):
        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=0.02,
            q3_fraction=0.03,
            alpha_override=0.2,
        )
        alpha_auto, alpha_effective = slit_mod._effective_alpha(100, 1000, target)

        assert alpha_auto == pytest.approx(0.1, abs=1e-7)
        assert alpha_effective == pytest.approx(0.2, abs=1e-7)

    def test_q4_fraction_is_derived_when_omitted(self):
        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=0.02,
            q3_fraction=0.03,
            t2_fraction=0.04,
            t3_fraction=0.01,
        )

        assert target.q4_fraction == pytest.approx(0.9, abs=1e-12)
        assert asdict(target)["q4_fraction"] == pytest.approx(0.9, abs=1e-12)

    def test_explicit_q4_fraction_must_match_remainder(self):
        with pytest.raises(ValueError, match="q4 fraction"):
            pms.ExperimentalSiliconStateTarget(
                q2_fraction=0.02,
                q3_fraction=0.03,
                q4_fraction=0.89,
                t2_fraction=0.04,
                t3_fraction=0.01,
            )

    def test_invalid_alpha_target_combinations_raise(self):
        with pytest.raises(ValueError):
            slit_mod._surface_target_from_experimental(
                pms.ExperimentalSiliconStateTarget(
                    q2_fraction=0.5,
                    q3_fraction=0.0,
                    alpha_override=0.4,
                ),
                0.4,
            )

    def test_bridge_algebra_matches_q_state_changes(self):
        cases = [
            ((2, 2), (-2, 2, 0), (1, 0)),
            ((2, 1), (-1, 0, 1), (1, 1)),
            ((1, 1), (0, -2, 2), (1, 2)),
        ]

        for pair_counts, expected_delta, expected_objectified in cases:
            config = pms.AmorphousSlitConfig(
                name="bridge_algebra_case",
                repeat_y=1,
                surface_target=self.surface_target,
            )
            system = self._build_uncondensed_slit(config)
            total_surface_si = len(system._site_in)
            sites = system._pore.get_sites()
            before = slit_mod._surface_composition(total_surface_si, sites)
            adjacency = slit_mod._build_slit_site_adjacency(
                system,
                sorted(system._site_in),
                config.siloxane_distance_range_nm,
            )
            pair, bridge_position = slit_mod._find_placeable_pair(
                system,
                sites,
                adjacency,
                *pair_counts,
            )

            assert pair is not None, pair_counts
            assert bridge_position is not None, pair_counts
            om_before = len(system._pore.get_mol_dict().get("OM", []))
            si_before = len(system._pore.get_mol_dict().get("SI", []))
            slit_mod._bridge_pair(system, pair, bridge_position=bridge_position)
            slit_mod._consume_pair(adjacency, pair)

            after = slit_mod._surface_composition(
                total_surface_si,
                system._pore.get_sites(),
            )
            assert after.q2_sites - before.q2_sites == expected_delta[0], pair_counts
            assert after.q3_sites - before.q3_sites == expected_delta[1], pair_counts
            assert after.q4_sites - before.q4_sites == expected_delta[2], pair_counts
            assert len(system._pore.get_mol_dict().get("OM", [])) - om_before == expected_objectified[0], pair_counts
            assert len(system._pore.get_mol_dict().get("SI", [])) - si_before == expected_objectified[1], pair_counts
            assert "SLX" not in system._pore.get_mol_dict()

    def test_tolerance_fallback_selects_nearest_realizable_target(self):
        config = pms.AmorphousSlitConfig(
            name="tolerance_case",
            repeat_y=1,
            surface_target=self.surface_target,
        )
        system = self._build_uncondensed_slit(config)
        total_surface_si = len(system._site_in)
        initial_surface = slit_mod._surface_composition(
            total_surface_si,
            system._pore.get_sites(),
        )
        requested_target = pms.SiliconStateFractions(66 / 957, 653 / 957, 238 / 957)
        exact_target = pms.SiliconStateComposition(
            total_surface_si=total_surface_si,
            q2_sites=66,
            q3_sites=653,
            q4_sites=238,
        )

        attempt = slit_mod._realize_surface_target(
            system,
            total_surface_si,
            initial_surface,
            requested_target,
            exact_target,
            config.surface_fraction_tolerance,
            config.siloxane_distance_range_nm,
            ligand=None,
        )
        errors = slit_mod._surface_fraction_errors(
            attempt.final_surface,
            requested_target,
        )

        assert attempt.used_surface_tolerance
        assert attempt.target_surface == pms.SiliconStateComposition(
                total_surface_si=total_surface_si,
                q2_sites=65,
                q3_sites=654,
                q4_sites=238,
            )
        assert attempt.prepared_surface == attempt.target_surface
        assert attempt.final_surface == attempt.target_surface
        assert all(error <= config.surface_fraction_tolerance for error in errors)

    def test_prepared_slit_remains_attachable(self):
        result = pms.prepare_amorphous_slit_surface(
            config=pms.AmorphousSlitConfig(
                name="attachable_bare_amorphous_slit",
                surface_target=self.surface_target,
            )
        )

        result.system.attach(
            pms.gen.tms(),
            0,
            [0, 1],
            1,
            site_type="in",
            is_proxi=False,
            is_g=False,
        )

        mol_dict = result.system._pore.get_mol_dict()
        assert "TMS" in mol_dict
        assert len(mol_dict["TMS"]) == 1

    def test_bare_builder_rejects_non_zero_t_states(self):
        with pytest.raises(ValueError):
            pms.prepare_amorphous_slit_surface(
                config=pms.AmorphousSlitConfig(
                    surface_target=pms.ExperimentalSiliconStateTarget(
                        q2_fraction=0.05,
                        q3_fraction=0.05,
                        q4_fraction=0.85,
                        t2_fraction=0.05,
                    )
                )
            )

    def test_bare_slit_files_are_written(self):
        expected_files = [
            "test_bare_amorphous_slit.gro",
            "test_bare_amorphous_slit.itp",
            "test_bare_amorphous_slit.top",
            "test_bare_amorphous_slit.yml",
            "test_bare_amorphous_slit_report.json",
        ]
        for file_name in expected_files:
            assert os.path.isfile(os.path.join(self.output_dir, file_name))
        assert not os.path.exists(os.path.join(self.output_dir, "grid.itp"))

        assert not (os.path.exists(os.path.join(self.output_dir, "test_bare_amorphous_slit.obj")))
        assert not (os.path.exists(
                os.path.join(self.output_dir, "test_bare_amorphous_slit_system.obj")
            ))
        with open(os.path.join(self.output_dir, "test_bare_amorphous_slit.top"), "r") as file_in:
            top_text = file_in.read()
        with open(os.path.join(self.output_dir, "test_bare_amorphous_slit.itp"), "r") as file_in:
            itp_text = file_in.read()

        assert '#include "test_bare_amorphous_slit.itp"' in top_text
        assert "TEST_BARE_AMORPHOUS_SLIT 1" in top_text
        assert "[ atomtypes ]" in itp_text
        assert "[ moleculetype ]" in itp_text

        atom_rows = itp_atom_rows(
            os.path.join(self.output_dir, "test_bare_amorphous_slit.itp")
        )
        total_charge = sum(row[3] for row in atom_rows)
        residue_counts = {
            residue_name: sum(1 for _atom_type, residue, _atom_name, _charge in atom_rows if residue == residue_name)
            for residue_name in {"OM", "SI", "SL", "SLG"}
        }

        diagnostics = self.stored_result.bare_charge_diagnostics
        assert diagnostics is not None
        assert total_charge == pytest.approx(0.0, abs=1e-8)
        assert residue_counts["OM"] == diagnostics.framework_oxygen.atom_count
        assert residue_counts["SI"] == diagnostics.framework_silicon.atom_count
        assert residue_counts["SL"] == (
            diagnostics.silanol_silicon.atom_count
            + diagnostics.silanol_oxygen.atom_count
            + diagnostics.silanol_hydrogen.atom_count
        )
        assert residue_counts["SLG"] == (
            diagnostics.geminal_silicon.atom_count
            + diagnostics.geminal_oxygen.atom_count
            + diagnostics.geminal_hydrogen.atom_count
        )

        report_path = os.path.join(
            self.output_dir,
            "test_bare_amorphous_slit_report.json",
        )
        with open(report_path, "r") as file_in:
            data = json.load(file_in)

        assert data["site_ex"] == 0
        assert data["siloxane_distance_range_nm"] == [0.4, 0.65]
        assert data["surface_fraction_tolerance"] == 0.005
        assert not (data["used_surface_tolerance"])
        assert data["alpha_auto"] == pytest.approx(self.stored_report.alpha_auto, abs=10 ** (-(8)))
        assert data["alpha_effective"] == 1.0
        assert data["final_surface"]["q2_sites"] == self.stored_report.final_surface.q2_sites
        assert data["final_surface"]["q3_sites"] == self.stored_report.final_surface.q3_sites
        assert data["final_surface"]["q4_sites"] == self.stored_report.final_surface.q4_sites
        assert data["preparation_diagnostics"]["inserted_bridge_oxygen"] == self.stored_report.preparation_diagnostics.inserted_bridge_oxygen
        assert not (os.path.exists(
                os.path.join(self.output_dir, "test_bare_amorphous_slit_next_steps.md")
            ))

    def test_finalized_bare_slit_connectivity_is_valid(self):
        report = pms.Store(self.stored_result.system._pore).validate_connectivity(
            use_atom_names=True
        )

        assert report.is_valid

    def test_validation_flags_broken_silanol_silicon_environment(self):
        store = pms.Store(
            self.stored_result.system._pore,
            sort_list=self.stored_result.system._sort_list,
        )
        cache = store._collect_structure_records(use_atom_names=True)
        graph = store.assembled_graph(use_atom_names=True)
        neighbors = store._connectivity_validation_neighbors(graph)
        record_by_serial = {record.serial: record for record in cache.atom_records}

        silanol_si_serial = next(
            record.serial
            for record in cache.atom_records
            if record.residue_name == "SL" and record.atom_type == "Si"
        )
        broken_neighbor = next(
            neighbor_id
            for neighbor_id in sorted(neighbors[silanol_si_serial])
            if record_by_serial[neighbor_id].atom_type == "O"
            and record_by_serial[neighbor_id].residue_name == "OM"
        )
        broken_graph = graph_without_bond(graph, silanol_si_serial, broken_neighbor)
        findings = store._connectivity_validation_findings(
            cache.atom_records,
            broken_graph,
        )

        assert any(
            finding.code == "silanol_silicon_environment"
            for finding in findings
        )

    def test_validation_flags_broken_geminal_silicon_environment(self):
        store = pms.Store(
            self.stored_result.system._pore,
            sort_list=self.stored_result.system._sort_list,
        )
        cache = store._collect_structure_records(use_atom_names=True)
        graph = store.assembled_graph(use_atom_names=True)
        neighbors = store._connectivity_validation_neighbors(graph)
        record_by_serial = {record.serial: record for record in cache.atom_records}

        geminal_si_serial = next(
            record.serial
            for record in cache.atom_records
            if record.residue_name == "SLG" and record.atom_type == "Si"
        )
        broken_neighbor = next(
            neighbor_id
            for neighbor_id in sorted(neighbors[geminal_si_serial])
            if record_by_serial[neighbor_id].atom_type == "O"
            and record_by_serial[neighbor_id].residue_name == "OM"
        )
        broken_graph = graph_without_bond(graph, geminal_si_serial, broken_neighbor)
        findings = store._connectivity_validation_findings(
            cache.atom_records,
            broken_graph,
        )

        assert any(
            finding.code == "geminal_silicon_environment"
            for finding in findings
        )

    def test_validation_flags_broken_silanol_hydroxyl_environment(self):
        store = pms.Store(
            self.stored_result.system._pore,
            sort_list=self.stored_result.system._sort_list,
        )
        cache = store._collect_structure_records(use_atom_names=True)
        graph = store.assembled_graph(use_atom_names=True)

        silanol_oxygen_serial = next(
            record.serial
            for record in cache.atom_records
            if record.residue_name == "SL" and record.atom_type == "O"
        )
        silanol_hydrogen_serial = next(
            record.serial
            for record in cache.atom_records
            if record.residue_name == "SL" and record.atom_type == "H"
        )
        broken_graph = graph_without_bond(
            graph,
            silanol_oxygen_serial,
            silanol_hydrogen_serial,
        )
        findings = store._connectivity_validation_findings(
            cache.atom_records,
            broken_graph,
        )

        finding_codes = {finding.code for finding in findings}
        assert "silanol_oxygen_environment" in finding_codes
        assert "silanol_hydrogen_environment" in finding_codes

    def test_bare_slit_object_files_are_opt_in(self, tmp_path):
        output_dir = tmp_path / "bare_amorphous_slit_preparation_with_objects"
        pms.write_bare_amorphous_slit(
            str(output_dir),
            config=pms.AmorphousSlitConfig(
                name="test_bare_amorphous_slit_with_objects",
                surface_target=self.surface_target,
            ),
            write_object_files=True,
        )

        assert (output_dir / "test_bare_amorphous_slit_with_objects.obj").is_file()
        assert (output_dir / "test_bare_amorphous_slit_with_objects_system.obj").is_file()

    def test_bare_slit_pdb_writes_conect_by_default(self, tmp_path):
        output_dir = tmp_path / "bare_amorphous_slit_preparation_with_pdb"
        pms.write_bare_amorphous_slit(
            str(output_dir),
            config=pms.AmorphousSlitConfig(
                name="test_bare_amorphous_slit_with_pdb",
                surface_target=self.surface_target,
            ),
            write_pdb=True,
        )

        pdb_path = output_dir / "test_bare_amorphous_slit_with_pdb.pdb"
        assert pdb_path.is_file()

        with open(pdb_path, "r") as file_in:
            pdb_lines = file_in.readlines()

        assert any(line.startswith("HETATM") for line in pdb_lines)
        assert any(line.startswith("CONECT") for line in pdb_lines)

    def test_bare_slit_cif_writes_bonds_by_default(self, tmp_path):
        output_dir = tmp_path / "bare_amorphous_slit_preparation_with_cif"
        pms.write_bare_amorphous_slit(
            str(output_dir),
            config=pms.AmorphousSlitConfig(
                name="test_bare_amorphous_slit_with_cif",
                surface_target=self.surface_target,
            ),
            write_cif=True,
        )

        cif_path = output_dir / "test_bare_amorphous_slit_with_cif.cif"
        assert cif_path.is_file()

        with open(cif_path, "r") as file_in:
            cif_text = file_in.read()

        assert "_atom_site.Cartn_x" in cif_text
        assert "_struct_conn.id" in cif_text

    def test_explicit_silica_topology_override_changes_bare_itp_terms(self, tmp_path):
        output_dir = tmp_path / "bare_amorphous_slit_custom_silica"
        silica_topology = pms.default_silica_topology()
        silica_topology.bond_terms.framework_si_o.force_constant = 123456.0

        result = pms.write_bare_amorphous_slit(
            str(output_dir),
            config=pms.AmorphousSlitConfig(
                name="bare_custom_silica",
                surface_target=self.surface_target,
                silica_topology=silica_topology,
            ),
            write_pdb=False,
            write_cif=False,
        )

        with open(output_dir / "bare_custom_silica.itp", "r") as file_in:
            itp_text = file_in.read()

        assert result.silica_topology is not silica_topology
        assert result.silica_topology.bond_terms.framework_si_o.force_constant == pytest.approx(123456.0)
        assert "0.16500 123456.000000" in itp_text

    def test_top_level_exports_and_version(self):
        assert pms.__version__ == EXPECTED_VERSION
        assert callable(pms.prepare_amorphous_slit_surface)
        assert callable(pms.write_bare_amorphous_slit)
        assert callable(pms.prepare_functionalized_amorphous_slit_surface)
        assert callable(pms.write_functionalized_amorphous_slit)
        assert callable(pms.default_silica_topology)
        assert isinstance(self.config, pms.AmorphousSlitConfig)
        assert isinstance(self.surface_target, pms.ExperimentalSiliconStateTarget)
        assert isinstance(self.prepared_result, pms.SlitPreparationResult)
        assert isinstance(self.prepared_report, pms.SlitPreparationReport)
        assert isinstance(self.prepared_report.prepared_surface, pms.SiliconStateComposition)
        assert isinstance(self.prepared_report.preparation_diagnostics, pms.SurfacePreparationDiagnostics)
        assert hasattr(pms, "SiliconStateFractions")
        assert hasattr(pms, "SurfacePreparationDiagnostics")
        assert hasattr(pms, "BareSilicaChargeContribution")
        assert hasattr(pms, "BareSilicaChargeDiagnostics")
        assert hasattr(pms, "FunctionalizedSlitChargeDiagnostics")
        assert hasattr(pms, "SilaneAttachmentConfig")
        assert hasattr(pms, "SlitTimingSummary")
        assert hasattr(pms, "FunctionalizedSlitProgressConfig")
        assert hasattr(pms, "FunctionalizedSlitStericConfig")
        assert hasattr(pms, "FunctionalizedAmorphousSlitConfig")
        assert hasattr(pms, "FunctionalizedSlitResult")
        assert hasattr(pms, "GeminalMountDihedralSpec")
        assert hasattr(pms, "SilaneGeminalCrossTerms")
        assert hasattr(pms, "GraphBond")
        assert hasattr(pms, "GraphAngle")
        assert hasattr(pms, "AttachmentRecord")
        assert hasattr(pms, "AssembledStructureGraph")
        assert hasattr(pms, "SilicaAtomTypeModel")
        assert hasattr(pms, "SilicaAtomTypeSet")
        assert hasattr(pms, "SilicaAtomAssignment")
        assert hasattr(pms, "SilicaAtomAssignmentSet")
        assert hasattr(pms, "SilicaBondTerm")
        assert hasattr(pms, "SilicaBondTermSet")
        assert hasattr(pms, "SilicaAngleTerm")
        assert hasattr(pms, "SilicaAngleTermSet")
        assert hasattr(pms, "SilicaTopologyModel")
        assert hasattr(pms, "SilaneTopologyConfig")
        assert hasattr(pms, "SlitJunctionParameters")


class TestFunctionalizedAmorphousSlit:
    def test_resolve_silane_topology_config_requires_explicit_topology_input(self):
        assert slit_mod.resolve_silane_topology_config(None) is None
        assert slit_mod.resolve_silane_topology_config(
            pms.SilaneAttachmentConfig(
                molecule=pms.gen.tms(),
                mount=0,
                axis=(0, 1),
            )
        ) is None

    def test_silane_attachment_config_defaults_to_ten_degree_rotation_scan(self):
        ligand = pms.SilaneAttachmentConfig(
            molecule=pms.gen.tms(),
            mount=0,
            axis=(0, 1),
        )

        assert ligand.rotate_about_axis
        assert ligand.rotate_step_deg == 10.0

    def test_functionalized_steric_config_defaults_to_relaxed_slit_scale(self):
        sterics = pms.FunctionalizedSlitStericConfig()

        assert sterics.enabled
        assert sterics.clearance_scale == pytest.approx(0.60)

    def test_functionalized_progress_config_defaults_to_auto_quiet_leave_false(self):
        progress = pms.FunctionalizedSlitProgressConfig()

        assert progress.enabled is None
        assert not (progress.leave)

    def test_progress_auto_mode_is_quiet_under_pytest(self):
        bar = slit_mod._create_progress_bar(
            total=3,
            desc="demo",
            progress_config=pms.FunctionalizedSlitProgressConfig(),
            unit="step",
        )

        assert isinstance(bar, slit_mod._NullProgressBar)

    def test_progress_can_be_forced_off_even_in_interactive_mode(self, monkeypatch):
        monkeypatch.setattr(slit_mod, "_is_interactive_progress_environment", lambda: True)

        bar = slit_mod._create_progress_bar(
            total=3,
            desc="demo",
            progress_config=pms.FunctionalizedSlitProgressConfig(enabled=False),
            unit="step",
        )

        assert isinstance(bar, slit_mod._NullProgressBar)

    def test_progress_can_be_forced_on_in_non_interactive_mode(self, monkeypatch):
        created = []

        def fake_tqdm(**kwargs):
            bar = RecordingProgressBar(
                total=kwargs["total"],
                desc=kwargs["desc"],
                unit=kwargs["unit"],
                leave=kwargs["leave"],
            )
            created.append(bar)
            return bar

        monkeypatch.setattr(slit_mod, "_is_interactive_progress_environment", lambda: False)
        monkeypatch.setattr(slit_mod, "_tqdm_auto", fake_tqdm)

        bar = slit_mod._create_progress_bar(
            total=4,
            desc="forced",
            progress_config=pms.FunctionalizedSlitProgressConfig(enabled=True, leave=True),
            unit="stage",
        )

        assert created
        assert bar is created[0]
        assert created[0].leave
        assert created[0].total == 4

    def test_generic_attach_default_steric_scale_is_unchanged(self):
        steric_parameter = inspect.signature(pms.Pore.attach).parameters["steric_clearance_scale"]

        assert steric_parameter.default == pytest.approx(0.85)

    def test_functionalized_path_uses_configured_steric_clearance_scale(self, monkeypatch):
        recorded_scales = []
        original_attach = pms.Pore.attach

        def recording_attach(self, *args, **kwargs):
            recorded_scales.append(kwargs.get("steric_clearance_scale"))
            return original_attach(self, *args, **kwargs)

        monkeypatch.setattr(pms.Pore, "attach", recording_attach)

        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=65 / 957,
            q3_fraction=651 / 957,
            q4_fraction=239 / 957,
            t2_fraction=1 / 957,
            t3_fraction=1 / 957,
            alpha_override=1.0,
        )
        config = pms.FunctionalizedAmorphousSlitConfig(
            slit_config=pms.AmorphousSlitConfig(
                name="functionalized_custom_sterics",
                repeat_y=1,
                surface_target=target,
            ),
            ligand=pms.SilaneAttachmentConfig(
                molecule=pms.gen.tms(),
                mount=0,
                axis=(0, 1),
                rotate_about_axis=False,
            ),
            steric_settings=pms.FunctionalizedSlitStericConfig(clearance_scale=0.55),
        )

        pms.prepare_functionalized_amorphous_slit_surface(config)

        assert recorded_scales
        assert 0.55 in recorded_scales

    def test_functionalized_path_batches_attachment_slots(self, monkeypatch):
        recorded_calls = []
        original_attach = pms.Pore.attach

        def recording_attach(self, *args, **kwargs):
            recorded_calls.append(
                {
                    "amount": args[4],
                    "sites_len": len(args[3]),
                    "is_g": kwargs.get("is_g"),
                    "progress_callback": kwargs.get("_progress_callback") is not None,
                }
            )
            return original_attach(self, *args, **kwargs)

        monkeypatch.setattr(pms.Pore, "attach", recording_attach)

        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=63 / 957,
            q3_fraction=648 / 957,
            q4_fraction=239 / 957,
            t2_fraction=3 / 957,
            t3_fraction=4 / 957,
            alpha_override=1.0,
        )
        config = pms.FunctionalizedAmorphousSlitConfig(
            slit_config=pms.AmorphousSlitConfig(
                name="functionalized_batched_attachment",
                repeat_y=1,
                surface_target=target,
            ),
            ligand=pms.SilaneAttachmentConfig(
                molecule=pms.gen.tms(),
                mount=0,
                axis=(0, 1),
                rotate_about_axis=False,
            ),
        )

        pms.prepare_functionalized_amorphous_slit_surface(config)

        batched_calls = [
            call for call in recorded_calls
            if call["progress_callback"]
        ]
        assert any(
            call["amount"] == 3 and call["sites_len"] >= 3 and call["is_g"]
            for call in batched_calls
        )
        assert any(
            call["amount"] == 4 and call["sites_len"] >= 4 and not call["is_g"]
            for call in batched_calls
        )

    def test_attachment_progress_description_includes_candidate_context(self):
        context = slit_mod._AttachmentPhaseProgressContext(
            phase_name="T2 attachment",
            requested_count=5,
            candidate_index=2,
            total_candidates=7,
        )

        assert (
            slit_mod._attachment_progress_description(context, attached_count=3)
            == "T2 attachment 3/5 attached [candidate 2/7]"
        )

    def test_prepare_progress_creates_outer_and_inner_bars(self, monkeypatch):
        created = []

        def fake_create_progress_bar(total, desc, progress_config, unit="it"):
            del progress_config
            bar = RecordingProgressBar(total=total, desc=desc, unit=unit, leave=False)
            created.append(bar)
            return bar

        monkeypatch.setattr(slit_mod, "_create_progress_bar", fake_create_progress_bar)

        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=65 / 957,
            q3_fraction=651 / 957,
            q4_fraction=239 / 957,
            t2_fraction=1 / 957,
            t3_fraction=1 / 957,
            alpha_override=1.0,
        )
        config = pms.FunctionalizedAmorphousSlitConfig(
            slit_config=pms.AmorphousSlitConfig(
                name="functionalized_progress_prepare",
                repeat_y=1,
                surface_target=target,
            ),
            ligand=pms.SilaneAttachmentConfig(
                molecule=pms.gen.tms(),
                mount=0,
                axis=(0, 1),
                rotate_about_axis=False,
            ),
            progress_settings=pms.FunctionalizedSlitProgressConfig(enabled=True),
        )

        result = pms.prepare_functionalized_amorphous_slit_surface(config)

        assert result.report.final_surface == pms.SiliconStateComposition(957, 65, 651, 239, 1, 1)
        stage_bars = [bar for bar in created if bar.unit == "stage"]
        site_bars = [bar for bar in created if bar.unit == "site"]
        assert len(stage_bars) == 1
        assert stage_bars[0].total == 4
        assert stage_bars[0].n == 4
        assert any(desc == "Base slit build" for desc in stage_bars[0].descriptions)
        assert any(desc == "Q-state preparation" for desc in stage_bars[0].descriptions)
        assert any(desc == "T2 attachment" for desc in stage_bars[0].descriptions)
        assert any(desc == "T3 attachment" for desc in stage_bars[0].descriptions)
        assert sorted(bar.total for bar in site_bars) == [1, 1]

    def test_write_progress_includes_finalize_and_store_stages(self, monkeypatch, tmp_path):
        created = []

        def fake_create_progress_bar(total, desc, progress_config, unit="it"):
            del progress_config
            bar = RecordingProgressBar(total=total, desc=desc, unit=unit, leave=False)
            created.append(bar)
            return bar

        monkeypatch.setattr(slit_mod, "_create_progress_bar", fake_create_progress_bar)

        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=65 / 957,
            q3_fraction=651 / 957,
            q4_fraction=239 / 957,
            t2_fraction=1 / 957,
            t3_fraction=1 / 957,
            alpha_override=1.0,
        )
        config = pms.FunctionalizedAmorphousSlitConfig(
            slit_config=pms.AmorphousSlitConfig(
                name="functionalized_progress_write",
                repeat_y=1,
                surface_target=target,
            ),
            ligand=pms.SilaneAttachmentConfig(
                molecule=pms.gen.tms(),
                mount=0,
                axis=(0, 1),
                rotate_about_axis=False,
            ),
            progress_settings=pms.FunctionalizedSlitProgressConfig(enabled=True),
        )

        result = pms.write_functionalized_amorphous_slit(
            str(tmp_path / "functionalized_progress_write"),
            config,
            write_pdb=False,
            write_cif=False,
        )

        assert result.charge_diagnostics is None
        assert result.report.timing_summary.finalize_s > 0
        assert result.report.timing_summary.store_export_s > 0
        assert not (tmp_path / "functionalized_progress_write" / "functionalized_progress_write.top").exists()
        assert not (tmp_path / "functionalized_progress_write" / "functionalized_progress_write.itp").exists()
        stage_bars = [bar for bar in created if bar.unit == "stage"]
        assert len(stage_bars) == 1
        assert stage_bars[0].total == 6
        assert stage_bars[0].n == 6
        assert any(desc == "Finalize" for desc in stage_bars[0].descriptions)
        assert any(desc == "Store/export" for desc in stage_bars[0].descriptions)

    def test_exact_functionalized_target_is_realized(self):
        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=63 / 957,
            q3_fraction=648 / 957,
            q4_fraction=239 / 957,
            t2_fraction=3 / 957,
            t3_fraction=4 / 957,
            alpha_override=1.0,
        )
        config = pms.FunctionalizedAmorphousSlitConfig(
            slit_config=pms.AmorphousSlitConfig(
                name="functionalized_exact_slit",
                repeat_y=1,
                surface_target=target,
            ),
            ligand=pms.SilaneAttachmentConfig(
                molecule=pms.gen.tms(),
                mount=0,
                axis=(0, 1),
            ),
        )

        result = pms.prepare_functionalized_amorphous_slit_surface(config)

        assert isinstance(result.silica_topology, pms.SilicaTopologyModel)
        assert not (result.report.used_surface_tolerance)
        assert result.report.prepared_surface == pms.SiliconStateComposition(957, 66, 652, 239)
        assert result.report.final_surface == pms.SiliconStateComposition(957, 63, 648, 239, 3, 4)
        assert result.report.final_surface == result.report.target_surface
        assert isinstance(result.report.timing_summary, pms.SlitTimingSummary)
        assert result.report.timing_summary.base_slit_build_s > 0
        assert result.report.timing_summary.q_state_preparation_s > 0
        assert result.report.timing_summary.t2_attachment_s > 0
        assert result.report.timing_summary.t3_attachment_s > 0
        assert len(result.system._pore.get_site_dict()["in"]["TMSG"]) == 3
        assert len(result.system._pore.get_site_dict()["in"]["TMS"]) == 4
        assert "SLX" not in result.system._pore.get_mol_dict()

    def test_functionalized_tolerance_fallback_selects_nearest_realizable_target(self):
        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=63 / 957,
            q3_fraction=649 / 957,
            q4_fraction=238 / 957,
            t2_fraction=3 / 957,
            t3_fraction=4 / 957,
            alpha_override=1.0,
        )
        config = pms.FunctionalizedAmorphousSlitConfig(
            slit_config=pms.AmorphousSlitConfig(
                name="functionalized_tolerance_slit",
                repeat_y=1,
                surface_target=target,
            ),
            ligand=pms.SilaneAttachmentConfig(
                molecule=pms.gen.tms(),
                mount=0,
                axis=(0, 1),
            ),
        )

        result = pms.prepare_functionalized_amorphous_slit_surface(config)

        assert result.report.used_surface_tolerance
        assert result.report.final_surface == pms.SiliconStateComposition(957, 63, 648, 239, 3, 4)
        errors = slit_mod._surface_fraction_errors(
            result.report.final_surface,
            result.report.derived_surface_target,
        )
        assert all(error <= result.report.surface_fraction_tolerance for error in errors)

    def test_functionalized_assembled_graph_contains_graft_junctions_and_angles(self):
        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=63 / 957,
            q3_fraction=648 / 957,
            q4_fraction=239 / 957,
            t2_fraction=3 / 957,
            t3_fraction=4 / 957,
            alpha_override=1.0,
        )
        config = pms.FunctionalizedAmorphousSlitConfig(
            slit_config=pms.AmorphousSlitConfig(
                name="functionalized_graph_slit",
                repeat_y=1,
                surface_target=target,
            ),
            ligand=pms.SilaneAttachmentConfig(
                molecule=pms.gen.tms(),
                mount=0,
                axis=(0, 1),
            ),
        )

        result = pms.prepare_functionalized_amorphous_slit_surface(config)
        store = pms.Store(result.system._pore)
        graph = store.assembled_graph(use_atom_names=True)
        report = store.validate_connectivity(use_atom_names=True)
        cache = store._collect_structure_records(use_atom_names=True)
        atom_records = cache.atom_records
        molecule_serials = cache.molecule_serials
        serials_by_molecule = {
            id(molecule): serials
            for molecule, serials in zip(store._mols, molecule_serials)
        }

        assert isinstance(graph, pms.AssembledStructureGraph)
        assert isinstance(report, pms.ConnectivityValidationReport)
        assert not (any(
                finding.code in {"framework_oxygen_environment", "framework_silicon_environment"}
                for finding in report.findings
            ))
        assert any(bond.provenance == "graft_junction" for bond in graph.bonds)
        assert any(bond.provenance == "ligand_explicit" for bond in graph.bonds)
        assert not (any(bond.provenance == "ligand_inferred" for bond in graph.bonds))

        attachment_record = result.system._pore.get_attachment_records()[0]
        mount_serial = serials_by_molecule[id(attachment_record.molecule)][
            attachment_record.mount_atom_local_id
        ]
        assert any(
                bond.provenance == "graft_junction"
                and mount_serial in (bond.atom_a, bond.atom_b)
                for bond in graph.bonds
            )
        assert any(angle.atom_b == mount_serial for angle in graph.angles)

        record_by_serial = {record.serial: record for record in atom_records}
        graft_neighbors = {
            bond.atom_b if bond.atom_a == mount_serial else bond.atom_a
            for bond in graph.bonds
            if bond.provenance == "graft_junction"
            and mount_serial in (bond.atom_a, bond.atom_b)
        }
        assert all(record_by_serial[serial].atom_type == "O" for serial in graft_neighbors)

    def test_finalized_functionalized_connectivity_is_valid(self):
        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=63 / 957,
            q3_fraction=648 / 957,
            q4_fraction=239 / 957,
            t2_fraction=3 / 957,
            t3_fraction=4 / 957,
            alpha_override=1.0,
        )
        config = pms.FunctionalizedAmorphousSlitConfig(
            slit_config=pms.AmorphousSlitConfig(
                name="functionalized_final_validation_slit",
                repeat_y=1,
                surface_target=target,
            ),
            ligand=pms.SilaneAttachmentConfig(
                molecule=pms.gen.tms(),
                mount=0,
                axis=(0, 1),
            ),
        )

        result = pms.prepare_functionalized_amorphous_slit_surface(config)
        result.system.finalize()
        report = pms.Store(result.system._pore).validate_connectivity(use_atom_names=True)

        assert report.is_valid

    def test_functionalized_bonded_exports_include_graft_connectivity(self, tmp_path):
        output_dir = tmp_path / "functionalized_amorphous_slit_bonded_export"
        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=63 / 957,
            q3_fraction=648 / 957,
            q4_fraction=239 / 957,
            t2_fraction=3 / 957,
            t3_fraction=4 / 957,
            alpha_override=1.0,
        )
        config = pms.FunctionalizedAmorphousSlitConfig(
            slit_config=pms.AmorphousSlitConfig(
                name="functionalized_export_slit",
                repeat_y=1,
                surface_target=target,
            ),
            ligand=pms.SilaneAttachmentConfig(
                molecule=pms.gen.tms(),
                mount=0,
                axis=(0, 1),
                topology=explicit_tms_topology_config(tmp_path),
            ),
        )

        result = pms.write_functionalized_amorphous_slit(
            str(output_dir),
            config,
            write_pdb=True,
            write_cif=True,
        )

        with open(output_dir / "functionalized_export_slit.pdb", "r") as file_in:
            pdb_text = file_in.read()
        with open(output_dir / "functionalized_export_slit.cif", "r") as file_in:
            cif_text = file_in.read()
        with open(output_dir / "functionalized_export_slit.top", "r") as file_in:
            top_text = file_in.read()
        with open(output_dir / "functionalized_export_slit.itp", "r") as file_in:
            itp_text = file_in.read()

        assert "CONECT" in pdb_text
        assert "_struct_conn.id" in cif_text
        assert " TMS " in pdb_text
        assert '#include "functionalized_export_slit.itp"' in top_text
        assert "FUNCTIONALIZED_EXPORT_SLIT 1" in top_text
        assert "[ atomtypes ]" in itp_text
        assert "[ dihedrals ]" in itp_text
        assert "si 14 28.08600" in itp_text
        assert "117.65432 432.123456" in itp_text
        assert "12.34567 0.98765 2" in itp_text
        assert "OM O1" not in top_text
        assert result.charge_diagnostics is not None
        assert result.charge_diagnostics.is_valid
        assert result.charge_diagnostics.expected_t3_fragment_charge == pytest.approx(0.825)
        assert result.charge_diagnostics.observed_t3_fragment_charge == pytest.approx(0.825, abs=1e-6)
        assert result.charge_diagnostics.derived_t2_fragment_charge == pytest.approx(0.55)
        assert result.charge_diagnostics.t2_site_count == 3
        assert result.charge_diagnostics.t3_site_count == 4
        assert result.charge_diagnostics.final_total_charge == pytest.approx(0.0, abs=1e-6)
        assert sum(row[3] for row in itp_atom_rows(output_dir / "functionalized_export_slit.itp")) == pytest.approx(0.0, abs=1e-6)
        assert result.report.timing_summary.finalize_s > 0
        assert result.report.timing_summary.store_export_s > 0

    def test_explicit_silica_topology_override_changes_functionalized_junction_angles(self, tmp_path):
        output_dir = tmp_path / "functionalized_amorphous_slit_custom_silica"
        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=65 / 957,
            q3_fraction=651 / 957,
            q4_fraction=239 / 957,
            t2_fraction=1 / 957,
            t3_fraction=1 / 957,
            alpha_override=1.0,
        )
        silica_topology = pms.default_silica_topology()
        silica_topology.angle_terms.graft_oxygen_mount_oxygen.angle_deg = 111.11111
        silica_topology.angle_terms.graft_oxygen_mount_oxygen.force_constant = 222.222222

        result = pms.write_functionalized_amorphous_slit(
            str(output_dir),
            pms.FunctionalizedAmorphousSlitConfig(
                slit_config=pms.AmorphousSlitConfig(
                    name="functionalized_custom_silica",
                    repeat_y=1,
                    surface_target=target,
                    silica_topology=silica_topology,
                ),
                ligand=pms.SilaneAttachmentConfig(
                    molecule=pms.gen.tms(),
                    mount=0,
                    axis=(0, 1),
                    topology=explicit_tms_topology_config(tmp_path),
                ),
                progress_settings=pms.FunctionalizedSlitProgressConfig(enabled=False),
            ),
            write_pdb=False,
            write_cif=False,
        )

        with open(output_dir / "functionalized_custom_silica.itp", "r") as file_in:
            itp_text = file_in.read()

        assert result.silica_topology is not silica_topology
        assert result.silica_topology.angle_terms.graft_oxygen_mount_oxygen.angle_deg == pytest.approx(111.11111)
        assert "111.11111 222.222222" in itp_text

    def test_legacy_junction_parameters_still_override_defaults_when_no_silica_model_is_supplied(self, tmp_path):
        output_dir = tmp_path / "functionalized_amorphous_slit_legacy_junction_override"
        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=65 / 957,
            q3_fraction=651 / 957,
            q4_fraction=239 / 957,
            t2_fraction=1 / 957,
            t3_fraction=1 / 957,
            alpha_override=1.0,
        )
        legacy_junctions = pms.SlitJunctionParameters(
            mount_scaffold_bond=topo_mod.GromacsBondParameters.harmonic(
                length_nm=0.16666,
                force_constant=123456.789,
            ),
            scaffold_si_scaffold_o_mount_angle=topo_mod.GromacsAngleParameters.harmonic(
                angle_deg=149.12345,
                force_constant=654.321,
            ),
            oxygen_mount_oxygen_angle=topo_mod.GromacsAngleParameters.harmonic(
                angle_deg=112.22222,
                force_constant=333.444,
            ),
        )

        result = pms.write_functionalized_amorphous_slit(
            str(output_dir),
            pms.FunctionalizedAmorphousSlitConfig(
                slit_config=pms.AmorphousSlitConfig(
                    name="functionalized_legacy_junction_override",
                    repeat_y=1,
                    surface_target=target,
                ),
                ligand=pms.SilaneAttachmentConfig(
                    molecule=pms.gen.tms(),
                    mount=0,
                    axis=(0, 1),
                    topology=explicit_tms_topology_config(
                        tmp_path,
                        junction_parameters=legacy_junctions,
                    ),
                ),
                progress_settings=pms.FunctionalizedSlitProgressConfig(enabled=False),
            ),
            write_pdb=False,
            write_cif=False,
        )

        with open(output_dir / "functionalized_legacy_junction_override.itp", "r") as file_in:
            itp_text = file_in.read()

        assert result.silica_topology.bond_terms.graft_mount_scaffold_si_o.length_nm == pytest.approx(0.16666)
        assert result.silica_topology.angle_terms.graft_scaffold_si_scaffold_o_mount.angle_deg == pytest.approx(149.12345)
        assert result.silica_topology.angle_terms.graft_oxygen_mount_oxygen.angle_deg == pytest.approx(112.22222)
        assert result.silica_topology.bond_terms.graft_mount_scaffold_si_o.origin == (
            "legacy SilaneTopologyConfig.junction_parameters.mount_scaffold_bond"
        )
        assert "0.16666 123456.789000" in itp_text
        assert "149.12345 654.321000" in itp_text
        assert "112.22222 333.444000" in itp_text

    def test_functionalized_write_skips_topology_without_explicit_silane_itp(self, tmp_path):
        output_dir = tmp_path / "functionalized_no_explicit_topology"
        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=65 / 957,
            q3_fraction=651 / 957,
            q4_fraction=239 / 957,
            t2_fraction=1 / 957,
            t3_fraction=1 / 957,
            alpha_override=1.0,
        )

        result = pms.write_functionalized_amorphous_slit(
            str(output_dir),
            pms.FunctionalizedAmorphousSlitConfig(
                slit_config=pms.AmorphousSlitConfig(
                    name="functionalized_no_explicit_topology",
                    repeat_y=1,
                    surface_target=target,
                ),
                ligand=pms.SilaneAttachmentConfig(
                    molecule=pms.gen.tms(),
                    mount=0,
                    axis=(0, 1),
                ),
                progress_settings=pms.FunctionalizedSlitProgressConfig(enabled=False),
            ),
            write_pdb=False,
            write_cif=False,
        )

        assert result.charge_diagnostics is None
        assert not (output_dir / "functionalized_no_explicit_topology.top").exists()
        assert not (output_dir / "functionalized_no_explicit_topology.itp").exists()

    def test_functionalized_export_rejects_charge_mismatched_t3_topology(self, tmp_path):
        output_dir = tmp_path / "functionalized_invalid_t3_charge"
        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=65 / 957,
            q3_fraction=651 / 957,
            q4_fraction=239 / 957,
            t2_fraction=1 / 957,
            t3_fraction=1 / 957,
            alpha_override=1.0,
        )

        with pytest.raises(ValueError, match="base T3 fragment topology"):
            pms.write_functionalized_amorphous_slit(
                str(output_dir),
                pms.FunctionalizedAmorphousSlitConfig(
                    slit_config=pms.AmorphousSlitConfig(
                        name="functionalized_invalid_t3_charge",
                        repeat_y=1,
                        surface_target=target,
                    ),
                    ligand=pms.SilaneAttachmentConfig(
                        molecule=pms.gen.tms(),
                        mount=0,
                        axis=(0, 1),
                        topology=pms.SilaneTopologyConfig(
                            itp_path=str(bundled_tms_template_path()),
                            moleculetype_name="TMS",
                            geminal_cross_terms=explicit_tms_geminal_cross_terms(),
                        ),
                    ),
                    progress_settings=pms.FunctionalizedSlitProgressConfig(enabled=False),
                ),
                write_pdb=False,
                write_cif=False,
            )

    def test_functionalized_export_rejects_missing_geminal_cross_terms(self, tmp_path):
        output_dir = tmp_path / "functionalized_missing_geminal_terms"
        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=65 / 957,
            q3_fraction=651 / 957,
            q4_fraction=239 / 957,
            t2_fraction=1 / 957,
            t3_fraction=1 / 957,
            alpha_override=1.0,
        )

        with pytest.raises(ValueError, match="geminal_cross_terms"):
            pms.write_functionalized_amorphous_slit(
                str(output_dir),
                pms.FunctionalizedAmorphousSlitConfig(
                    slit_config=pms.AmorphousSlitConfig(
                        name="functionalized_missing_geminal_terms",
                        repeat_y=1,
                        surface_target=target,
                    ),
                    ligand=pms.SilaneAttachmentConfig(
                        molecule=pms.gen.tms(),
                        mount=0,
                        axis=(0, 1),
                        topology=explicit_tms_topology_config(
                            tmp_path,
                            include_geminal_terms=False,
                        ),
                    ),
                    progress_settings=pms.FunctionalizedSlitProgressConfig(enabled=False),
                ),
                write_pdb=False,
                write_cif=False,
            )

    def test_teps_exports_use_pdb_aliases_and_consistent_mmcif_metadata(self, tmp_path, repo_root):
        output_dir = tmp_path / "functionalized_amorphous_slit_teps_export"
        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=65 / 957,
            q3_fraction=651 / 957,
            q4_fraction=239 / 957,
            t2_fraction=1 / 957,
            t3_fraction=1 / 957,
            alpha_override=1.0,
        )
        config = pms.FunctionalizedAmorphousSlitConfig(
            slit_config=pms.AmorphousSlitConfig(
                name="functionalized_teps_export",
                repeat_y=1,
                surface_target=target,
            ),
            ligand=pms.SilaneAttachmentConfig(
                molecule=teps_ligand(repo_root),
                mount=0,
                axis=(0, 1),
                rotate_about_axis=False,
            ),
            steric_settings=pms.FunctionalizedSlitStericConfig(clearance_scale=0.60),
            progress_settings=pms.FunctionalizedSlitProgressConfig(enabled=False),
        )

        pms.write_functionalized_amorphous_slit(
            str(output_dir),
            config,
            write_pdb=True,
            write_cif=True,
        )

        with open(output_dir / "functionalized_teps_export.pdb", "r") as file_in:
            pdb_text = file_in.read()
        with open(output_dir / "functionalized_teps_export.cif", "r") as file_in:
            cif_text = file_in.read()

        alias_map = {}
        for line in pdb_text.splitlines():
            if line.startswith("REMARK 250 RESIDUE_ALIAS "):
                parts = line.split()
                alias_map[parts[3]] = parts[5]

        assert "CRYST1" in pdb_text
        assert alias_map["TEPS"] != "TEPS"
        assert alias_map["TEPSG"] != "TEPSG"
        assert len(alias_map["TEPS"]) == 3
        assert len(alias_map["TEPSG"]) == 3

        atom_lines = [
            line for line in pdb_text.splitlines()
            if line.startswith("HETATM")
        ]
        teps_atom_lines = [
            line for line in atom_lines
            if line[17:20].strip() == alias_map["TEPS"]
        ]
        tepsg_atom_lines = [
            line for line in atom_lines
            if line[17:20].strip() == alias_map["TEPSG"]
        ]

        assert teps_atom_lines
        assert tepsg_atom_lines
        for line in teps_atom_lines[:3] + tepsg_atom_lines[:3]:
            assert line[21] == "A"
            assert line[26] == " "
            assert store_mod._decode_hybrid36(4, line[22:26]) >= 1

        entity_tags, entity_rows = cif_loop_rows(cif_text, "_entity.id")
        asym_tags, asym_rows = cif_loop_rows(cif_text, "_struct_asym.id")
        atom_tags, atom_rows = cif_loop_rows(cif_text, "_atom_site.group_PDB")
        conn_tags, conn_rows = cif_loop_rows(cif_text, "_struct_conn.id")

        entity_id_index = entity_tags.index("_entity.id")
        asym_id_index = asym_tags.index("_struct_asym.id")
        atom_entity_index = atom_tags.index("_atom_site.label_entity_id")
        atom_asym_index = atom_tags.index("_atom_site.label_asym_id")
        atom_comp_index = atom_tags.index("_atom_site.label_comp_id")
        conn_asym_1_index = conn_tags.index("_struct_conn.ptnr1_label_asym_id")
        conn_asym_2_index = conn_tags.index("_struct_conn.ptnr2_label_asym_id")

        entity_ids = {row[entity_id_index] for row in entity_rows}
        asym_ids = {row[asym_id_index] for row in asym_rows}

        assert any(row[atom_comp_index] == "TEPS" for row in atom_rows)
        assert any(row[atom_comp_index] == "TEPSG" for row in atom_rows)
        assert {row[atom_entity_index] for row in atom_rows} <= entity_ids
        assert {row[atom_asym_index] for row in atom_rows} <= asym_ids
        assert {row[conn_asym_1_index] for row in conn_rows} <= asym_ids
        assert {row[conn_asym_2_index] for row in conn_rows} <= asym_ids
        assert not (output_dir / "functionalized_teps_export.top").exists()
        assert not (output_dir / "functionalized_teps_export.itp").exists()

    def test_small_teps_functionalized_smoke_populates_timing_summary(self, repo_root):
        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=65 / 957,
            q3_fraction=651 / 957,
            q4_fraction=239 / 957,
            t2_fraction=1 / 957,
            t3_fraction=1 / 957,
            alpha_override=1.0,
        )
        config = pms.FunctionalizedAmorphousSlitConfig(
            slit_config=pms.AmorphousSlitConfig(
                name="functionalized_teps_smoke",
                repeat_y=1,
                surface_target=target,
            ),
            ligand=pms.SilaneAttachmentConfig(
                molecule=teps_ligand(repo_root),
                mount=0,
                axis=(0, 1),
                rotate_about_axis=False,
            ),
            steric_settings=pms.FunctionalizedSlitStericConfig(clearance_scale=0.60),
        )

        result = pms.prepare_functionalized_amorphous_slit_surface(config)

        assert result.report.final_surface == pms.SiliconStateComposition(957, 65, 651, 239, 1, 1)
        assert result.report.timing_summary.base_slit_build_s > 0
        assert result.report.timing_summary.q_state_preparation_s > 0
        assert result.report.timing_summary.t2_attachment_s > 0
        assert result.report.timing_summary.t3_attachment_s > 0
