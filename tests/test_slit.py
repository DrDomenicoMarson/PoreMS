import inspect
import json
import os
from pathlib import Path
import numpy as np
import pytest

import porems as pms
import porems.slit as slit_mod
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
            q4_fraction=0.5,
            alpha_override=0.5,
        )
        surface_target = slit_mod._surface_target_from_experimental(target, 0.5)

        assert surface_target == pms.SiliconStateFractions(1.0, 0.0, 0.0)

    def test_alpha_override_precedence(self):
        target = pms.ExperimentalSiliconStateTarget(
            q2_fraction=0.02,
            q3_fraction=0.03,
            q4_fraction=0.95,
            alpha_override=0.2,
        )
        alpha_auto, alpha_effective = slit_mod._effective_alpha(100, 1000, target)

        assert alpha_auto == pytest.approx(0.1, abs=1e-7)
        assert alpha_effective == pytest.approx(0.2, abs=1e-7)

    def test_invalid_alpha_target_combinations_raise(self):
        with pytest.raises(ValueError):
            slit_mod._surface_target_from_experimental(
                pms.ExperimentalSiliconStateTarget(
                    q2_fraction=0.5,
                    q3_fraction=0.0,
                    q4_fraction=0.5,
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
            "grid.itp",
            "test_bare_amorphous_slit.gro",
            "test_bare_amorphous_slit.top",
            "test_bare_amorphous_slit.yml",
            "test_bare_amorphous_slit_report.json",
        ]
        for file_name in expected_files:
            assert os.path.isfile(os.path.join(self.output_dir, file_name))

        assert not (os.path.exists(os.path.join(self.output_dir, "test_bare_amorphous_slit.obj")))
        assert not (os.path.exists(
                os.path.join(self.output_dir, "test_bare_amorphous_slit_system.obj")
            ))

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

    def test_top_level_exports_and_version(self):
        assert pms.__version__ == EXPECTED_VERSION
        assert callable(pms.prepare_amorphous_slit_surface)
        assert callable(pms.write_bare_amorphous_slit)
        assert callable(pms.prepare_functionalized_amorphous_slit_surface)
        assert callable(pms.write_functionalized_amorphous_slit)
        assert isinstance(self.config, pms.AmorphousSlitConfig)
        assert isinstance(self.surface_target, pms.ExperimentalSiliconStateTarget)
        assert isinstance(self.prepared_result, pms.SlitPreparationResult)
        assert isinstance(self.prepared_report, pms.SlitPreparationReport)
        assert isinstance(self.prepared_report.prepared_surface, pms.SiliconStateComposition)
        assert isinstance(self.prepared_report.preparation_diagnostics, pms.SurfacePreparationDiagnostics)
        assert hasattr(pms, "SiliconStateFractions")
        assert hasattr(pms, "SurfacePreparationDiagnostics")
        assert hasattr(pms, "SilaneAttachmentConfig")
        assert hasattr(pms, "SlitTimingSummary")
        assert hasattr(pms, "FunctionalizedSlitProgressConfig")
        assert hasattr(pms, "FunctionalizedSlitStericConfig")
        assert hasattr(pms, "FunctionalizedAmorphousSlitConfig")
        assert hasattr(pms, "FunctionalizedSlitResult")
        assert hasattr(pms, "GraphBond")
        assert hasattr(pms, "GraphAngle")
        assert hasattr(pms, "AttachmentRecord")
        assert hasattr(pms, "AssembledStructureGraph")


class TestFunctionalizedAmorphousSlit:
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

        assert result.report.timing_summary.finalize_s > 0
        assert result.report.timing_summary.store_export_s > 0
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
        atom_records, molecule_serials = store._collect_structure_records(use_atom_names=True)
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

        assert "CONECT" in pdb_text
        assert "_struct_conn.id" in cif_text
        assert " TMS " in pdb_text
        mol_counts = {
            short: len(mols)
            for short, mols in result.system._pore.get_mol_dict().items()
        }
        assert f"OM {mol_counts['OM']}" in top_text
        assert f"SI {mol_counts['SI']}" in top_text
        assert f"TMS {mol_counts['TMS']}" in top_text
        assert f"TMSG {mol_counts['TMSG']}" in top_text
        assert result.report.timing_summary.finalize_s > 0
        assert result.report.timing_summary.store_export_s > 0

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
