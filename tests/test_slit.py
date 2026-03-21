import json
import os
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
