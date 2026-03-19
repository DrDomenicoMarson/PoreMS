import json
import os
import shutil
import unittest

import porems as pms
import porems.slit as slit_mod
from porems._version import __version__ as EXPECTED_VERSION


class AmorphousSlitPreparationCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_dir = os.path.join(
            os.path.dirname(__file__),
            "output",
            "bare_amorphous_slit_preparation",
        )
        if os.path.isdir(cls.output_dir):
            shutil.rmtree(cls.output_dir)

        cls.config = pms.AmorphousSlitConfig(name="test_bare_amorphous_slit")
        cls.prepared_result = pms.prepare_amorphous_slit_surface(config=cls.config)
        cls.prepared_report = cls.prepared_result.report
        cls.stored_result = pms.write_bare_amorphous_slit(
            cls.output_dir,
            config=cls.config,
        )
        cls.stored_report = cls.stored_result.report

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
        self.assertEqual(self.prepared_report.site_ex, 0)
        self.assertEqual(self.prepared_result.system._site_ex, [])
        self.assertEqual(self.prepared_result.system._pore.get_site_dict()["ex"], {})
        self.assertIsInstance(
            next(iter(self.prepared_result.system._pore.get_sites().values())),
            pms.BindingSite,
        )

        expected_box = [9.605, 19.210, 9.605]
        for actual, expected in zip(self.prepared_report.box_nm, expected_box):
            self.assertAlmostEqual(actual, expected, places=3)

        self.assertAlmostEqual(self.prepared_report.slit_width_nm, 7.0, places=3)
        self.assertAlmostEqual(self.prepared_report.wall_thickness_nm, 6.105, places=3)
        self.assertEqual(self.prepared_report.siloxane_distance_range_nm, (0.4, 0.65))
        self.assertEqual(sorted(self.prepared_result.system._pore.sites_sl_shape), [0])

    def test_prepared_surface_composition_matches_target(self):
        self.assertEqual(self.prepared_report.prepared_surface, self.prepared_report.target_surface)
        self.assertEqual(self.prepared_report.prepared_surface.total_surface_si, 954)
        self.assertEqual(self.prepared_report.prepared_surface.q2_sites, 66)
        self.assertEqual(self.prepared_report.prepared_surface.q3_sites, 650)
        self.assertEqual(self.prepared_report.prepared_surface.q4_sites, 238)
        self.assertFalse(self.prepared_report.used_surface_tolerance)
        self.assertAlmostEqual(self.prepared_report.surface_fraction_tolerance, 0.005)
        self.assertAlmostEqual(
            self.prepared_report.prepared_surface.q2_fraction,
            self.config.surface_target.q2_fraction,
            delta=1e-3,
        )
        self.assertAlmostEqual(
            self.prepared_report.prepared_surface.q3_fraction,
            self.config.surface_target.q3_fraction,
            delta=1e-3,
        )
        self.assertAlmostEqual(
            self.prepared_report.prepared_surface.q4_fraction,
            self.config.surface_target.q4_fraction,
            delta=1e-3,
        )

    def test_repeat_y_one_reaches_requested_surface_target(self):
        result = pms.prepare_amorphous_slit_surface(
            config=pms.AmorphousSlitConfig(
                name="thin_bare_amorphous_slit",
                repeat_y=1,
            )
        )

        self.assertEqual(result.report.site_ex, 0)
        self.assertAlmostEqual(result.report.wall_thickness_nm, 1.3025, places=4)
        self.assertFalse(result.report.used_surface_tolerance)
        self.assertEqual(result.report.prepared_surface, result.report.target_surface)
        self.assertEqual(result.report.prepared_surface.total_surface_si, 957)
        self.assertEqual(result.report.prepared_surface.q2_sites, 66)
        self.assertEqual(result.report.prepared_surface.q3_sites, 652)
        self.assertEqual(result.report.prepared_surface.q4_sites, 239)

    def test_bridge_algebra_matches_q_state_changes(self):
        cases = [
            ((2, 2), (-2, 2, 0)),
            ((2, 1), (-1, 0, 1)),
            ((1, 1), (0, -2, 2)),
        ]

        for pair_counts, expected_delta in cases:
            with self.subTest(pair_counts=pair_counts):
                config = pms.AmorphousSlitConfig(name="bridge_algebra_case", repeat_y=1)
                system = self._build_uncondensed_slit(config)
                total_surface_si = len(system._site_in)
                sites = system._pore.get_sites()
                before = slit_mod._surface_composition(total_surface_si, sites)
                adjacency = slit_mod._build_slit_site_adjacency(
                    system,
                    sorted(system._site_in),
                    config.siloxane_distance_range_nm,
                )
                pair = slit_mod._find_pair(sites, adjacency, *pair_counts)

                self.assertIsNotNone(pair)
                slit_mod._bridge_pair(system, pair)
                slit_mod._consume_pair(adjacency, pair)

                after = slit_mod._surface_composition(
                    total_surface_si,
                    system._pore.get_sites(),
                )
                self.assertEqual(after.q2_sites - before.q2_sites, expected_delta[0])
                self.assertEqual(after.q3_sites - before.q3_sites, expected_delta[1])
                self.assertEqual(after.q4_sites - before.q4_sites, expected_delta[2])

    def test_tolerance_fallback_selects_nearest_realizable_target(self):
        config = pms.AmorphousSlitConfig(name="tolerance_case", repeat_y=1)
        system = self._build_uncondensed_slit(config)
        total_surface_si = len(system._site_in)
        initial_surface = slit_mod._surface_composition(
            total_surface_si,
            system._pore.get_sites(),
        )
        requested_target = config.surface_target
        exact_target = pms.SurfaceComposition(
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
        )
        errors = slit_mod._surface_fraction_errors(
            attempt.prepared_surface,
            requested_target,
        )

        self.assertTrue(attempt.used_surface_tolerance)
        self.assertEqual(
            attempt.target_surface,
            pms.SurfaceComposition(
                total_surface_si=total_surface_si,
                q2_sites=66,
                q3_sites=652,
                q4_sites=239,
            ),
        )
        self.assertEqual(attempt.prepared_surface, attempt.target_surface)
        self.assertTrue(all(error <= config.surface_fraction_tolerance for error in errors))

    def test_prepared_slit_remains_attachable(self):
        result = pms.prepare_amorphous_slit_surface(
            config=pms.AmorphousSlitConfig(name="attachable_bare_amorphous_slit")
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
        self.assertIn("TMS", mol_dict)
        self.assertEqual(len(mol_dict["TMS"]), 1)

    def test_bare_slit_files_are_written(self):
        expected_files = [
            "grid.itp",
            "test_bare_amorphous_slit.gro",
            "test_bare_amorphous_slit.obj",
            "test_bare_amorphous_slit.top",
            "test_bare_amorphous_slit.yml",
            "test_bare_amorphous_slit_report.json",
            "test_bare_amorphous_slit_system.obj",
        ]
        for file_name in expected_files:
            self.assertTrue(os.path.isfile(os.path.join(self.output_dir, file_name)))

        report_path = os.path.join(
            self.output_dir,
            "test_bare_amorphous_slit_report.json",
        )
        with open(report_path, "r") as file_in:
            data = json.load(file_in)

        self.assertEqual(data["site_ex"], 0)
        self.assertEqual(data["siloxane_distance_range_nm"], [0.4, 0.65])
        self.assertEqual(data["surface_fraction_tolerance"], 0.005)
        self.assertFalse(data["used_surface_tolerance"])
        self.assertEqual(
            data["prepared_surface"]["q2_sites"],
            self.stored_report.prepared_surface.q2_sites,
        )
        self.assertEqual(
            data["prepared_surface"]["q3_sites"],
            self.stored_report.prepared_surface.q3_sites,
        )
        self.assertEqual(
            data["prepared_surface"]["q4_sites"],
            self.stored_report.prepared_surface.q4_sites,
        )
        self.assertFalse(
            os.path.exists(
                os.path.join(self.output_dir, "test_bare_amorphous_slit_next_steps.md")
            )
        )

    def test_top_level_exports_and_version(self):
        self.assertEqual(pms.__version__, EXPECTED_VERSION)
        self.assertTrue(callable(pms.prepare_amorphous_slit_surface))
        self.assertTrue(callable(pms.write_bare_amorphous_slit))
        self.assertIsInstance(self.config, pms.AmorphousSlitConfig)
        self.assertIsInstance(self.prepared_result, pms.SlitPreparationResult)
        self.assertIsInstance(self.prepared_report, pms.SlitPreparationReport)
        self.assertIsInstance(self.prepared_report.prepared_surface, pms.SurfaceComposition)
        self.assertTrue(hasattr(pms, "RoughnessProfile"))
        self.assertTrue(hasattr(pms, "SurfaceAreaSummary"))
        self.assertTrue(hasattr(pms, "SurfaceAllocationStats"))
        self.assertTrue(hasattr(pms, "AllocationSummary"))
        self.assertTrue(hasattr(pms, "BindingSite"))
        self.assertTrue(hasattr(pms, "ShapeAttachmentSummary"))


if __name__ == "__main__":
    unittest.main()
