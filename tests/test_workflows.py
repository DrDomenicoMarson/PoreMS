import json
import os
import shutil
import unittest

import porems as pms
from porems._version import __version__ as EXPECTED_VERSION


class BareAmorphousSlitWorkflowCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_dir = os.path.join(
            os.path.dirname(__file__),
            "output",
            "bare_amorphous_slit_workflow",
        )
        if os.path.isdir(cls.output_dir):
            shutil.rmtree(cls.output_dir)

        cls.config = pms.BareAmorphousSlitConfig(name="test_bare_amorphous_slit")
        cls.result = pms.write_bare_amorphous_slit_study(
            cls.output_dir,
            config=cls.config,
        )
        cls.report = cls.result.report

    def test_periodic_slit_geometry(self):
        self.assertEqual(self.report.site_ex, 0)
        self.assertEqual(self.result.system._site_ex, [])
        self.assertEqual(self.result.system._pore.get_site_dict()["ex"], {})

        expected_box = [9.605, 19.210, 9.605]
        for actual, expected in zip(self.report.box_nm, expected_box):
            self.assertAlmostEqual(actual, expected, places=3)

        self.assertAlmostEqual(self.report.slit_width_nm, 7.0, places=3)
        self.assertAlmostEqual(self.report.wall_thickness_nm, 6.105, places=3)
        self.assertEqual(self.report.siloxane_distance_range_nm, (0.4, 0.65))
        self.assertEqual(sorted(self.result.system._pore.sites_sl_shape), [0])

    def test_surface_composition_matches_target(self):
        self.assertEqual(self.report.final_surface, self.report.target_surface)
        self.assertEqual(self.report.final_surface.total_surface_si, 954)
        self.assertEqual(self.report.final_surface.q2_sites, 66)
        self.assertEqual(self.report.final_surface.q3_sites, 650)
        self.assertEqual(self.report.final_surface.q4_sites, 238)
        self.assertAlmostEqual(
            self.report.final_surface.q2_fraction,
            self.config.surface_target.q2_fraction,
            delta=1e-3,
        )
        self.assertAlmostEqual(
            self.report.final_surface.q3_fraction,
            self.config.surface_target.q3_fraction,
            delta=1e-3,
        )
        self.assertAlmostEqual(
            self.report.final_surface.q4_fraction,
            self.config.surface_target.q4_fraction,
            delta=1e-3,
        )

    def test_study_files_are_written(self):
        expected_files = [
            "grid.itp",
            "test_bare_amorphous_slit.gro",
            "test_bare_amorphous_slit.obj",
            "test_bare_amorphous_slit.top",
            "test_bare_amorphous_slit.yml",
            "test_bare_amorphous_slit_next_steps.md",
            "test_bare_amorphous_slit_study.json",
            "test_bare_amorphous_slit_system.obj",
        ]
        for file_name in expected_files:
            self.assertTrue(os.path.isfile(os.path.join(self.output_dir, file_name)))

        report_path = os.path.join(
            self.output_dir,
            "test_bare_amorphous_slit_study.json",
        )
        with open(report_path, "r") as file_in:
            data = json.load(file_in)

        self.assertEqual(data["site_ex"], 0)
        self.assertEqual(data["siloxane_distance_range_nm"], [0.4, 0.65])
        self.assertEqual(
            data["final_surface"]["q2_sites"],
            self.report.final_surface.q2_sites,
        )
        self.assertEqual(
            data["final_surface"]["q3_sites"],
            self.report.final_surface.q3_sites,
        )
        self.assertEqual(
            data["final_surface"]["q4_sites"],
            self.report.final_surface.q4_sites,
        )

        notes_path = os.path.join(
            self.output_dir,
            "test_bare_amorphous_slit_next_steps.md",
        )
        with open(notes_path, "r") as file_in:
            notes = file_in.read()

        self.assertIn("GAFF2 + AM1-BCC", notes)
        self.assertIn("0.400 - 0.650 nm", notes)
        self.assertIn("relative adsorption-strength benchmark", notes)

    def test_top_level_exports_and_version(self):
        self.assertEqual(pms.__version__, EXPECTED_VERSION)
        self.assertTrue(callable(pms.build_periodic_amorphous_slit))
        self.assertTrue(callable(pms.write_bare_amorphous_slit_study))
        self.assertIsInstance(self.config, pms.BareAmorphousSlitConfig)
        self.assertIsInstance(self.result, pms.SlitBuildResult)
        self.assertIsInstance(self.report, pms.SlitBuildReport)
        self.assertIsInstance(self.report.final_surface, pms.SurfaceComposition)


if __name__ == "__main__":
    unittest.main()
