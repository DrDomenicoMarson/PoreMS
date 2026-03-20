#!/usr/bin/env python3
"""Prepare and store a periodic functionalized amorphous silica slit."""


import os

from dataclasses import dataclass

import porems as pms


@dataclass(frozen=True)
class ScriptConfig:
    """Internal configuration for the functionalized-slit builder script.

    Parameters
    ----------
    output_dir : str
        Output directory for the stored slit system and metadata files.
    slit_config : FunctionalizedAmorphousSlitConfig
        Functionalized slit-preparation settings used to generate the stored
        slit.
    """

    output_dir: str
    slit_config: pms.FunctionalizedAmorphousSlitConfig


def build_script_config():
    """Return the fixed script configuration.

    Returns
    -------
    config : ScriptConfig
        Output path and slit-preparation settings used by this script.
    """
    surface_target = pms.ExperimentalSiliconStateTarget(
        q2_fraction=1.01 / 100.0,
        q3_fraction=12.75 / 100.0,
        q4_fraction=68.58 / 100.0,
        t2_fraction=6.22 / 100.0,
        t3_fraction=11.44 / 100.0,
        alpha_override=0.328,
    )
    bare_slit = pms.AmorphousSlitConfig(
        name="functionalized_amorphous_silica_slit",
        slit_width_nm=7.0,
        repeat_y=1,
        temperature_k=300.0,
        surface_target=surface_target,
    )
    slit_config = pms.FunctionalizedAmorphousSlitConfig(
        slit_config=bare_slit,
        ligand=pms.SilaneAttachmentConfig(
            molecule=pms.gen.tms(),
            mount=0,
            axis=(0, 1),
        ),
    )
    return ScriptConfig(
        output_dir=os.path.join("output", "functionalized_amorphous_silica_slit"),
        slit_config=slit_config,
    )


def main():
    """Prepare and store the functionalized amorphous silica slit.

    Returns
    -------
    exit_code : int
        Shell-compatible exit code.
    """
    script_config = build_script_config()
    output_dir = os.path.abspath(script_config.output_dir)
    result = pms.write_functionalized_amorphous_slit(
        output_dir,
        config=script_config.slit_config,
    )
    report = result.report

    print(f"Stored functionalized slit in {output_dir}")
    print(
        "Box (nm): "
        f"{report.box_nm[0]:.3f} x {report.box_nm[1]:.3f} x {report.box_nm[2]:.3f}"
    )
    print(f"Slit width (nm): {report.slit_width_nm:.3f}")
    print(f"Wall thickness per side (nm): {report.wall_thickness_nm:.3f}")
    print(f"Exterior sites: {report.site_ex}")
    print(f"Alpha (auto/effective): {report.alpha_auto:.5f}/{report.alpha_effective:.5f}")
    print(
        "Experimental Q2/Q3/Q4/T2/T3 (%): "
        f"{100.0 * report.experimental_target.q2_fraction:.2f}/"
        f"{100.0 * report.experimental_target.q3_fraction:.2f}/"
        f"{100.0 * report.experimental_target.q4_fraction:.2f}/"
        f"{100.0 * report.experimental_target.t2_fraction:.2f}/"
        f"{100.0 * report.experimental_target.t3_fraction:.2f}"
    )
    print(
        "Siloxane search window (nm): "
        f"{report.siloxane_distance_range_nm[0]:.3f}-"
        f"{report.siloxane_distance_range_nm[1]:.3f}"
    )
    print(
        "Target Q2/Q3/Q4/T2/T3 counts: "
        f"{report.target_surface.q2_sites}/"
        f"{report.target_surface.q3_sites}/"
        f"{report.target_surface.q4_sites}/"
        f"{report.target_surface.t2_sites}/"
        f"{report.target_surface.t3_sites}"
    )
    print(
        "Final Q2/Q3/Q4/T2/T3 counts: "
        f"{report.final_surface.q2_sites}/"
        f"{report.final_surface.q3_sites}/"
        f"{report.final_surface.q4_sites}/"
        f"{report.final_surface.t2_sites}/"
        f"{report.final_surface.t3_sites}"
    )
    print(f"Siloxane bridges introduced: {report.siloxane_bridges}")
    print(
        "Surface diagnostics: "
        f"stripped Si={report.preparation_diagnostics.stripped_silicon_total}, "
        f"orphan O removed={report.preparation_diagnostics.removed_orphan_oxygen}, "
        f"bridge O inserted={report.preparation_diagnostics.inserted_bridge_oxygen}, "
        f"final handles={report.preparation_diagnostics.final_surface_oxygen_handles}, "
        f"final framework O={report.preparation_diagnostics.final_framework_oxygen}"
    )

    return 0


if __name__ == "__main__":
    main()
