#!/usr/bin/env python3
"""Prepare and store a fully periodic bare amorphous silica slit."""


import os

from dataclasses import dataclass

import porems as pms


@dataclass(frozen=True)
class ScriptConfig:
    """Internal configuration for the bare-slit builder script.

    Parameters
    ----------
    output_dir : str
        Output directory for the stored slit system and metadata files.
    slit_config : AmorphousSlitConfig
        Slit-preparation settings used to generate the stored slit.
    """

    output_dir: str
    slit_config: pms.AmorphousSlitConfig


def build_script_config():
    """Return the fixed script configuration.

    Returns
    -------
    config : ScriptConfig
        Output path and slit-preparation settings used by this script.
    """
    surface_target = pms.ExperimentalSiliconStateTarget(
        q2_fraction=66 / 20000,
        q3_fraction=652 / 20000,
        q4_fraction=1.0 - ((66 + 652) / 20000),
    )
    slit_config = pms.AmorphousSlitConfig(
        name="bare_amorphous_silica_slit",
        slit_width_nm=7.0,
        repeat_y=1,
        temperature_k=300.0,
        surface_target=surface_target,
    )
    return ScriptConfig(
        output_dir=os.path.join("output", "bare_amorphous_silica_slit"),
        slit_config=slit_config,
    )


def main():
    """Prepare and store the bare amorphous silica slit.

    Returns
    -------
    exit_code : int
        Shell-compatible exit code.
    """
    script_config = build_script_config()
    output_dir = os.path.abspath(script_config.output_dir)
    result = pms.write_bare_amorphous_slit(
        output_dir,
        config=script_config.slit_config,
    )
    report = result.report

    print(f"Stored slit in {output_dir}")
    print(
        "Box (nm): "
        f"{report.box_nm[0]:.3f} x {report.box_nm[1]:.3f} x {report.box_nm[2]:.3f}"
    )
    print(f"Slit width (nm): {report.slit_width_nm:.3f}")
    print(f"Wall thickness per side (nm): {report.wall_thickness_nm:.3f}")
    print(f"Exterior sites: {report.site_ex}")
    print(f"Alpha (auto/effective): {report.alpha_auto:.5f}/{report.alpha_effective:.5f}")
    print(
        "Siloxane search window (nm): "
        f"{report.siloxane_distance_range_nm[0]:.3f}-"
        f"{report.siloxane_distance_range_nm[1]:.3f}"
    )
    print(
        "Surface Q2/Q3/Q4/T2/T3 counts: "
        f"{report.final_surface.q2_sites}/"
        f"{report.final_surface.q3_sites}/"
        f"{report.final_surface.q4_sites}/"
        f"{report.final_surface.t2_sites}/"
        f"{report.final_surface.t3_sites}"
    )
    print(f"Siloxane bridges introduced: {report.siloxane_bridges}")

    return 0


if __name__ == "__main__":
    main()
