#!/usr/bin/env python3
"""Prepare and store a fully periodic bare amorphous silica slit."""


import argparse
import os

from dataclasses import dataclass

import porems as pms


@dataclass(frozen=True)
class CliOptions:
    """Command-line options for the bare-slit builder.

    Parameters
    ----------
    output_dir : str
        Output directory for the stored slit system and metadata files.
    name : str
        Base name used for the generated slit files.
    slit_width_nm : float
        Requested slit width in nanometers.
    repeat_y : int
        Number of amorphous template copies stacked along the slit-normal
        direction.
    temperature_k : float
        Target simulation temperature in Kelvin.
    q2_fraction : float
        Target ``Q2`` fraction over exposed surface silicon atoms.
    q3_fraction : float
        Target ``Q3`` fraction over exposed surface silicon atoms.
    q4_fraction : float
        Target ``Q4`` fraction over exposed surface silicon atoms.
    """

    output_dir: str
    name: str
    slit_width_nm: float
    repeat_y: int
    temperature_k: float
    q2_fraction: float
    q3_fraction: float
    q4_fraction: float


def parse_args(argv=None):
    """Parse command-line arguments for the bare-slit builder.

    Parameters
    ----------
    argv : list, optional
        Command-line arguments. If omitted, ``sys.argv`` is used.

    Returns
    -------
    options : CliOptions
        Parsed command-line options.
    """
    parser = argparse.ArgumentParser(
        description="Prepare and store a fully periodic bare amorphous silica slit."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("output", "bare_amorphous_silica_slit"),
        help="Directory used to store the slit system and JSON metadata.",
    )
    parser.add_argument(
        "--name",
        default="bare_amorphous_silica_slit",
        help="Base name for the generated slit files.",
    )
    parser.add_argument(
        "--slit-width-nm",
        type=float,
        default=7.0,
        help="Slit width in nanometers.",
    )
    parser.add_argument(
        "--repeat-y",
        type=int,
        default=2,
        help="Number of amorphous template copies stacked along y.",
    )
    parser.add_argument(
        "--temperature-k",
        type=float,
        default=300.0,
        help="Target simulation temperature in Kelvin.",
    )
    parser.add_argument(
        "--q2-fraction",
        type=float,
        default=0.069,
        help="Target exposed-surface Q2 fraction.",
    )
    parser.add_argument(
        "--q3-fraction",
        type=float,
        default=0.681,
        help="Target exposed-surface Q3 fraction.",
    )
    parser.add_argument(
        "--q4-fraction",
        type=float,
        default=0.25,
        help="Target exposed-surface Q4 fraction.",
    )

    args = parser.parse_args(argv)
    return CliOptions(
        output_dir=args.output_dir,
        name=args.name,
        slit_width_nm=args.slit_width_nm,
        repeat_y=args.repeat_y,
        temperature_k=args.temperature_k,
        q2_fraction=args.q2_fraction,
        q3_fraction=args.q3_fraction,
        q4_fraction=args.q4_fraction,
    )


def build_config(options):
    """Translate command-line options into a slit-preparation configuration.

    Parameters
    ----------
    options : CliOptions
        Parsed command-line options.

    Returns
    -------
    config : AmorphousSlitConfig
        Preparation configuration for the slit build.
    """
    surface_target = pms.SurfaceCompositionTarget(
        q2_fraction=options.q2_fraction,
        q3_fraction=options.q3_fraction,
        q4_fraction=options.q4_fraction,
    )
    return pms.AmorphousSlitConfig(
        name=options.name,
        slit_width_nm=options.slit_width_nm,
        repeat_y=options.repeat_y,
        temperature_k=options.temperature_k,
        surface_target=surface_target,
    )


def main(argv=None):
    """Prepare and store the bare amorphous silica slit.

    Parameters
    ----------
    argv : list, optional
        Command-line arguments. If omitted, ``sys.argv`` is used.

    Returns
    -------
    exit_code : int
        Shell-compatible exit code.
    """
    options = parse_args(argv)
    output_dir = os.path.abspath(options.output_dir)
    config = build_config(options)
    result = pms.write_bare_amorphous_slit(output_dir, config=config)
    report = result.report

    print(f"Stored slit in {output_dir}")
    print(
        "Box (nm): "
        f"{report.box_nm[0]:.3f} x {report.box_nm[1]:.3f} x {report.box_nm[2]:.3f}"
    )
    print(f"Slit width (nm): {report.slit_width_nm:.3f}")
    print(f"Wall thickness per side (nm): {report.wall_thickness_nm:.3f}")
    print(f"Exterior sites: {report.site_ex}")
    print(
        "Siloxane search window (nm): "
        f"{report.siloxane_distance_range_nm[0]:.3f}-"
        f"{report.siloxane_distance_range_nm[1]:.3f}"
    )
    print(
        "Surface Q2/Q3/Q4 counts: "
        f"{report.prepared_surface.q2_sites}/"
        f"{report.prepared_surface.q3_sites}/"
        f"{report.prepared_surface.q4_sites}"
    )
    print(f"Siloxane bridges introduced: {report.siloxane_bridges}")

    return 0


if __name__ == "__main__":
    main()
