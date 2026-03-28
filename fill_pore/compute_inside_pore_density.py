#!/usr/bin/env python3
"""Compatibility wrapper for the packaged slit-density CLI."""

from __future__ import annotations

from collections.abc import Sequence

from porems.slit_fill import SlitDensityReport, estimate_guest_density_main


def main(argv: Sequence[str] | None = None) -> SlitDensityReport:
    """Run the packaged slit-density CLI.

    Parameters
    ----------
    argv : sequence[str] or None, optional
        Optional argument vector. When omitted, arguments are read from the
        process command line.

    Returns
    -------
    SlitDensityReport
        Structured report returned by
        :func:`porems.slit_fill.estimate_guest_density_main`.
    """

    return estimate_guest_density_main(argv)


if __name__ == "__main__":
    main()
