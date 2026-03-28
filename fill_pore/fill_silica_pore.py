#!/usr/bin/env python3
"""Compatibility wrapper for the packaged slit-fill CLI."""

from __future__ import annotations

from collections.abc import Sequence

from porems.slit_fill import SlitFillReport, fill_slit_main


def main(argv: Sequence[str] | None = None) -> SlitFillReport:
    """Run the packaged slit-fill CLI.

    Parameters
    ----------
    argv : sequence[str] or None, optional
        Optional argument vector. When omitted, arguments are read from the
        process command line.

    Returns
    -------
    SlitFillReport
        Structured report returned by :func:`porems.slit_fill.fill_slit_main`.
    """

    return fill_slit_main(argv)


if __name__ == "__main__":
    main()
