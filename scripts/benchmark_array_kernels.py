#!/usr/bin/env python3
"""Benchmark the array-backed connectivity and slit-preparation kernels."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from statistics import mean
from time import perf_counter

import porems as pms


os.environ.setdefault("MPLCONFIGDIR", "/tmp/porems_benchmark_mpl")


@dataclass(frozen=True)
class BenchmarkResult:
    """One benchmark measurement summary.

    Parameters
    ----------
    name : str
        Human-readable benchmark name.
    repeats : int
        Number of timing repeats.
    timings_s : list[float]
        Wall-clock timings in seconds.
    mean_s : float
        Arithmetic mean of ``timings_s``.
    """

    name: str
    repeats: int
    timings_s: list[float]
    mean_s: float


def benchmark_build(repeats):
    """Benchmark ``PoreKit.build()`` on the cristobalite fixture.

    Parameters
    ----------
    repeats : int
        Number of timing repeats.

    Returns
    -------
    result : BenchmarkResult
        Timing summary for the build benchmark.
    """
    timings = []
    for _repeat in range(repeats):
        block = pms.BetaCristobalit().generate([6, 6, 6], "z")
        kit = pms.PoreKit()
        kit.structure(block)
        start = perf_counter()
        kit.build()
        timings.append(perf_counter() - start)

    return BenchmarkResult(
        name="porekit_build_beta_cristobalit_6x6x6",
        repeats=repeats,
        timings_s=timings,
        mean_s=mean(timings),
    )


def benchmark_slit_prepare(repeats):
    """Benchmark ``prepare_amorphous_slit_surface()`` on the repeat-y-1 case.

    Parameters
    ----------
    repeats : int
        Number of timing repeats.

    Returns
    -------
    result : BenchmarkResult
        Timing summary for the slit-preparation benchmark.
    """
    config = pms.AmorphousSlitConfig(
        name="benchmark_repeat_y1",
        repeat_y=1,
        surface_target=pms.ExperimentalSiliconStateTarget(
            q2_fraction=0.069,
            q3_fraction=0.681,
            q4_fraction=0.25,
            alpha_override=1.0,
        ),
    )

    timings = []
    for _repeat in range(repeats):
        start = perf_counter()
        pms.prepare_amorphous_slit_surface(config=config)
        timings.append(perf_counter() - start)

    return BenchmarkResult(
        name="prepare_amorphous_slit_surface_repeat_y1",
        repeats=repeats,
        timings_s=timings,
        mean_s=mean(timings),
    )


def main():
    """Run the available benchmark scenarios."""
    parser = argparse.ArgumentParser(
        description="Benchmark the array-backed PoreMS kernels."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timing repeats for each benchmark.",
    )
    args = parser.parse_args()

    results = [
        benchmark_build(args.repeats),
        benchmark_slit_prepare(args.repeats),
    ]
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
