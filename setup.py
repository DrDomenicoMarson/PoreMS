from pathlib import Path

from setuptools import setup

version_ns = {}
exec(
    (Path(__file__).resolve().parent / "porems" / "_version.py").read_text(encoding="utf-8"),
    version_ns,
)

setup(
    name="porems",
    version=version_ns["__version__"],
    packages=["porems"],
    package_data={"porems": ["templates/*", "py.typed"]},
    install_requires=["numpy", "scipy", "numba", "matplotlib", "pandas", "seaborn", "pyyaml", "tqdm"],
    extras_require={"test": ["pytest", "pytest-cov"]},
    entry_points={
        "console_scripts": [
            "porems-fill-slit=porems.slit_fill:fill_slit_main",
            "porems-slit-density=porems.slit_fill:estimate_guest_density_main",
        ]
    },
    python_requires=">=3.14",
)
