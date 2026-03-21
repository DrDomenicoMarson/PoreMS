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
    install_requires=["numpy", "matplotlib", "pandas", "seaborn", "pyyaml"],
    python_requires=">=3.14",
)
