from setuptools import setup


setup(
    name="porems",
    packages=["porems"],
    package_data={"porems": ["templates/*"]},
    install_requires=["numpy", "matplotlib", "pandas", "seaborn", "pyyaml"],
    python_requires=">=3.11",
)
