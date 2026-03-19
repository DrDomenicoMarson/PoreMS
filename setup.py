import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="porems",
    version="0.4.0",
    author="Hamzeh Kraus",
    author_email="kraus@itt.uni-stuttgart.de",
    description="Pore Generator for Molecular Simulations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/porems/PoreMS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11,<3.15',
    install_requires=['numpy', 'matplotlib', 'pandas', 'seaborn', 'pyyaml'],
    include_package_data=True,
)
