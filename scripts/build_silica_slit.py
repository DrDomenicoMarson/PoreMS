#!/usr/bin/env python3
"""Build one example amorphous silica slit with the full-topology exporter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import porems as pms


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = Path(pms.__file__).resolve().parent


@dataclass(frozen=True)
class BuildSilicaSlitSettings:
    """Settings for the slit-build example script.

    Parameters
    ----------
    ligand_name : str, optional
        Ligand selector. Supported values are ``"bare"``, ``"TMS"``, and
        ``"TEPS"``.
    teps_itp_path : Path or None, optional
        Optional self-contained flat ``.itp`` file used when ``ligand_name``
        is ``"TEPS"``. The script does not use the legacy helper-topology
        export path anymore, so custom ligands must provide explicit full
        topology input.
    teps_moleculetype_name : str, optional
        Optional explicit ``[ moleculetype ]`` name expected inside
        ``teps_itp_path``.
    output_dir : Path or None, optional
        Optional explicit export directory. When omitted, the script writes to
        ``scripts/built_<LIGAND>``.
    write_pdb : bool, optional
        When ``True``, also write an inspection-oriented PDB file.
    write_cif : bool, optional
        When ``True``, also write an inspection-oriented mmCIF file.
    """

    ligand_name: str = "TMS"
    teps_itp_path: Path | None = None
    teps_moleculetype_name: str = ""
    output_dir: Path | None = None
    write_pdb: bool = True
    write_cif: bool = True


SETTINGS = BuildSilicaSlitSettings()


def _bundled_tms_topology_path():
    """Return the bundled self-contained TMS topology path.

    Returns
    -------
    topology_path : Path
        Absolute path to the package-provided flat TMS ``.itp`` bundle.
    """
    return PACKAGE_DIR / "templates" / "tms_slit.itp"


def _surface_target(ligand_name):
    """Return the example experimental surface target.

    Parameters
    ----------
    ligand_name : str
        Ligand selector used by the script.

    Returns
    -------
    target : pms.ExperimentalSiliconStateTarget
        Surface-state target for the requested build.

    Raises
    ------
    ValueError
        Raised when ``ligand_name`` is not supported by the script.
    """
    ligand_key = ligand_name.upper()
    if ligand_key == "BARE":
        return pms.ExperimentalSiliconStateTarget(
            q2_fraction=1.70 / 100.0,
            q3_fraction=16.75 / 100.0,
            q4_fraction=81.55 / 100.0,
            t2_fraction=0.0,
            t3_fraction=0.0,
            alpha_override=0.328,
        )
    if ligand_key in {"TEPS", "TMS"}:
        return pms.ExperimentalSiliconStateTarget(
            q2_fraction=1.01 / 100.0,
            q3_fraction=12.75 / 100.0,
            q4_fraction=68.58 / 100.0,
            t2_fraction=6.22 / 100.0,
            t3_fraction=11.44 / 100.0,
            alpha_override=0.328,
        )
    raise ValueError(f"Unknown ligand selector {ligand_name!r}.")


def _slit_config(ligand_name):
    """Return the shared base slit configuration for the example script.

    Parameters
    ----------
    ligand_name : str
        Ligand selector used to choose the matching surface target.

    Returns
    -------
    config : pms.AmorphousSlitConfig
        Base slit configuration for the requested build.
    """
    return pms.AmorphousSlitConfig(
        name="test",
        slit_width_nm=7.0,
        repeat_y=1,
        temperature_k=308.0,
        surface_target=_surface_target(ligand_name),
    )


def _teps_topology_config(settings):
    """Return the explicit full-topology input for the TEPS example.

    Parameters
    ----------
    settings : BuildSilicaSlitSettings
        Script settings carrying the optional custom TEPS topology path.

    Returns
    -------
    topology : pms.SilaneTopologyConfig
        Self-contained TEPS topology input forwarded to the slit exporter.

    Raises
    ------
    ValueError
        Raised when the script is asked to build the TEPS case without an
        explicit flat topology bundle.
    """
    if settings.teps_itp_path is None:
        raise ValueError(
            "The TEPS example now requires a self-contained flat .itp bundle "
            "via BuildSilicaSlitSettings.teps_itp_path so the script can use "
            "the full slit topology exporter."
        )

    return pms.SilaneTopologyConfig(
        itp_path=str(settings.teps_itp_path),
        moleculetype_name=settings.teps_moleculetype_name,
    )


def _functionalized_config(settings):
    """Return the functionalized slit configuration for one ligand example.

    Parameters
    ----------
    settings : BuildSilicaSlitSettings
        Script settings carrying the ligand selector and optional topology
        bundle paths.

    Returns
    -------
    config : pms.FunctionalizedAmorphousSlitConfig
        Functionalized slit configuration for the requested ligand.

    Raises
    ------
    ValueError
        Raised when ``settings.ligand_name`` is not ``"TMS"`` or ``"TEPS"``.
    """
    ligand_key = settings.ligand_name.upper()
    if ligand_key == "TMS":
        molecule = pms.gen.tms()
        topology = pms.SilaneTopologyConfig(
            itp_path=str(_bundled_tms_topology_path()),
            moleculetype_name="TMS",
        )
    elif ligand_key == "TEPS":
        molecule = pms.Molecule("TEPS", "TEPS", str(SCRIPT_DIR / "TEPS.pdb"))
        topology = _teps_topology_config(settings)
    else:
        raise ValueError(
            "Functionalized slit example supports only 'TMS' and 'TEPS'."
        )

    return pms.FunctionalizedAmorphousSlitConfig(
        slit_config=_slit_config(ligand_key),
        ligand=pms.SilaneAttachmentConfig(
            molecule=molecule,
            mount=0,
            axis=(0, 1),
            rotate_about_axis=True,
            rotate_step_deg=30.0,
            topology=topology,
        ),
        progress_settings=pms.FunctionalizedSlitProgressConfig(),
    )


def _output_dir(settings):
    """Return the export directory for one script run.

    Parameters
    ----------
    settings : BuildSilicaSlitSettings
        Script settings carrying the ligand selector.

    Returns
    -------
    output_dir : Path
        Output directory stored below the script directory unless an explicit
        path was configured.
    """
    if settings.output_dir is not None:
        return settings.output_dir
    return SCRIPT_DIR / f"built_{settings.ligand_name.upper()}"


def main(settings=SETTINGS):
    """Build and store the configured slit example.

    Parameters
    ----------
    settings : BuildSilicaSlitSettings, optional
        Script settings describing which example to build and which optional
        inspection files to write.

    Returns
    -------
    result : pms.SlitPreparationResult or pms.FunctionalizedSlitResult
        Export result returned by the selected slit writer.

    Raises
    ------
    ValueError
        Raised when ``settings.ligand_name`` is not supported by the script.
    """
    ligand_key = settings.ligand_name.upper()
    output_dir = str(_output_dir(settings))

    if ligand_key == "BARE":
        return pms.write_bare_amorphous_slit(
            output_dir,
            config=_slit_config(ligand_key),
            write_pdb=settings.write_pdb,
            write_cif=settings.write_cif,
        )

    if ligand_key in {"TEPS", "TMS"}:
        return pms.write_functionalized_amorphous_slit(
            output_dir,
            config=_functionalized_config(settings),
            write_pdb=settings.write_pdb,
            write_cif=settings.write_cif,
        )

    raise ValueError(f"Unknown ligand selector {settings.ligand_name!r}.")


if __name__ == "__main__":
    main()
