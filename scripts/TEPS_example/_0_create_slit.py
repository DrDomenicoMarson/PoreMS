#!/usr/bin/env python3

import porems as pms

SLIT_WIDTH = 7.0

systems = {
    "0_0": pms.ExperimentalSiliconStateTarget(alpha_override=0.400, q2_fraction=0.0170, q3_fraction=0.1675),
    "9_1": pms.ExperimentalSiliconStateTarget(alpha_override=0.397, q2_fraction=0.0133, q3_fraction=0.1735, t2_fraction=0.0195, t3_fraction=0.0367),
    "8_2": pms.ExperimentalSiliconStateTarget(alpha_override=0.391, q2_fraction=0.0158, q3_fraction=0.1637, t2_fraction=0.0508, t3_fraction=0.0818),
    "7_3": pms.ExperimentalSiliconStateTarget(alpha_override=0.328, q2_fraction=0.0101, q3_fraction=0.1275, t2_fraction=0.0622, t3_fraction=0.1144),
    "6_4": pms.ExperimentalSiliconStateTarget(alpha_override=0.277, q2_fraction=0.0149, q3_fraction=0.0976, t2_fraction=0.0605, t3_fraction=0.1836),
}



def do_stuff(select):
    NAME = f"msn_{select}"
    slit_config = pms.AmorphousSlitConfig(
        name=NAME,
        slit_width_nm=SLIT_WIDTH,
        repeat_y=1,
        surface_target=systems[select],
    )

    if select == "0_0":
        result = pms.prepare_amorphous_slit_surface(slit_config)
        print(result.report.final_surface)
        result = pms.write_bare_amorphous_slit(f"msn_{select}/", slit_config, write_pdb=False, write_cif=False)
        print(result.bare_charge_diagnostics.is_neutral)
    else:
        geminal_cross_terms = pms.SilaneGeminalCrossTerms(
                first_ligand_atom_name="CA1",
                scaffold_oxygen_mount_ligand_angle=pms.GromacsAngleParameters.harmonic(
                    angle_deg=103.7000444,
                    force_constant=836.8,
                    ),
                geminal_oxygen_mount_ligand_angle=pms.GromacsAngleParameters.harmonic(
                    angle_deg=103.7000444,
                    force_constant=1034.033760,
                    ),
                geminal_dihedrals=(),
                )

        topology = pms.SilaneTopologyConfig(
            itp_path="TEPS_T2.itp",
            moleculetype_name="TPS",
            geminal_cross_terms=geminal_cross_terms,
            )

        ligand = pms.SilaneAttachmentConfig(
            molecule=pms.Molecule("TPS", "TPS", "TEPS.pdb"),
            mount=3,        # 0-based atom indexes
            axis=(3, 2),    # 0-based atom indexes
            rotate_about_axis=True,
            rotate_step_deg=20.0,
            topology=topology,
            )

        functionalized_config = pms.FunctionalizedAmorphousSlitConfig(
            slit_config=slit_config,
            ligand=ligand
            )

        result = pms.write_functionalized_amorphous_slit(
            f"msn_{select}/",
            functionalized_config,
            write_pdb=False, write_cif=False,
            )
        print(result.report.final_surface)
        print(result.charge_diagnostics.is_valid)

for sel in ["0_0", "9_1", "8_2", "7_3", "6_4"]:
    do_stuff(sel)
