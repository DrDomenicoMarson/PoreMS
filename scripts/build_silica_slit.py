#!/usr/bin/env python3

import porems as pms

LIGAND = "TEPS" # can be "TEPS" "TMS" or "bare"

if LIGAND == "TEPS":
    lig = pms.Molecule("TEPS", "TEPS", "TEPS.pdb")
else:
    lig = pms.gen.tms()
# elif LIGAND == "bare":
#     lig = None
# else:
#     raise ValueError(f"Unknown ligand {LIGAND}")


if LIGAND != "bare":
    surface_target = pms.ExperimentalSiliconStateTarget(
        q2_fraction=1.01 / 100.0,
        q3_fraction=12.75 / 100.0,
        q4_fraction=68.58 / 100.0,
        t2_fraction=6.22 / 100.0,
        t3_fraction=11.44 / 100.0,
        alpha_override=0.328,
    )
else:
    surface_target = pms.ExperimentalSiliconStateTarget(
        q2_fraction=1.70 / 100.0,
        q3_fraction=16.75 / 100.0,
        q4_fraction=81.55 / 100.0,
        t2_fraction=0.0,
        t3_fraction=0.0,
        alpha_override=0.328,
    )


bare_slit = pms.AmorphousSlitConfig(
    name="test",
    slit_width_nm=7.0,
    repeat_y=1,
    temperature_k=308.0,
    surface_target=surface_target,
)
slit_config = pms.FunctionalizedAmorphousSlitConfig(
    slit_config=bare_slit,
    ligand=pms.SilaneAttachmentConfig(
        molecule=lig,
        mount=0,
        rotate_about_axis=False,
        rotate_step_deg=30.0,
        axis=(0, 1),
    ),
    progress_settings=pms.FunctionalizedSlitProgressConfig(),
)


output_dir = f"./built_{LIGAND}"
result = pms.write_functionalized_amorphous_slit(output_dir, write_pdb=True, write_cif=True,
                                                 config=slit_config,
    )
report = result.report
