import os
import copy
import warnings

import matplotlib.pyplot as plt
import pytest

import porems as pms
import porems.store as store_mod


pytestmark = pytest.mark.usefixtures("module_workspace")


class TestUserModel:


    #########
    # Utils #
    #########
    def test_utils(self):
        text_link = "output/test/test.txt"
        text_copy_link = "output/test/test_copy.txt"
        pickle_link = "output/test/test.pkl"

        pms.utils.mkdirp("output/test")

        with open(text_link, "w") as file_out:
            file_out.write("TEST")
        pms.utils.copy(text_link, text_copy_link)
        pms.utils.replace(text_copy_link, "TEST", "DOTA")
        with open(text_copy_link, "r") as file_in:
            for line in file_in:
                assert line == "DOTA\n"

        assert pms.utils.column([[1, 1, 1], [2, 2, 2]]) == [[1, 2], [1, 2], [1, 2]]

        pms.utils.save([1, 1, 1], pickle_link)
        assert pms.utils.load(pickle_link) == [1, 1, 1]

        assert round(pms.utils.mumol_m2_to_mols(3, 100), 4) == 180.66
        assert round(pms.utils.mols_to_mumol_m2(180, 100), 4) == 2.989
        assert round(pms.utils.mmol_g_to_mumol_m2(0.072, 512), 2) == 0.14
        assert round(pms.utils.mmol_l_to_mols(30, 1000), 4) == 18.066
        assert round(pms.utils.mols_to_mmol_l(18, 1000), 4) == 29.8904

        print()
        pms.utils.toc(pms.utils.tic(), message="Test", is_print=True)
        assert round(pms.utils.toc(pms.utils.tic(), is_print=True)) == 0


    ############
    # Geometry #
    ############
    def test_geometry(self):
        vec_a = [1, 1, 2]
        vec_b = [0, 3, 2]

        print()

        assert round(pms.geom.dot_product(vec_a, vec_b), 4) == 7
        assert round(pms.geom.length(vec_a), 4) == 2.4495
        assert [round(x, 4) for x in pms.geom.vector(vec_a, vec_b)] == [-1, 2, 0]
        with pytest.raises(ValueError, match="Wrong dimensions"):
            pms.geom.vector([0, 1], [0, 0, 0])
        assert [round(x, 4) for x in pms.geom.unit(vec_a)] == [0.4082, 0.4082, 0.8165]
        assert [round(x, 4) for x in pms.geom.cross_product(vec_a, vec_b)] == [-4, -2, 3]
        assert round(pms.geom.angle(vec_a, vec_b), 4) == 37.5714
        assert round(pms.geom.angle_polar(vec_a), 4) == 0.7854
        assert round(pms.geom.angle_azi(vec_b), 4) == 0.9828
        assert round(pms.geom.angle_azi([0, 0, 0]), 4) == 1.5708
        assert [round(x, 4) for x in pms.geom.main_axis(1)] == [1, 0, 0]
        assert [round(x, 4) for x in pms.geom.main_axis(2)] == [0, 1, 0]
        assert [round(x, 4) for x in pms.geom.main_axis(3)] == [0, 0, 1]
        assert [round(x, 4) for x in pms.geom.main_axis("x")] == [1, 0, 0]
        assert [round(x, 4) for x in pms.geom.main_axis("y")] == [0, 1, 0]
        assert [round(x, 4) for x in pms.geom.main_axis("z")] == [0, 0, 1]
        with pytest.raises(ValueError, match="Wrong axis definition"):
            pms.geom.main_axis("h")
        with pytest.raises(ValueError, match="Wrong axis definition"):
            pms.geom.main_axis(100)
        with pytest.raises(ValueError, match="Wrong axis definition"):
            pms.geom.main_axis(0.1)
        assert [round(x, 4) for x in pms.geom.rotate(vec_a, "x", 90, True)] == [1.0, -2.0, 1.0]
        with pytest.raises(ValueError, match="Wrong vector dimensions"):
            pms.geom.rotate(vec_a, [0, 1, 2, 3], 90, True)
        with pytest.raises(ValueError, match="Wrong axis definition"):
            pms.geom.rotate(vec_a, "h", 90, True)


    ############
    # Database #
    ############
    def test_database(self):
        print()
        assert pms.db.get_mass("H") == 1.0079
        assert pms.db.get_element("Si") == "Si"
        assert pms.db.get_element("Ci") == "C"
        assert pms.db.get_pdb_element("CA") == "C"
        assert pms.db.get_pdb_element("CD1") == "C"
        assert pms.db.get_pdb_element("SI1") == "Si"
        assert pms.db.get_covalent_radius("OM1") == pytest.approx(0.066, abs=1e-7)
        with pytest.raises(ValueError, match="Atom name not found"):
            pms.db.get_mass("DOTA")


    ########
    # Atom #
    ########
    def test_atom(self):
        atom = pms.Atom([0, 0, 0], "O", "O", 5)

        atom.set_pos([0.0, 0.1, 0.2])
        atom.set_atom_type("H")
        atom.set_name("HO1")
        atom.set_residue(0)

        assert atom.get_pos() == [0.0, 0.1, 0.2]
        assert atom.get_atom_type() == "H"
        assert atom.get_name() == "HO1"
        assert atom.get_residue() == 0

        assert atom.__str__() == "   Residue Name Type    x    y    z\n0        0  HO1    H  0.0  0.1  0.2"


    ############
    # Molecule #
    ############
    def test_molecule_loading(self):
        mol_gro = pms.Molecule(inp="data/benzene.gro")
        mol_pdb = pms.Molecule(inp="data/benzene.pdb")
        mol_mol2 = pms.Molecule(inp="data/benzene.mol2")

        mol_atom = pms.Molecule(inp=mol_mol2.get_atom_list())
        mol_concat = pms.Molecule(inp=[mol_gro, mol_pdb])

        mol_append = pms.Molecule(inp="data/benzene.gro")
        mol_append.append(mol_gro)

        pos_gro = [[round(x, 4) for x in col] for col in mol_gro.column_pos()]
        pos_pdb = [[round(x, 4) for x in col] for col in mol_pdb.column_pos()]
        pos_mol2 = [[round(x, 4) for x in col] for col in mol_mol2.column_pos()]
        pos_atom = [[round(x, 4) for x in col] for col in mol_atom.column_pos()]
        pos_concat = [[round(x, 4) for x in col] for col in mol_concat.column_pos()]
        pos_append = [[round(x, 4) for x in col] for col in mol_append.column_pos()]

        assert pos_gro == pos_pdb
        assert pos_gro == pos_mol2
        assert pos_gro == pos_atom
        assert [col+col for col in pos_gro] == pos_concat
        assert [col+col for col in pos_gro] == pos_append
        assert mol_gro.get_bonds() == []
        assert mol_pdb.get_bonds() == []
        assert len(mol_mol2.get_bonds()) == 12
        assert len(mol_gro.infer_bonds()) == 12

        print()
        with pytest.raises(ValueError, match="Unsupported filetype"):
            pms.Molecule(inp="data/benzene.DOTA")

    def test_molecule_loads_pdb_conect_and_preserves_bonds(self):
        pdb_path = os.path.join("output", "bonded_probe.pdb")
        with open(pdb_path, "w") as file_out:
            file_out.write(
                "HETATM    1 SI1 TMS A   1       0.000   0.000   0.000  1.00  0.00          Si  \n"
                "HETATM    2  O1 TMS A   1       1.640   0.000   0.000  1.00  0.00           O  \n"
                "HETATM    3  C1 TMS A   1       2.800   0.000   0.000  1.00  0.00           C  \n"
                "CONECT    1    2\n"
                "CONECT    2    1    3\n"
                "CONECT    3    2\n"
                "TER\nEND\n"
            )

        mol = pms.Molecule(inp=pdb_path)
        assert mol.get_bonds() == [(0, 1), (1, 2)]

        graph = pms.Store(mol, "output").assembled_graph(use_atom_names=True)
        assert len(graph.bonds) == 2
        assert all(bond.provenance == "ligand_explicit" for bond in graph.bonds)

    def test_molecule_pdb_name_fallback_keeps_phenyl_atoms_as_carbon(self):
        pdb_path = os.path.join("output", "phenyl_silane_probe.pdb")
        with open(pdb_path, "w") as file_out:
            file_out.write(
                "HETATM    1 SI1 PHS A   1       0.000   0.000   0.000  1.00  0.00              \n"
                "HETATM    2  CA PHS A   1       1.860   0.000   0.000  1.00  0.00              \n"
                "HETATM    3 CD1 PHS A   1       2.560   1.210   0.000  1.00  0.00              \n"
                "HETATM    4 CE1 PHS A   1       3.960   1.210   0.000  1.00  0.00              \n"
                "CONECT    1    2\n"
                "CONECT    2    1    3\n"
                "CONECT    3    2    4\n"
                "CONECT    4    3\n"
                "TER\nEND\n"
            )

        mol = pms.Molecule(inp=pdb_path)

        assert [mol.get_atom_type(index) for index in range(mol.get_num())] == ["Si", "C", "C", "C"]
        assert mol.get_bonds() == [(0, 1), (1, 2), (2, 3)]

    def test_molecule_bond_graph_is_preserved_through_edits(self):
        mol = pms.Molecule()
        mol.add("Si", [0.0, 0.0, 0.0], name="SI1")
        mol.add("O", 0, r=0.164, name="O1")
        mol.add("H", 1, r=0.098, name="H1")

        assert mol.get_bonds() == [(0, 1), (1, 2)]

        mol_copy = copy.deepcopy(mol)
        assert mol_copy.get_bonds() == [(0, 1), (1, 2)]

        mol.switch_atom_order(0, 2)
        assert mol.get_bonds() == [(0, 1), (1, 2)]

        mol.delete(2)
        assert mol.get_bonds() == [(0, 1)]

    def test_molecule_properties(self):
        mol = pms.Molecule(inp="data/benzene.gro")

        assert mol.pos(0) == [0.0935, 0.0000, 0.3143]
        assert [round(x, 4) for x in mol.bond(0, 1)] == [0.1191, 0.0, 0.0687]
        assert mol.bond([1, 0, 0], [0, 0, 0]) == [-1, 0, 0]
        assert mol.get_box() == [0.4252, 0.001, 0.491]
        assert [round(x, 4) for x in mol.centroid()] == [0.2126, 0.0, 0.2455]
        assert [round(x, 4) for x in mol.com()] == [0.2126, 0.0, 0.2455]

    def test_molecule_editing(self):
        mol = pms.Molecule(inp="data/benzene.gro")

        mol.translate([0, 0.1, 0.2])
        assert [round(x, 4) for x in mol.pos(3)] == [0.3317, 0.1000, 0.3768]
        mol.rotate("x", 45)
        assert [round(x, 4) for x in mol.pos(3)] == [0.3317, -0.1957, 0.3371]
        mol.move(0, [1, 1, 1])
        assert [round(x, 4) for x in mol.pos(3)] == [1.2382, 1.0972, 0.9028]
        mol.zero()
        assert [round(x, 4) for x in mol.pos(3)] == [0.3317, 0.2222, 0.1250]
        mol.put(3, [0, 0, 0])
        assert [round(x, 4) for x in mol.pos(3)] == [0.0000, 0.0000, 0.0000]
        mol.part_move([0, 1], [2, 3, 4], 0.5)
        mol.part_move([0, 1], 1, 0.5)
        assert [round(x, 4) for x in mol.pos(3)] == [0.3140, -0.1281, 0.1281]
        mol.part_rotate([0, 1], [2, 3, 4], 45, 1)
        mol.part_rotate([0, 1], 1, 45, 1)
        assert [round(x, 4) for x in mol.pos(3)] == [-0.1277, 0.0849, -0.3176]
        mol.part_angle([0, 1], [1, 2], [1, 2, 3, 4], 45, 1)
        assert [round(x, 4) for x in mol.pos(3)] == [-0.1360, -0.1084, -0.3068]
        mol.part_angle([0, 0, 1], [0, 1, 0], 1, 45, 1)
        assert [round(x, 4) for x in mol.pos(3)] == [-0.1360, -0.1084, -0.3068]

        print()

        with pytest.raises(ValueError, match="Wrong input"):
            mol._vector(0.1, 0.1)
        with pytest.raises(ValueError, match="Wrong dimensions"):
            mol._vector([0, 0], [0, 0])

        with pytest.raises(ValueError, match="Wrong bond input"):
            mol.part_angle([0, 0, 1, 0], [0, 1, 0, 0], 1, 45, 1)
        with pytest.raises(ValueError, match="Wrong bond dimensions"):
            mol.part_angle([0, 0], [0, 1, 2], 1, 45, 1)

    def test_molecule_creation(self):
        mol = pms.Molecule()

        mol.add("C", [0, 0.1, 0.2])
        mol.add("C", 0, r=0.1, theta=90)
        mol.add("C", 1, [0, 1], r=0.1, theta=90)
        mol.add("C", 2, [0, 2], r=0.1, theta=90, phi=45)
        assert [round(x, 4) for x in mol.pos(3)] == [0.0500, 0.0500, 0.0293]
        mol.delete(2)
        assert [round(x, 4) for x in mol.pos(2)] == [0.0500, 0.0500, 0.0293]
        mol.add("C", [0, 0.1, 0.2])
        assert mol.overlap() == {0: [3]}
        mol.switch_atom_order(0, 2)
        assert [round(x, 4) for x in mol.pos(0)] == [0.0500, 0.0500, 0.0293]
        mol.set_atom_type(0, "R")
        assert mol.get_atom_list()[0].get_atom_type() == "R"
        assert mol.get_atom_type(0) == "R"
        mol.set_atom_name(0, "RuX")
        assert mol.get_atom_list()[0].get_name() == "RuX"
        mol.set_atom_residue(0, 1)
        assert mol.get_atom_list()[0].get_residue() == 1

    def test_molecule_set_get(self):
        mol = pms.Molecule()

        mol.set_name("test_mol")
        mol.set_short("TMOL")
        mol.set_box([1, 1, 1])
        mol.set_charge(1.5)
        mol.set_masses([1, 2, 3])

        assert mol.get_name() == "test_mol"
        assert mol.get_short() == "TMOL"
        assert mol.get_box() == [1, 1, 1]
        assert mol.get_num() == 0
        assert mol.get_charge() == 1.5
        assert mol.get_masses() == [1, 2, 3]
        assert mol.get_mass() == 6

    def test_molecule_representation(self):
        mol = pms.Molecule()
        mol.add("H", [0.0, 0.1, 0.2], name="HO1")

        assert mol.__str__() == "   Residue Name Type    x    y    z\n0        0  HO1    H  0.0  0.1  0.2"


    ###########
    # Generic #
    ###########
    def test_generic(self):
        assert [round(x, 4) for x in pms.gen.alkane(10, "decane", "DEC").pos(5)] == [0.0472, 0.1028, 0.7170]
        assert [round(x, 4) for x in pms.gen.alkane(1, "methane", "MET").pos(0)] == [0.0514, 0.0890, 0.0363]
        assert [round(x, 4) for x in pms.gen.alcohol(10, "decanol", "DCOL").pos(5)] == [0.0363, 0.1028, 0.7170]
        assert [round(x, 4) for x in pms.gen.alcohol(1, "methanol", "MEOL").pos(0)] == [0.0715, 0.0890, 0.0363]
        assert [round(x, 4) for x in pms.gen.ketone(10, 5, "decanone", "DCON").pos(5)] == [0.0472, 0.1028, 0.7170]
        assert [round(x, 4) for x in  pms.gen.tms(separation=30).pos(5)] == [0.0273, 0.0472, 0.4525]
        assert [round(x, 4) for x in  pms.gen.tms(is_si=False).pos(5)] == [0.0273, 0.0472, 0.4976]
        assert [round(x, 4) for x in  pms.gen.silanol().pos(0)] == [0.000, 0.000, 0.000]

        print()
        with pytest.raises(ValueError, match="too small for ketones"):
            pms.gen.ketone(2, 0)


    #########
    # Store #
    #########
    def test_hybrid36_helpers(self):
        reference_vectors = (
            (4, 9999, "9999"),
            (4, 10000, "A000"),
            (4, 10035, "A00Z"),
            (4, 10036, "A010"),
            (4, 10000 + (26 * (36 ** 3)), "a000"),
            (5, 99999, "99999"),
            (5, 100000, "A0000"),
            (5, 100035, "A000Z"),
            (5, 100036, "A0010"),
            (5, 100000 + (26 * (36 ** 4)), "a0000"),
        )

        for width, value, token in reference_vectors:
            assert store_mod._encode_hybrid36(width, value) == token
            assert store_mod._decode_hybrid36(width, token) == value

        with pytest.raises(ValueError, match="exceeds the hybrid-36 range"):
            store_mod._encode_hybrid36(4, store_mod._hybrid36_max_value(4) + 1)

    def test_pdb_writer_uses_hybrid36_for_overflowed_ids(self):
        mol = pms.Molecule("overflow_writer", "OVF")
        mol.add("C", [0.0, 0.0, 0.0], name="C1")
        mol.add("C", [0.1, 0.0, 0.0], name="C2")

        store = pms.Store(mol, "output")
        atom_records = [
            store_mod._StructureAtomRecord(
                serial=99999,
                pdb_serial_token=store_mod._encode_hybrid36(5, 99999),
                molecule_index=0,
                local_atom_index=0,
                residue_name="TEPS",
                pdb_residue_name="TEP",
                residue_id=9999,
                pdb_residue_id_token=store_mod._encode_hybrid36(4, 9999),
                atom_name="C1",
                atom_type="C",
                position=(0.0, 0.0, 0.0),
                source_id=None,
                pdb_chain_id="A",
                cif_entity_id="1",
                cif_asym_id="A9999",
                cif_label_seq_id="1",
            ),
            store_mod._StructureAtomRecord(
                serial=100000,
                pdb_serial_token=store_mod._encode_hybrid36(5, 100000),
                molecule_index=0,
                local_atom_index=1,
                residue_name="TEPS",
                pdb_residue_name="TEP",
                residue_id=10000,
                pdb_residue_id_token=store_mod._encode_hybrid36(4, 10000),
                atom_name="C2",
                atom_type="C",
                position=(0.1, 0.0, 0.0),
                source_id=None,
                pdb_chain_id="A",
                cif_entity_id="1",
                cif_asym_id="A10000",
                cif_label_seq_id="1",
            ),
        ]
        graph = pms.AssembledStructureGraph.from_bonds(
            (99999, 100000),
            [pms.GraphBond(99999, 100000, "ligand_explicit")],
        )
        store._structure_export_cache[True] = store_mod._StructureExportCache(
            atom_records=atom_records,
            molecule_serials=[[99999, 100000]],
            residue_alias_records=[
                store_mod._PdbResidueAliasRecord(full_name="TEPS", pdb_name="TEP")
            ],
            entity_records=[
                store_mod._CifEntityRecord(entity_id="1", residue_name="TEPS")
            ],
            struct_asym_records=[
                store_mod._CifStructAsymRecord(
                    asym_id="A9999",
                    entity_id="1",
                    residue_id=9999,
                    residue_name="TEPS",
                ),
                store_mod._CifStructAsymRecord(
                    asym_id="A10000",
                    entity_id="1",
                    residue_id=10000,
                    residue_name="TEPS",
                ),
            ],
            graph=graph,
        )

        store.pdb("store_hybrid36_overflow.pdb", use_atom_names=True)

        with open("output/store_hybrid36_overflow.pdb", "r") as file_in:
            pdb_lines = [
                line.rstrip("\n")
                for line in file_in
                if line.startswith(("HETATM", "CONECT"))
            ]

        assert pdb_lines[0][6:11] == "99999"
        assert pdb_lines[1][6:11] == "A0000"
        assert pdb_lines[0][22:26] == "9999"
        assert pdb_lines[1][22:26] == "A000"
        assert store_mod._decode_hybrid36(5, pdb_lines[1][6:11]) == 100000
        assert store_mod._decode_hybrid36(4, pdb_lines[1][22:26]) == 10000
        assert pdb_lines[2].startswith("CONECT")
        assert "A0000" in pdb_lines[2]

    def test_store(self):
        mol = pms.Molecule(inp="data/benzene.gro")

        mol.set_atom_residue(1, 1)

        pms.Store(mol, "output").job("store_job", "store_master.job")
        pms.Store(mol, "output").obj("store_obj.obj")
        pms.Store(mol, "output").gro("store_gro.gro", True)
        pms.Store(mol, "output").pdb("store_pdb.pdb", True)
        pms.Store(mol, "output").cif("store_cif.cif", True)
        pms.Store(mol, "output").xyz("store_xyz.xyz")
        pms.Store(mol, "output").lmp("store_lmp.lmp")
        pms.Store(mol, "output").grid("store_grid.itp")

        with open("output/store_cif.cif", "r") as file_in:
            cif_text = file_in.read()
        assert "_atom_site.Cartn_x" in cif_text
        assert "_struct_conn.id" in cif_text

        with open("output/store_pdb.pdb", "r") as file_in:
            pdb_text = file_in.read()
        assert "CONECT" in pdb_text

        graph = pms.Store(mol, "output").assembled_graph(use_atom_names=True)
        assert isinstance(graph, pms.AssembledStructureGraph)
        assert len(graph.bonds) == 12
        assert all(bond.provenance == "ligand_inferred" for bond in graph.bonds)
        assert len(graph.angles) > 0

        report = pms.Store(mol, "output").validate_connectivity(use_atom_names=True)
        assert isinstance(report, pms.ConnectivityValidationReport)
        assert report.is_valid

        print()
        with pytest.raises(TypeError, match="Unsupported input type"):
            pms.Store({})
        with pytest.raises(TypeError, match="Unsupported input type for topology creation"):
            pms.Store(mol).top()

    def test_connectivity_validation_reports_invalid_local_valence(self):
        mol = pms.Molecule("invalid_valence", "IVL")
        mol.add("O", [0.0, 0.0, 0.0], name="O1")
        mol.add("H", 0, r=0.098, name="H1")
        mol.add("H", 0, r=0.098, theta=120, name="H2")
        mol.add("H", 0, r=0.098, theta=240, name="H3")

        store = pms.Store(mol, "output")
        report = store.validate_connectivity(use_atom_names=True)

        assert not (report.is_valid)
        assert report.error_count > 0
        assert any(finding.code == "unexpected_degree" for finding in report.findings)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            store.gro("invalid_valence_warn.gro", use_atom_names=True, validate_connectivity="warn")
        assert any("Connectivity validation found" in str(warning.message) for warning in caught)

        with pytest.raises(ValueError, match="Connectivity validation found"):
            store.gro("invalid_valence_strict.gro", use_atom_names=True, validate_connectivity="strict")

    def test_connectivity_validation_allows_stretched_but_element_sane_bonds(self):
        mol = pms.Molecule("stretched_silica_fragment", "SSF")
        mol.add("Si", [0.0, 0.0, 0.0], name="SI1")
        mol.add("O", [0.32, 0.0, 0.0], name="O1")
        mol.add("H", [0.50, 0.0, 0.0], name="H1")
        mol.add_bond(0, 1)
        mol.add_bond(1, 2)

        report = pms.Store(mol, "output").validate_connectivity(use_atom_names=True)

        assert report.is_valid
        assert not any(
            finding.code in {"invalid_silica_bond", "hydrogen_degree", "unexpected_degree"}
            for finding in report.findings
        )

    def test_structure_writers_normalize_slx_to_exported_om(self):
        mol = pms.Molecule("siloxane_bridge_probe", "SLX")
        mol.set_box([1.0, 1.0, 1.0])
        mol.add("O", [0.0, 0.0, 0.0], name="OM1")

        store = pms.Store(mol, "output")
        store.gro(
            "siloxane_bridge_probe.gro",
            use_atom_names=True,
            validate_connectivity="off",
        )
        store.pdb(
            "siloxane_bridge_probe.pdb",
            use_atom_names=True,
            validate_connectivity="off",
        )
        store.cif(
            "siloxane_bridge_probe.cif",
            use_atom_names=True,
            validate_connectivity="off",
        )

        with open("output/siloxane_bridge_probe.gro", "r") as file_in:
            gro_text = file_in.read()
        with open("output/siloxane_bridge_probe.pdb", "r") as file_in:
            pdb_text = file_in.read()
        with open("output/siloxane_bridge_probe.cif", "r") as file_in:
            cif_text = file_in.read()

        assert "SLX" not in gro_text
        assert "SLX" not in pdb_text
        assert "SLX" not in cif_text
        assert "OM" in gro_text
        assert " OM " in pdb_text
        assert " OM " in cif_text

    def test_connectivity_validation_normalizes_siloxane_bridges_to_framework_oxygen_rules(self):
        mol = pms.Molecule("invalid_siloxane_bridge", "SLX")
        mol.add("O", [0.0, 0.0, 0.0], name="OM1")
        mol.add("Si", [0.16, 0.0, 0.0], name="SI1")
        mol.add("H", [-0.10, 0.0, 0.0], name="H1")
        mol.add_bond(0, 1)
        mol.add_bond(0, 2)

        report = pms.Store(mol, "output").validate_connectivity(use_atom_names=True)

        assert not report.is_valid
        assert any(
            finding.code == "framework_oxygen_environment"
            for finding in report.findings
        )

    def test_pdb_uses_hybrid36_when_fixed_width_limits_are_exceeded(self):
        atom = pms.Molecule("single_atom", "SIN")
        atom.add("H", [0.0, 0.0, 0.0], name="H1")
        store = pms.Store(atom, "output")
        store._mols = [atom] * 100000

        cache = store._collect_structure_records(use_atom_names=True)
        records = cache.atom_records
        store._validate_pdb_hybrid36_limits(records)

        assert records[9998].pdb_residue_id_token == "9999"
        assert records[9999].pdb_residue_id_token == "A000"
        assert store_mod._decode_hybrid36(4, records[-1].pdb_residue_id_token) == records[-1].residue_id

    def test_steric_grid_matches_bruteforce_clearance(self):
        block = pms.Molecule("steric_block", "SBL")
        block.set_box([1.0, 1.0, 1.0])
        block.add("Si", [0.10, 0.10, 0.10], name="SI1")
        block.add("O", [0.26, 0.10, 0.10], name="OM1")
        block.add("Si", [0.42, 0.10, 0.10], name="SI2")

        pore = pms.Pore(block, pms.Matrix([[0, [1]], [2, [1]]]))

        attached = pms.Molecule("attached_probe", "ATP")
        attached.set_box([1.0, 1.0, 1.0])
        attached.add("C", [0.30, 0.30, 0.10], name="C1")
        attached.add("H", [0.40, 0.30, 0.10], name="H1")
        pore._mol_dict["in"]["ATP"] = [attached]

        candidate = pms.Molecule("candidate_probe", "CDP")
        candidate.set_box([1.0, 1.0, 1.0])
        candidate.add("Si", [0.22, 0.23, 0.10], name="SI1")
        candidate.add("C", [0.34, 0.23, 0.10], name="C1")

        ignored = {0}
        brute_force = pore._placement_clearance(candidate, ignored_block_atoms=ignored)
        grid = pore._build_steric_grid()
        local_grid = pore._placement_clearance(
            candidate,
            steric_grid=grid,
            ignored_block_atoms=ignored,
        )

        assert local_grid == pytest.approx(brute_force, abs=1e-12)

    def test_attachment_clearance_scale_can_relax_a_crowded_pose(self):
        block = pms.Molecule("clearance_block", "CLB")
        block.set_box([1.0, 1.0, 1.0])
        block.add("O", [0.00, 0.00, 0.00], name="O1")
        pore = pms.Pore(block, pms.Matrix([[0, []]]))

        candidate = pms.Molecule("clearance_candidate", "CLC")
        candidate.set_box([1.0, 1.0, 1.0])
        candidate.add("C", [0.10, 0.00, 0.00], name="C1")

        strict_clearance = pore._placement_clearance(
            candidate,
            steric_clearance_scale=0.85,
        )
        relaxed_clearance = pore._placement_clearance(
            candidate,
            steric_clearance_scale=0.60,
        )

        assert strict_clearance < 0
        assert relaxed_clearance > 0

    def test_optimize_attachment_pose_matches_bruteforce_rotation_scan(self):
        block = pms.Molecule("rotation_block", "RBL")
        block.set_box([1.0, 1.0, 1.0])
        block.add("O", [0.20, 0.00, 0.00], name="O1")
        pore = pms.Pore(block, pms.Matrix([[0, []]]))

        candidate = pms.Molecule("rotation_candidate", "RCD")
        candidate.set_box([1.0, 1.0, 1.0])
        candidate.add("Si", [0.00, 0.00, 0.00], name="SI1")
        candidate.add("C", [0.10, 0.00, 0.00], name="C1")

        grid = pore._build_steric_grid()
        optimized = pore._optimize_attachment_pose(
            candidate,
            0,
            [0.0, 0.0, 1.0],
            set(),
            grid,
            True,
            90,
            0.85,
        )

        brute_force_best = None
        brute_force_clearance = float("-inf")
        rotation_axis = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        for angle in pore._rotation_angles(90):
            brute_force_candidate = copy.deepcopy(candidate)
            if angle != 0:
                brute_force_candidate.rotate(rotation_axis, angle)
            clearance = pore._placement_clearance(
                brute_force_candidate,
                steric_grid=grid,
                steric_clearance_scale=0.85,
            )
            if clearance > brute_force_clearance:
                brute_force_clearance = clearance
                brute_force_best = brute_force_candidate

        assert optimized is not None
        assert pore._placement_clearance(
            optimized,
            steric_grid=grid,
            steric_clearance_scale=0.85,
        ) == pytest.approx(brute_force_clearance, abs=1e-12)
        assert optimized.pos(1) == pytest.approx(brute_force_best.pos(1), abs=1e-12)


    ###########
    # Pattern #
    ###########
    def test_pattern_beta_cristobalit(self):
        # Initialize
        beta_cristobalit = pms.BetaCristobalit()

        # Pattern and output
        pattern = beta_cristobalit.pattern()
        pattern.set_name("pattern_beta_cbt_minimal")
        assert pattern.get_num() == 36
        pms.Store(pattern, "output").gro()

        # Generation and Orientation
        beta_cristobalit = pms.BetaCristobalit()
        beta_cristobalit.generate([2, 2, 2], "x")
        beta_cristobalit.get_block().set_name("pattern_beta_cbt_x")
        assert beta_cristobalit.get_size() == [2.480, 1.754, 2.024]
        assert [round(x, 3) for x in beta_cristobalit.get_block().get_box()] == [2.480, 1.754, 2.024]
        pms.Store(beta_cristobalit.get_block(), "output").gro()

        beta_cristobalit = pms.BetaCristobalit()
        beta_cristobalit.generate([2, 2, 2], "y")
        beta_cristobalit.get_block().set_name("pattern_beta_cbt_y")
        assert beta_cristobalit.get_size() == [2.024, 2.480, 1.754]
        assert [round(x, 3) for x in beta_cristobalit.get_block().get_box()] == [2.024, 2.480, 1.754]
        pms.Store(beta_cristobalit.get_block(), "output").gro()

        beta_cristobalit = pms.BetaCristobalit()
        beta_cristobalit.generate([2, 2, 2], "z")
        beta_cristobalit.get_block().set_name("pattern_beta_cbt_z")
        assert beta_cristobalit.get_size() == [2.024, 1.754, 2.480]
        assert [round(x, 3) for x in beta_cristobalit.get_block().get_box()] == [2.024, 1.754, 2.480]
        pms.Store(beta_cristobalit.get_block(), "output").gro()
        pms.Store(beta_cristobalit.get_block(), "output").lmp()

        # Misc
        beta_cristobalit = pms.BetaCristobalit()
        beta_cristobalit.generate([2, 2, 2], "z")
        beta_cristobalit.get_block().set_name("DOTA")

        # Overlap and output
        assert beta_cristobalit.get_block().get_num() == 576
        assert beta_cristobalit.get_block().overlap() == {}

        # Getter
        assert beta_cristobalit.get_repeat() == [0.506, 0.877, 1.240]
        assert beta_cristobalit.get_gap() == [0.126, 0.073, 0.155]
        assert beta_cristobalit.get_orient() == "z"
        assert beta_cristobalit.get_block().get_name() == "DOTA"

    def test_alpha_cristobalit(self):
        # Initialize
        alpha_cristobalit = pms.AlphaCristobalit()

        # Pattern and output
        pattern = alpha_cristobalit.pattern()
        pattern.set_name("pattern_alpha_cbt_minimal")
        assert pattern.get_num() == 12
        pms.Store(pattern, "output").gro()

        # Generation and Orientation
        alpha_cristobalit = pms.AlphaCristobalit()
        alpha_cristobalit.generate([2, 2, 2], "x")
        alpha_cristobalit.get_block().set_name("pattern_alpha_cbt_x")
        assert alpha_cristobalit.get_size() == [2.0844, 1.9912, 1.9912]
        assert [round(x, 3) for x in alpha_cristobalit.get_block().get_box()] == [2.084, 1.991, 1.991]
        pms.Store(alpha_cristobalit.get_block(), "output").gro()

        alpha_cristobalit = pms.AlphaCristobalit()
        alpha_cristobalit.generate([2, 2, 2], "y")
        alpha_cristobalit.get_block().set_name("pattern_alpha_cbt_y")
        assert alpha_cristobalit.get_size() == [1.9912, 2.0844, 1.9912]
        assert [round(x, 3) for x in alpha_cristobalit.get_block().get_box()] == [1.991, 2.084, 1.991]
        pms.Store(alpha_cristobalit.get_block(), "output").gro()

        alpha_cristobalit = pms.AlphaCristobalit()
        alpha_cristobalit.generate([2, 2, 2], "z")
        alpha_cristobalit.get_block().set_name("pattern_alpha_cbt_z")
        assert alpha_cristobalit.get_size() == [1.9912, 1.9912, 2.0844]
        assert [round(x, 3) for x in alpha_cristobalit.get_block().get_box()] == [1.991, 1.991, 2.084]
        pms.Store(alpha_cristobalit.get_block(), "output").gro()
        pms.Store(alpha_cristobalit.get_block(), "output").lmp()

        # Misc
        alpha_cristobalit = pms.AlphaCristobalit()
        alpha_cristobalit.generate([2, 2, 2], "z")
        alpha_cristobalit.get_block().set_name("DOTA")

        # Overlap and output
        assert alpha_cristobalit.get_block().get_num() == 576
        assert alpha_cristobalit.get_block().overlap() == {}

        # Getter
        assert alpha_cristobalit.get_repeat() == [0.4978, 0.4978, 0.6948]
        assert alpha_cristobalit.get_orient() == "z"
        assert alpha_cristobalit.get_block().get_name() == "DOTA"


    ########
    # Dice #
    ########
    def test_dice(self):
        block = pms.BetaCristobalit().generate([2, 2, 2], "z")
        block.set_name("dice")
        pms.Store(block, "output").gro()
        dice = pms.Dice(block, 0.4, True)

        # Splitting and filling
        assert len(dice.get_origin()) == 120
        assert dice.get_origin()[(1, 1, 1)] == [0.4, 0.4, 0.4]
        assert dice.get_pointer()[(1, 1, 1)] == [14, 51, 52, 64, 65, 67]

        # Iterator
        assert dice._right((1, 1, 1)) == (2, 1, 1)
        assert dice._left((1, 1, 1)) == (0, 1, 1)
        assert dice._top((1, 1, 1)) == (1, 2, 1)
        assert dice._bot((1, 1, 1)) == (1, 0, 1)
        assert dice._front((1, 1, 1)) == (1, 1, 2)
        assert dice._back((1, 1, 1)) == (1, 1, 0)
        assert len(dice.neighbor((1, 1, 1))) == 27
        assert len(dice.neighbor((1, 1, 1), False)) == 26

        # Search
        assert dice.find([(1, 1, 1)], ["Si", "O"], [0.155-0.005, 0.155+0.005]) == [[51, [46, 14, 52, 65]], [64, [26, 63, 65, 67]]]
        assert dice.find([(1, 1, 1)], ["O", "Si"], [0.155-0.005, 0.155+0.005]) == [[14, [51, 13]], [52, [51, 49]], [65, [51, 64]], [67, [64, 69]]]
        assert dice.find([(0, 0, 0)], ["Si", "O"], [0.155-0.005, 0.155+0.005]) == [[3, [4, 9, 2, 174]], [5, [306, 110, 4, 6]]]
        assert dice.find([(0, 0, 0)], ["O", "Si"], [0.155-0.005, 0.155+0.005]) == [[4, [3, 5]], [6, [7, 5]], [9, [3, 11]]]

        # Full search
        assert len(dice.find(None, ["Si", "O"], [0.155-0.005, 0.155+0.005])) == 192
        assert len(dice.find(None, ["O", "Si"], [0.155-0.005, 0.155+0.005])) == 384

        # Setter Getter
        dice.set_pbc(True)
        assert dice.get_count() == [5, 4, 6]
        assert dice.get_size() == 0.4
        assert dice.get_mol().get_name() == "dice"


    ##########
    # Matrix #
    ##########
    def test_matrix(self):
        orient = "z"
        block = pms.BetaCristobalit().generate([1, 1, 1], orient)
        block.set_name("matrix")
        pms.Store(block, "output").gro()
        dice = pms.Dice(block, 0.2, True)
        bonds = dice.find(None, ["Si", "O"], [0.155-1e-2, 0.155+1e-2])

        matrix = pms.Matrix(bonds)
        connect = matrix.get_matrix()
        matrix.split(0, 17)
        assert connect[0]["atoms"] == [30, 8, 1]
        assert connect[17]["atoms"] == [19]
        matrix.strip(0)
        assert connect[0]["atoms"] == []
        assert connect[1]["atoms"] == [43]
        assert connect[8]["atoms"] == [7]
        assert connect[30]["atoms"] == [3]
        assert matrix.bound(0) == [0]
        assert matrix.bound(1, "lt") == [0]
        assert matrix.bound(4, "gt") == []
        matrix.add(0, 17)
        assert connect[0]["atoms"] == [17]
        assert connect[17]["atoms"] == [19, 0]

        print()
        with pytest.raises(ValueError, match="Wrong logic statement"):
            matrix.bound(4, "test")


    #########
    # Shape #
    #########
    def test_shape_cylinder(self):
        block = pms.BetaCristobalit().generate([6, 6, 6], "z")
        block.set_name("shape_cylinder")
        dice = pms.Dice(block, 0.4, True)
        matrix = pms.Matrix(dice.find(None, ["Si", "O"], [0.155-1e-2, 0.155+1e-2]))
        centroid = block.centroid()
        central = pms.geom.unit(pms.geom.rotate([0, 0, 1], [1, 0, 0], 45, True))

        cylinder = pms.Cylinder(
            pms.CylinderConfig(
                centroid=tuple(centroid),
                central=tuple(central),
                length=3,
                diameter=4,
            )
        )
        assert isinstance(cylinder.get_config(), pms.CylinderConfig)

        # Properties
        assert round(cylinder.volume(), 4) == 37.6991
        assert round(cylinder.surface(), 4) == 37.6991

        # Test vector
        vec = [3.6086, 4.4076, 0.2065]

        # Surface
        assert [round(x[0][20], 4) for x in cylinder.surf(num=100)] == vec
        assert [round(x[0][20], 4) for x in cylinder.rim(0, num=100)] == vec

        # Normal
        assert [round(x, 4) for x in cylinder.convert([0, 0, 0], False)] == [3.0147, 3.0572, 1.5569]
        assert [round(x, 4) for x in cylinder.normal(vec)] == [0.5939, 2.9704, 0.0000]

        # Positioning
        del_list = [atom_id for atom_id, atom in enumerate(block.get_atom_list()) if cylinder.is_in(atom.get_pos())]
        matrix.strip(del_list)
        block.delete(matrix.bound(0))
        assert block.get_num() == 12650

        # Store molecule
        pms.Store(block, "output").gro()

        # Plot surface
        plt.figure()
        cylinder.plot(vec=[3.17290646, 4.50630614, 0.22183271])
        # plt.show()

    def test_shape_sphere(self):
        block = pms.BetaCristobalit().generate([6, 6, 6], "z")
        block.set_name("shape_sphere")
        dice = pms.Dice(block, 0.4, True)
        matrix = pms.Matrix(dice.find(None, ["Si", "O"], [0.155-1e-2, 0.155+1e-2]))
        centroid = block.centroid()
        central = pms.geom.unit(pms.geom.rotate([0, 0, 1], [1, 0, 0], 0, True))

        sphere = pms.Sphere(
            pms.SphereConfig(
                centroid=tuple(centroid),
                central=tuple(central),
                diameter=4,
            )
        )
        assert isinstance(sphere.get_config(), pms.SphereConfig)

        # Properties
        assert round(sphere.volume(), 4) == 33.5103
        assert round(sphere.surface(), 4) == 50.2655

        # Surface
        assert [round(x[0][20], 4) for x in sphere.surf(num=100)] == [4.2006, 3.0572, 4.6675]
        assert [round(x[0][20], 4) for x in sphere.rim(0, num=100)] == [4.9245, 3.0572, 3.6508]

        # Normal
        assert [round(x, 4) for x in sphere.convert([0, 0, 0], False)] == [3.0147, 3.0572, 3.0569]
        assert [round(x, 4) for x in sphere.normal([4.2006, 3.0572, 4.6675])] == [1.4063, 0.0000, 1.9099]

        # Positioning
        del_list = [atom_id for atom_id, atom in enumerate(block.get_atom_list()) if sphere.is_in(atom.get_pos())]
        matrix.strip(del_list)
        block.delete(matrix.bound(0))
        assert block.get_num() == 12934

        # Store molecule
        pms.Store(block, "output").gro()

        # Plot surface
        sphere.plot(inp=3.14, vec=[1.08001048, 3.09687610, 1.72960828])
        # plt.show()

    def test_shape_cuboid(self):
        block = pms.BetaCristobalit().generate([6, 6, 6], "z")
        block.set_name("shape_cuboid")
        dice = pms.Dice(block, 0.4, True)
        matrix = pms.Matrix(dice.find(None, ["Si", "O"], [0.155-1e-2, 0.155+1e-2]))
        centroid = block.centroid()
        central = pms.geom.unit(pms.geom.rotate([0, 0, 1], [1, 0, 0], 0, True))

        cuboid = pms.Cuboid(
            pms.CuboidConfig(
                centroid=tuple(centroid),
                central=tuple(central),
                length=10,
                width=6,
                height=4,
            )
        )
        assert isinstance(cuboid.get_config(), pms.CuboidConfig)

        # Properties
        assert round(cuboid.volume(), 4) == 240
        assert round(cuboid.surface(), 4) == 248

        # Normal
        assert [round(x, 4) for x in cuboid.convert([0, 0, 0], False)] == [0.0147, 1.0572, -1.9431]
        assert [round(x, 4) for x in cuboid.normal([4.2636, 3.0937, 4.745])] == [0, 1, 0]

        # Positioning
        del_list = [atom_id for atom_id, atom in enumerate(block.get_atom_list()) if cuboid.is_in(atom.get_pos())]
        matrix.strip(del_list)
        block.delete(matrix.bound(0))
        assert block.get_num() == 5160

        # Store molecule
        pms.Store(block, "output").gro()

        # Plot surface
        cuboid.plot()
        # plt.show()

    def test_shape_cone(self):
        block = pms.BetaCristobalit().generate([6, 6, 6], "z")
        block.set_name("shape_cone")
        dice = pms.Dice(block, 0.4, True)
        matrix = pms.Matrix(dice.find(None, ["Si", "O"], [0.155-1e-2, 0.155+1e-2]))
        centroid = block.centroid()
        central = pms.geom.unit(pms.geom.rotate([0, 0, 1], [1, 0, 0], 45, True))

        cone = pms.Cone(
            pms.ConeConfig(
                centroid=tuple(centroid),
                central=tuple(central),
                length=6,
                diameter_1=4,
                diameter_2=1,
            )
        )
        assert isinstance(cone.get_config(), pms.ConeConfig)

        # Properties
        assert round(cone.volume(), 4) == 32.9867
        assert round(cone.surface(), 4) == 48.5742

        # Test vector
        vec = [3.6977, 4.6102, -1.4961]

        # Surface
        assert [round(x[0][20], 4) for x in cone.surf(num=100)] == vec
        assert [round(x[0][20], 4) for x in cone.rim(0, num=100)] == vec

        # Normal
        assert [round(x, 4) for x in cone.convert([0, 0, 0], False)] == [3.0147, 3.0572, 0.0569]
        assert [round(x, 4) for x in cone.normal(vec)] == [0.3182, 2.0114, 0.6109]

        # Positioning
        del_list = [atom_id for atom_id, atom in enumerate(block.get_atom_list()) if cone.is_in(atom.get_pos())]
        matrix.strip(del_list)
        block.delete(matrix.bound(0))
        assert block.get_num() == 12486

        # Store molecule
        pms.Store(block, "output").gro()

        # Plot surface
        plt.figure()
        cone.plot(vec=[3.17290646, 4.50630614, 0.22183271])
        # plt.show()


    ########
    # Pore #
    ########
    def test_pore(self):
        # No exterior surface
        orient = "z"
        pattern = pms.BetaCristobalit()
        pattern.generate([6, 6, 6], orient)

        block = pattern.get_block()
        block.set_name("pore_cylinder_block")

        dice = pms.Dice(block, 0.4, True)
        bond_list = dice.find(None, ["Si", "O"], [0.155-1e-2, 0.155+1e-2])
        matrix = pms.Matrix(bond_list)

        pore = pms.Pore(block, matrix)

        centroid = block.centroid()
        central = pms.geom.unit(pms.geom.rotate([0, 0, 1], [1, 0, 0], 0, True))
        cylinder = pms.Cylinder(
            pms.CylinderConfig(
                centroid=tuple(centroid),
                central=tuple(central),
                length=6,
                diameter=4,
            )
        )
        del_list = [atom_id for atom_id, atom in enumerate(block.get_atom_list()) if cylinder.is_in(atom.get_pos())]
        matrix.strip(del_list)

        pore.prepare()
        pore.sites()
        assert len(pore.get_sites()) == 455

        block.delete(matrix.bound(0))
        pms.Store(block, "output").gro("pore_no_ex.gro")

        # With exterior surface
        orient = "z"
        pattern = pms.BetaCristobalit()
        pattern.generate([6, 6, 6], orient)

        block = pattern.get_block()
        block.set_name("pore_cylinder_block")

        dice = pms.Dice(block, 0.4, True)
        bond_list = dice.find(None, ["Si", "O"], [0.155-1e-2, 0.155+1e-2])
        matrix = pms.Matrix(bond_list)

        pore = pms.Pore(block, matrix)
        pore.exterior()

        centroid = block.centroid()
        central = pms.geom.unit(pms.geom.rotate([0, 0, 1], [1, 0, 0], 0, True))
        cylinder = pms.Cylinder(
            pms.CylinderConfig(
                centroid=tuple(centroid),
                central=tuple(central),
                length=6,
                diameter=4,
            )
        )
        del_list = [atom_id for atom_id, atom in enumerate(block.get_atom_list()) if cylinder.is_in(atom.get_pos())]
        matrix.strip(del_list)

        pore.prepare()
        pore.amorph()
        assert len(matrix.bound(1)) == 710
        pore.sites()
        site_list = pore.get_sites()
        assert isinstance(next(iter(site_list.values())), pms.BindingSite)
        site_in = [site_key for site_key, site_val in site_list.items() if site_val.site_type == "in"]
        site_ex = [site_key for site_key, site_val in site_list.items() if site_val.site_type == "ex"]
        assert len(site_in) == 432
        assert len(site_ex) == 201

        si_pos_in = [block.pos(site_key) for site_key, site_val in site_list.items() if site_val.site_type == "in"]
        si_pos_ex = [block.pos(site_key) for site_key, site_val in site_list.items() if site_val.site_type == "ex"]

        if si_pos_in:
            temp_mol = pms.Molecule()
            for pos in si_pos_in:
                temp_mol.add("Si", pos)
            pms.Store(temp_mol).gro("output/pore_cylinder_si_in.gro")

        if si_pos_ex:
            temp_mol = pms.Molecule()
            for pos in si_pos_ex:
                temp_mol.add("Si", pos)
            pms.Store(temp_mol).gro("output/pore_cylinder_si_ex.gro")

        # Objectify grid
        non_grid = matrix.bound(1)+list(site_list.keys())
        bonded = matrix.bound(0, "gt")
        grid_atoms = [atom for atom in bonded if not atom in non_grid]
        mol_obj = pore.objectify(grid_atoms)
        assert len(mol_obj) == 8279
        pms.Store(pms.Molecule(name="pore_cylinder_grid", inp=mol_obj), "output").gro(use_atom_names=True)

        # Attachment
        mol = pms.gen.tms()

        def normal(pos):
            return [0, 0, -1] if pos[2] < centroid[2] else [0, 0, 1]

        for site in site_in:
            site_list[site].normal = cylinder.normal
        for site in site_ex:
            site_list[site].normal = normal

        ## Siloxane
        mols_siloxane = pore.siloxane(site_in, 100)
        assert "SLX" in pore.get_mol_dict()
        site_in = [site_key for site_key, site_val in site_list.items() if site_val.site_type == "in"]

        ## Normal
        mols_in = pore.attach(mol, 0, [0, 1], site_in, 100, site_type="in")
        mols_ex = pore.attach(mol, 0, [0, 1], site_ex, 20, site_type="ex")

        ## Filling
        mols_in_fill = pore.fill_sites(site_in, site_type="in")
        mols_ex_fill = pore.fill_sites(site_ex, site_type="ex")

        ## Storage
        pms.Store(pms.Molecule(name="pore_cylinder_siloxane", inp=mols_siloxane), "output").gro()
        pms.Store(pms.Molecule(name="pore_cylinder_in", inp=mols_in), "output").gro()
        pms.Store(pms.Molecule(name="pore_cylinder_ex", inp=mols_ex), "output").gro()
        pms.Store(pms.Molecule(name="pore_cylinder_in_fill", inp=mols_in_fill), "output").gro()
        pms.Store(pms.Molecule(name="pore_cylinder_ex_fill", inp=mols_ex_fill), "output").gro()

        # Delete atoms
        block.delete(matrix.bound(0))
        pms.Store(block, "output").gro()

        # Set reservoir
        pore.reservoir(5)
        assert [round(x) for x in pore.get_box()] == [6, 6, 17]

        # Output
        pore.set_name("pore_cylinder_full")
        pms.Store(pore, "output").gro(use_atom_names=True)

        pore.set_name("pore_cylinder_full_sort")
        sort_list = ["OM", "SI", "SLX", "SL", "SLG", "TMS", "TMSG"]
        store = pms.Store(pore, "output", sort_list=sort_list)
        store.gro(use_atom_names=True)
        store.pdb(use_atom_names=True)
        store.cif(use_atom_names=True)
        store.top()
        store.grid("pore_cylinder_full_sort_grid.itp")

        with open("output/pore_cylinder_full_sort.gro", "r") as file_in:
            gro_text = file_in.read()
        with open("output/pore_cylinder_full_sort.pdb", "r") as file_in:
            pdb_text = file_in.read()
        with open("output/pore_cylinder_full_sort.cif", "r") as file_in:
            cif_text = file_in.read()
        with open("output/pore_cylinder_full_sort.top", "r") as file_in:
            top_text = file_in.read()
        with open("output/pore_cylinder_full_sort_grid.itp", "r") as file_in:
            grid_text = file_in.read()

        molecules_lines = [
            line.split()
            for line in top_text.splitlines()
            if line and not line.startswith("[") and not line.startswith("#") and " " in line
        ]
        molecules = {
            tokens[0]: int(tokens[1])
            for tokens in molecules_lines
            if len(tokens) == 2 and tokens[0].isalpha() and tokens[1].isdigit()
        }
        expected_om_count = (
            len(pore.get_mol_dict().get("OM", []))
            + len(pore.get_mol_dict().get("SLX", []))
        )

        assert "SLX" not in gro_text
        assert "SLX" not in pdb_text
        assert "SLX" not in cif_text
        assert "SLX" not in grid_text
        assert "OM" in gro_text
        assert " OM " in pdb_text
        assert " OM " in cif_text
        assert "SLX " not in top_text
        assert molecules["OM"] == expected_om_count

        # Store test
        print()
        with pytest.raises(ValueError, match="Sorting list does not contain all keys"):
            pms.Store(pore, "output", sort_list=sort_list[:-1])

        # Error test
        with pytest.raises(ValueError, match="site_type"):
            pore.attach(mol, 0, [0, 1], site_in, 0, cylinder.normal, site_type="DOTA")
        with pytest.raises(ValueError, match="site_type"):
            pore.siloxane(site_in, 0, cylinder.normal, site_type="DOTA")

        # Getter and Setter
        assert pore.get_block().get_name() == "pore_cylinder_block"
        assert len(pore.get_site_dict()) == 3
        assert pore.get_num_in_ex() == 23

    def test_pore_exterior(self):
        # x-axis
        pattern = pms.BetaCristobalit()
        block = pattern.generate([2, 2, 2], "x")
        block.set_name("pattern_beta_cbt_ex_x")
        dice = pms.Dice(block, 0.2, True)
        bonds = dice.find(None, ["Si", "O"], [0.155-1e-2, 0.155+1e-2])
        matrix = pms.Matrix(bonds)
        pore = pms.Pore(block, matrix)
        pore.prepare()
        pore.exterior()
        pore.sites()
        pms.Store(block, "output").gro()

        # y-axis
        pattern = pms.BetaCristobalit()
        block = pattern.generate([2, 2, 2], "y")
        block.set_name("pattern_beta_cbt_ex_y")
        dice = pms.Dice(block, 0.2, True)
        bonds = dice.find(None, ["Si", "O"], [0.155-1e-2, 0.155+1e-2])
        matrix = pms.Matrix(bonds)
        pore = pms.Pore(block, matrix)
        pore.prepare()
        pore.exterior()
        pore.sites()
        pms.Store(block, "output").gro()

        # z-axis
        pattern = pms.BetaCristobalit()
        block = pattern.generate([2, 2, 2], "z")
        block.set_name("pattern_beta_cbt_ex_z")
        dice = pms.Dice(block, 0.2, True)
        bonds = dice.find(None, ["Si", "O"], [0.155-1e-2, 0.155+1e-2])
        matrix = pms.Matrix(bonds)
        pore = pms.Pore(block, matrix)
        pore.prepare()
        pore.exterior()
        pore.sites()
        pms.Store(block, "output").gro()

        # Amorph
        pattern = pms.BetaCristobalit()
        pattern.generate([2, 2, 2], "z")

        pattern._structure = pms.Molecule(inp="data/amorph.gro")
        pattern._size = [2.014, 1.751, 2.468]

        block = pattern.get_block()
        block.set_name("pattern_beta_cbt_ex_amoprh")

        dice = pms.Dice(block, 0.4, True)
        matrix = pms.Matrix(dice.find(None, ["Si", "O"], [0.160-0.02, 0.160+0.02]))

        connect = matrix.get_matrix()
        matrix.split(57790, 2524)

        pore = pms.Pore(block, matrix)
        pore.prepare()
        pore.exterior()
        pore.sites()

        pms.Store(block, "output").gro()

    def test_pore_kit(self):
        unassigned_warning = "Some interior silicon binding sites could not be assigned"

        pore = pms.PoreKit()
        pore.structure(pms.BetaCristobalit().generate([5, 5, 10], "z"))
        pore.build()
        pore.exterior(5, hydro=0.4)
        invalid_shape = pms.ShapeSpec(
            "DOTA",
            pms.Cylinder(
                pms.CylinderConfig(
                    centroid=(3.5, 3.5, 5.0),
                    central=(0.0, 0.0, 1.0),
                    length=10,
                    diameter=1.5,
                )
            ),
        )
        with pytest.raises(ValueError, match="shape type"):
            pore.add_shape(invalid_shape, hydro=0.4)
        assert isinstance(pore.shape_cylinder(2, 10, [3.5, 3.5, 5]), pms.ShapeSpec)
        pore.add_shape(pore.shape_cylinder(2, 10, [3.5, 3.5, 5]), hydro=0.4)
        pore.add_shape(pore.shape_cylinder(2, 10, [1.5, 1.5, 5]), hydro=0.4)
        with pytest.warns(RuntimeWarning, match=unassigned_warning):
            pore.prepare()
        pore.attach(pms.gen.tms(), 0, [0, 1], 100, "in")
        pore.attach(pms.gen.tms(), 0, [0, 1], 20, "ex")
        pore.finalize()
        pore.store("output/kit_parallel/")
        # pore.table()

        pore = pms.PoreKit()
        pore.structure(pms.BetaCristobalit().generate([7, 7, 10], "z"))
        pore.build()
        pore.exterior(5, hydro=0.4)
        pore.add_shape(pore.shape_cylinder(6, 4, [3.5, 3.5, 2]), section=pms.ShapeSection(z=(0, 4)), hydro=0.4)
        pore.add_shape(pore.shape_cone(4.5, 3, 2,  [3.5, 3.5, 5]), section=pms.ShapeSection(z=(4, 6)), hydro=0.4)
        pore.add_shape(pore.shape_cylinder(4, 4, [3.5, 3.5, 8]), section=pms.ShapeSection(z=(6, 10)), hydro=0.4)
        with pytest.warns(RuntimeWarning, match=unassigned_warning):
            pore.prepare()
        pore.attach(pms.gen.tms(), 0, [0, 1], 100, "in")
        pore.attach(pms.gen.tms(), 0, [0, 1], 20, "ex")
        pore.finalize()
        pore.store("output/kit_narrow/")

    def test_porekit_percent_shape_specific(self):
        pore = pms.PoreKit()
        pore.structure(pms.BetaCristobalit().generate([7, 7, 10], "z"))
        pore.build()
        pore.add_shape(
            pore.shape_cylinder(4, 5, [3.5, 3.5, 2.5]),
            section=pms.ShapeSection(z=(0, 5)),
        )
        pore.add_shape(
            pore.shape_cylinder(3, 5, [3.5, 3.5, 7.5]),
            section=pms.ShapeSection(z=(5, 10)),
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", RuntimeWarning)
            pore.prepare()

        site_lookup = pore._pore.get_sites()
        sites_by_shape = pore._pore.sites_sl_shape
        valid_shape_keys = sorted(
            shape_key for shape_key in sites_by_shape
            if shape_key < len(pore.shape())
        )
        assert valid_shape_keys == [0, 1]

        global_geminal_count = sum(
            1
            for site in site_lookup.values()
            if site.site_type == "in" and site.oxygen_count == 2
        )

        selected_shape = None
        for shape_key in valid_shape_keys:
            shape_sites = sites_by_shape[shape_key]
            shape_oh_count = sum(site_lookup[site_id].oxygen_count for site_id in shape_sites)
            buggy_oh_count = len(shape_sites) + global_geminal_count
            for percent in range(10, 101, 5):
                correct_amount = int(percent / 100 * shape_oh_count)
                buggy_amount = int(percent / 100 * buggy_oh_count)
                if (
                    correct_amount > 0
                    and correct_amount <= len(shape_sites)
                    and correct_amount != buggy_amount
                ):
                    selected_shape = (
                        shape_key,
                        percent,
                        correct_amount,
                        buggy_amount,
                    )
                    break
            if selected_shape is not None:
                break

        assert selected_shape is not None
        shape_key, percent, expected_amount, buggy_amount = selected_shape
        marker = pms.gen.tms()
        marker.set_short("TMSP")
        pore.attach(
            marker,
            0,
            [0, 1],
            percent,
            "in",
            inp="percent",
            shape=f"shape_{shape_key}",
            trials=2000,
            is_proxi=False,
        )

        site_dict = pore._pore.get_site_dict()["in"]
        attached_count = len(site_dict.get("TMSP", [])) + len(site_dict.get("TMSPG", []))
        assert attached_count == expected_amount
        assert attached_count != buggy_amount

        table = pore.table()
        shape_label = f"Pore {shape_key + 1}"
        assert list(table.columns) == ["Interior", "Exterior"]
        assert f"Surface chemistry - Before Functionalization ({shape_label})" in table.index
        assert f"Surface chemistry - After Functionalization ({shape_label})" in table.index
        assert any(
                label.startswith(f"    {shape_label} Number of ")
                for label in table.index
            )

    def test_pore_cylinder(self):
        # Empty pore
        pore = pms.PoreCylinder([4, 4, 4], 2, 0)
        pore.finalize()

        # Filled pore
        pore = pms.PoreCylinder([6, 6, 6], 4, 5, [5, 5])

        ## Attachment
        # pore.attach_special(pms.gen.tms(),  0, [0, 1], 5)
        # pore.attach_special(pms.gen.tms(),  0, [0, 1], 3, symmetry="mirror")

        tms2 = pms.gen.tms()
        tms2.set_short("TMS2")

        pore.attach(tms2, 0, [0, 1], 10, "in", trials=10, inp="percent")
        pore.attach(tms2, 0, [0, 1], 1, "in", trials=10, inp="molar")
        pore.attach(tms2, 0, [0, 1], 0.1, "ex", trials=10, inp="molar")

        # Special cases
        print()
        with pytest.raises(ValueError, match="site_type"):
            pore.attach(pms.gen.tms(), 0, [0, 1], 100, site_type="DOTA")
        with pytest.raises(ValueError, match="inp"):
            pore.attach(pms.gen.tms(), 0, [0, 1], 100, "in", inp="DOTA")
        with pytest.raises(ValueError, match="positions"):
            pore.attach(pms.gen.tms(), 0, [0, 1], 100, pos_list=[[1, 3, 3], [7, 4, 2]])
        with pytest.raises(ValueError, match="symmetry"):
            pore.attach_special(pms.gen.tms(),  0, [0, 1], 3, symmetry="DOTA")

        # Finalize
        pore.finalize()
        pore.store("output/cylinder/")
        table = pore.table()
        print(table)
        assert list(table.columns) == ["Interior", "Exterior"]
        assert "Surface area (nm^2)" in table.index
        assert "Surface chemistry - Before Functionalization" in table.index
        assert "Surface chemistry - After Functionalization" in table.index

        ## Properties
        roughness = pore.roughness()
        surface = pore.surface()
        allocation = pore.allocation()
        assert isinstance(roughness, pms.RoughnessProfile)
        assert isinstance(surface, pms.SurfaceAreaSummary)
        assert isinstance(allocation["Hydro"], pms.AllocationSummary)
        assert isinstance(allocation["Hydro"].interior, pms.SurfaceAllocationStats)
        assert round(pore.diameter()[0]) == 4
        assert [round(x, 4) for x in pore.centroid()] == [3.0147, 3.0572, 3.0569]
        assert round(roughness.interior[0], 1) == 0.1
        assert round(roughness.exterior, 1) == 0.0
        assert pore.volume() == pytest.approx(77.8, abs=1.0)
        assert surface.interior == pytest.approx(78.0, abs=1.0)
        assert surface.exterior == pytest.approx(49.0, abs=1.0)

    def test_pore_slit(self):
        # Empty pore
        pore = pms.PoreSlit([4, 4, 4], 2)
        pore.finalize()

        # Filled pore
        pore = pms.PoreSlit([6, 6, 6], 3, 5, [5, 5])

        ## Attachment
        # pore.attach_special(pms.gen.tms(),  0, [0, 1], 5)
        # pore.attach_special(pms.gen.tms(),  0, [0, 1], 3, symmetry="mirror")

        tms2 = pms.gen.tms()
        tms2.set_short("TMS2")

        pore.attach(tms2, 0, [0, 1], 10, "in", trials=10, inp="percent")
        pore.attach(tms2, 0, [0, 1], 1, "in", trials=10, inp="molar")
        pore.attach(tms2, 0, [0, 1], 0.1, "ex", trials=10, inp="molar")

        # Special cases
        print()
        with pytest.raises(ValueError, match="site_type"):
            pore.attach(pms.gen.tms(), 0, [0, 1], 100, site_type="DOTA")
        with pytest.raises(ValueError, match="inp"):
            pore.attach(pms.gen.tms(), 0, [0, 1], 100, "in", inp="DOTA")
        with pytest.raises(ValueError, match="symmetry"):
            pore.attach_special(pms.gen.tms(),  0, [0, 1], 3, symmetry="DOTA")

        # Finalize
        pore.finalize()
        pore.store("output/slit/")
        print(pore.table())

        ## Properties
        roughness = pore.roughness()
        surface = pore.surface()
        assert round(pore.diameter()[0]) == 3
        assert [round(x, 4) for x in pore.centroid()] == [3.0147, 3.0572, 3.0569]
        assert round(roughness.interior[0], 1) == 0.1
        assert round(roughness.exterior, 1) == 0.0
        assert pore.volume() == pytest.approx(112.4, abs=1.0)
        assert surface.interior == pytest.approx(74.3, abs=1.0)

    def test_pore_capsule(self):
        unassigned_warning = "Some interior silicon binding sites could not be assigned"

        # Empty pore
        pore = pms.PoreCapsule([3, 3, 6], 2, 1, 2.5)
        pore.finalize()

        # Filled pore
        with pytest.warns(RuntimeWarning, match=unassigned_warning):
            pore = pms.PoreCapsule([6, 6, 10], 4, 2, 5, [5, 5])

        ## Attachment
        # pore.attach_special(pms.gen.tms(),  0, [0, 1], 5)
        # pore.attach_special(pms.gen.tms(),  0, [0, 1], 3, symmetry="mirror")

        tms2 = pms.gen.tms()
        tms2.set_short("TMS2")

        pore.attach(tms2, 0, [0, 1], 10, "in", trials=10, inp="percent")
        pore.attach(tms2, 0, [0, 1], 1, "in", trials=10, inp="molar")
        pore.attach(tms2, 0, [0, 1], 0.1, "ex", trials=10, inp="molar")

        # Special cases
        print()
        with pytest.raises(ValueError, match="site_type"):
            pore.attach(pms.gen.tms(), 0, [0, 1], 100, site_type="DOTA")
        with pytest.raises(ValueError, match="inp"):
            pore.attach(pms.gen.tms(), 0, [0, 1], 100, "in", inp="DOTA")
        # Finalize
        pore.finalize()
        #pore.store("output/capsule/")
        print(pore.table())

        # Properties
        roughness = pore.roughness()
        surface = pore.surface()
        for actual, expected in zip(pore.diameter(), [4.234, 4.4864, 4.5656, 4.214]):
            assert actual == pytest.approx(expected, abs=0.05)
        assert [round(x, 4) for x in pore.centroid()] == [3.0147, 3.0572, 4.9169]
        for actual, expected in zip(roughness.interior, [0.1287, 0.1065, 0.1439, 0.1226]):
            assert actual == pytest.approx(expected, abs=0.05)
        assert round(roughness.exterior, 1) == 0.0
        assert pore.volume() == pytest.approx(153.2, abs=1.0) # not correct volume because sphere and cyclinder merged correct is 100
        assert surface.interior == pytest.approx(182.0, abs=1.0) # not correct because whole sphere surface is take in to account, correct is 113
        assert surface.exterior == pytest.approx(44.3, abs=1.0)

    def test_pore_cylinder_amorph(self):
        unassigned_warning = "Some interior silicon binding sites could not be assigned"

        # Empty pore
        pore = pms.PoreAmorphCylinder(2, 0)
        pore.finalize()

        # Filled pore
        with pytest.warns(RuntimeWarning, match=unassigned_warning):
            pore = pms.PoreAmorphCylinder(4, 5, [2, 2])

        ## Attachment
        # pore.attach_special(pms.gen.tms(),  0, [0, 1], 5)
        # pore.attach_special(pms.gen.tms(),  0, [0, 1], 3, symmetry="mirror")

        tms2 = pms.gen.tms()
        tms2.set_short("TMS2")

        pore.attach(tms2, 0, [0, 1], 10, "in", trials=10, inp="percent")
        pore.attach(tms2, 0, [0, 1], 1, "in", trials=10, inp="molar")
        pore.attach(tms2, 0, [0, 1], 0.1, "ex", trials=10, inp="molar")

        # Special cases
        print()
        with pytest.raises(ValueError, match="site_type"):
            pore.attach(pms.gen.tms(), 0, [0, 1], 100, site_type="DOTA")
        with pytest.raises(ValueError, match="inp"):
            pore.attach(pms.gen.tms(), 0, [0, 1], 100, "in", inp="DOTA")
        with pytest.raises(ValueError, match="positions"):
            pore.attach(pms.gen.tms(), 0, [0, 1], 100, pos_list=[[1, 3, 3], [7, 4, 2]])
        with pytest.raises(ValueError, match="symmetry"):
            pore.attach_special(pms.gen.tms(),  0, [0, 1], 3, symmetry="DOTA")

        # Finalize
        pore.finalize()
        pore.store("output/cylinder_amorph/")
        print(pore.table())

        ## Properties
        roughness = pore.roughness()
        surface = pore.surface()
        assert round(pore.diameter()[0]) == 4
        assert [round(x, 4) for x in pore.centroid()] == [4.7958, 4.7978, 4.807]
        assert round(roughness.interior[0], 1) == 0.1
        assert round(roughness.exterior, 1) == 0.3
        assert pore.volume() == pytest.approx(119.6, abs=1.0)
        assert surface.interior == pytest.approx(120.1, abs=1.0)
        assert surface.exterior == pytest.approx(159.8, abs=1.0)
