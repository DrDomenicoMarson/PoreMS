################################################################################
# Store Class                                                                  #
#                                                                              #
"""Writers for structure, topology, and helper simulation files."""
################################################################################


from dataclasses import dataclass
import os
import shutil
import warnings

import porems.utils as utils
import porems.database as db

from porems.molecule import Molecule
from porems.pore import Pore


@dataclass(frozen=True)
class _StructureAtomRecord:
    """One serialized atom record used by structure writers.

    Parameters
    ----------
    serial : int
        One-based atom serial number in writer order.
    molecule_index : int
        Zero-based molecule index in the writer output order.
    local_atom_index : int
        Zero-based atom index inside the source molecule.
    residue_short : str
        Short residue identifier of the source molecule.
    residue_id : int
        One-based residue identifier in writer order.
    atom_name : str
        Final atom name written by the structure writer.
    atom_type : str
        Element or atom-type token of the source atom.
    position : tuple[float, float, float]
        Cartesian position in nanometers.
    source_id : int or None
        Optional source atom identifier from the originating pore block.
    """

    serial: int
    molecule_index: int
    local_atom_index: int
    residue_short: str
    residue_id: int
    atom_name: str
    atom_type: str
    position: tuple[float, float, float]
    source_id: int | None


class Store:
    """Write molecule and pore objects to simulation input files.

    The writer supports several structure formats for
    :class:`porems.molecule.Molecule` objects and adds GROMACS-specific
    topology helpers when the input is a :class:`porems.pore.Pore`.
    Template-based helper files for Antechamber and grid topologies are also
    available through this interface.

    Parameters
    ----------
    inp : Molecule or Pore
        Molecule or pore object to serialize.
    link : str, optional
        Output directory for generated files.
    sort_list : list, optional
        Optional pore molecule ordering used when writing topology entries.

    Examples
    --------
    Assuming a pore or molecule object have already been created, the structure
    files can be generated like following examples

    .. code-block:: python

        Store(mol).pdb()
        Store(pore, "output").gro("pore.gro")
    """
    def __init__(self, inp, link="./", sort_list=None):
        """Initialize a structure writer for a molecule or pore object.

        Parameters
        ----------
        inp : Molecule or Pore
            Molecule or pore object to serialize.
        link : str, optional
            Output directory for generated files.
        sort_list : list, optional
            Molecule short-name ordering for pore outputs.

        Raises
        ------
        TypeError
            Raised when ``inp`` is neither a :class:`porems.molecule.Molecule`
            nor a :class:`porems.pore.Pore`.
        ValueError
            Raised when a provided pore ``sort_list`` does not match the pore
            molecule dictionary keys exactly.
        """
        # Initialize
        self._dim = 3
        self._link = link if link[-1] == "/" else link+"/"
        self._inp = inp
        sort_list = [] if sort_list is None else sort_list

        # Process input
        if isinstance(inp, Molecule):
            self._mols = [self._inp]
        elif isinstance(inp, Pore):
            if sort_list:
                if sorted(sort_list) == sorted(list(inp.get_mol_dict().keys())):
                    self._mols = sum([inp.get_mol_dict()[x] for x in sort_list], [])
                    self._short_list = sort_list
                else:
                    raise ValueError("Store: Sorting list does not contain all keys...")
            else:
                self._mols = sum([x for x in inp.get_mol_dict().values()], [])
                self._short_list = list(x for x in inp.get_mol_dict().keys())
        else:
            raise TypeError("Store: Unsupported input type...")

        # Get properties after input checking type
        self._name = inp.get_name() if inp.get_name() else "molecule"
        self._box = inp.get_box() if inp.get_box() else [0 for x in range(self._dim)]

        # Create output folder
        utils.mkdirp(link)


    ###############
    # Antechamber #
    ###############
    def job(self, name="", master=""):
        """Write Antechamber helper scripts for the current molecule.

        A shell job file and a matching ``tleap`` input are generated from the
        bundled templates. They can be used to derive force-field parameters
        for standalone molecules outside the pore workflow.

        Parameters
        ----------
        name : str, optional
            Base filename for the generated helper files.
        master : str, optional
            Optional master shell script to which execution commands for the
            generated job are appended.
        """
        # Initialize
        link = self._link
        mol_name = self._name.lower()
        short = self._inp.get_short()
        name = name if name else self._name

        # Template directory
        package_dir = os.path.split(__file__)[0]+"/"

        # Job file
        file_in = package_dir+"templates/antechamber.job"
        file_out = link+name+".job"
        shutil.copy(file_in, file_out)

        utils.replace(file_out, "MOLNAME", mol_name)

        # Tleap file
        file_in = package_dir+"templates/antechamber.tleap"
        file_out = link+name+".tleap"
        shutil.copy(file_in, file_out)

        utils.replace(file_out, "MOLSHORTLOWER", mol_name)
        utils.replace(file_out, "MOLSHORT", short)
        utils.replace(file_out, "MOLNAME", mol_name)

        # Add to master run
        if master:
            fileMaster = open(link+master, "a")
            fileMaster.write("cd "+mol_name+" #\n")
            fileMaster.write("sh "+mol_name+".job #\n")
            fileMaster.write("cd .. #\n")
            fileMaster.write("echo \"Finished "+mol_name+"...\"\n")
            fileMaster.write("#\n")
            fileMaster.close()


    #############
    # Structure #
    #############
    def obj(self, name=""):
        """Serialize the wrapped object with pickle.

        Parameters
        ----------
        name : str, optional
            Output filename. Defaults to ``<name>.obj``.
        """
        # Initialize
        link = self._link
        link += name if name else self._name+".obj"

        # Save object
        utils.save(self._inp, link)

    def _collect_structure_records(self, use_atom_names=False):
        """Collect serialized atom metadata in the current writer order.

        Parameters
        ----------
        use_atom_names : bool, optional
            True to preserve explicit atom names when available. False to
            enumerate atom names from atom types.

        Returns
        -------
        atom_records : list[_StructureAtomRecord]
            Serialized atom metadata in file order.
        molecule_serials : list[list[int]]
            Atom serial numbers grouped per written molecule.
        """
        atom_records = []
        molecule_serials = []
        atom_serial = 1
        residue_serial = 1

        for molecule_index, mol in enumerate(self._mols):
            atom_types = {}
            temp_residue = 0
            molecule_serial = []

            for local_atom_index, atom in enumerate(mol.get_atom_list()):
                if atom.get_residue() != temp_residue:
                    residue_serial += 1
                    temp_residue = atom.get_residue()

                atom_type = atom.get_atom_type()
                if atom_type not in atom_types:
                    atom_types[atom_type] = 1

                if use_atom_names and atom.get_name():
                    atom_name = atom.get_name()
                else:
                    atom_name = atom_type + str(atom_types[atom_type])

                atom_records.append(
                    _StructureAtomRecord(
                        serial=atom_serial,
                        molecule_index=molecule_index,
                        local_atom_index=local_atom_index,
                        residue_short=mol.get_short(),
                        residue_id=residue_serial,
                        atom_name=atom_name,
                        atom_type=atom_type,
                        position=tuple(atom.get_pos()),
                        source_id=atom.get_source_id(),
                    )
                )
                molecule_serial.append(atom_serial)

                atom_serial += 1
                atom_types[atom_type] = atom_types[atom_type] + 1 if atom_types[atom_type] < 99 else 1

            molecule_serials.append(molecule_serial)
            residue_serial += 1

        return atom_records, molecule_serials

    def _warn_if_pdb_limits_exceeded(self, atom_records):
        """Warn when serialized PDB identifiers exceed fixed-width fields.

        Parameters
        ----------
        atom_records : list[_StructureAtomRecord]
            Serialized atom metadata in file order.
        """
        if not atom_records:
            return

        max_atom_serial = max(record.serial for record in atom_records)
        max_residue_serial = max(record.residue_id for record in atom_records)
        if max_atom_serial <= 99999 and max_residue_serial <= 9999:
            return

        warnings.warn(
            "PDB fixed-width fields are exceeded by this export "
            f"(max atom serial={max_atom_serial}, max residue id={max_residue_serial}). "
            "PoreMS will still write the full identifiers, but the resulting file is "
            "non-standard and some viewers may misread atom or residue numbering. "
            "Prefer mmCIF for large systems.",
            UserWarning,
            stacklevel=2,
        )

    def _pdb_known_residue_bonds(self, residue_short, atom_count):
        """Return built-in internal bonds for known surface residue templates.

        Parameters
        ----------
        residue_short : str
            Short residue identifier of the written molecule.
        atom_count : int
            Number of atoms present in that molecule.

        Returns
        -------
        bonds : tuple[tuple[int, int], ...]
            Zero-based local atom-index pairs that should be written as
            ``CONECT`` records.
        """
        if residue_short == "SL" and atom_count >= 3:
            return ((0, 1), (1, 2))
        if residue_short == "SLG" and atom_count >= 5:
            return ((0, 1), (1, 2), (0, 3), (3, 4))
        return ()

    def _pdb_conect_pairs(self, atom_records, molecule_serials):
        """Collect inspection-only PDB connectivity pairs.

        Parameters
        ----------
        atom_records : list[_StructureAtomRecord]
            All atom records written to the current PDB file.
        molecule_serials : list[list[int]]
            Atom serial numbers grouped per written molecule.

        Returns
        -------
        bond_pairs : list[tuple[int, int]]
            Sorted unique atom-serial pairs to emit as ``CONECT`` records.
        """
        bond_pairs = set()

        source_serials = {
            record.source_id: record.serial
            for record in atom_records
            if record.source_id is not None
        }
        if isinstance(self._inp, Pore):
            for atom_id, props in self._inp._matrix.get_matrix().items():
                if atom_id not in source_serials:
                    continue
                for neighbor_id in props["atoms"]:
                    if neighbor_id not in source_serials:
                        continue
                    serial_a = source_serials[atom_id]
                    serial_b = source_serials[neighbor_id]
                    bond_pairs.add(tuple(sorted((serial_a, serial_b))))

        for molecule_index, serials in enumerate(molecule_serials):
            residue_short = self._mols[molecule_index].get_short()
            for atom_a, atom_b in self._pdb_known_residue_bonds(residue_short, len(serials)):
                if atom_a < len(serials) and atom_b < len(serials):
                    bond_pairs.add(tuple(sorted((serials[atom_a], serials[atom_b]))))

        return sorted(bond_pairs)

    def _write_pdb_conect_records(self, file_out, bond_pairs):
        """Write ``CONECT`` records for previously collected bond pairs.

        Parameters
        ----------
        file_out : TextIO
            Open output stream.
        bond_pairs : list[tuple[int, int]]
            Sorted unique atom-serial pairs.
        """
        neighbors = {}
        for serial_a, serial_b in bond_pairs:
            neighbors.setdefault(serial_a, []).append(serial_b)
            neighbors.setdefault(serial_b, []).append(serial_a)

        for serial in sorted(neighbors):
            bonded = sorted(set(neighbors[serial]))
            for start in range(0, len(bonded), 4):
                chunk = bonded[start:start + 4]
                file_out.write(
                    "CONECT"
                    + f"{serial:5d}"
                    + "".join(f"{neighbor:5d}" for neighbor in chunk)
                    + "\n"
                )

    def _write_cif_struct_conn_loop(self, file_out, atom_records, molecule_serials):
        """Write an mmCIF bond loop for known scaffold and surface connectivity.

        Parameters
        ----------
        file_out : TextIO
            Open output stream.
        atom_records : list[_StructureAtomRecord]
            Serialized atom metadata in file order.
        molecule_serials : list[list[int]]
            Atom serial numbers grouped per written molecule.
        """
        bond_pairs = self._pdb_conect_pairs(atom_records, molecule_serials)
        if not bond_pairs:
            return

        record_by_serial = {record.serial: record for record in atom_records}
        file_out.write("#\n")
        file_out.write("loop_\n")
        file_out.write("_struct_conn.id\n")
        file_out.write("_struct_conn.conn_type_id\n")
        file_out.write("_struct_conn.ptnr1_label_asym_id\n")
        file_out.write("_struct_conn.ptnr1_label_comp_id\n")
        file_out.write("_struct_conn.ptnr1_label_seq_id\n")
        file_out.write("_struct_conn.ptnr1_label_atom_id\n")
        file_out.write("_struct_conn.ptnr2_label_asym_id\n")
        file_out.write("_struct_conn.ptnr2_label_comp_id\n")
        file_out.write("_struct_conn.ptnr2_label_seq_id\n")
        file_out.write("_struct_conn.ptnr2_label_atom_id\n")

        for conn_index, (serial_a, serial_b) in enumerate(bond_pairs, start=1):
            record_a = record_by_serial[serial_a]
            record_b = record_by_serial[serial_b]
            file_out.write(
                f"conn{conn_index} covale "
                f"A {record_a.residue_short} {record_a.residue_id} {record_a.atom_name} "
                f"A {record_b.residue_short} {record_b.residue_id} {record_b.atom_name}\n"
            )

    def cif(self, name="", use_atom_names=False, write_bonds=True):
        """Write the current structure in mmCIF format.

        Parameters
        ----------
        name : str, optional
            Output filename. Defaults to ``<name>.cif``.
        use_atom_names : bool, optional
            True to preserve explicit atom names when available. False to
            enumerate atom names from atom types.
        write_bonds : bool, optional
            When ``True`` (the default), also emit an ``_struct_conn`` loop
            for the active silica scaffold and built-in ``SL``/``SLG``
            surface residues. Ligand-internal connectivity is not
            reconstructed here.
        """
        link = self._link
        link += name if name else self._name + ".cif"
        atom_records, molecule_serials = self._collect_structure_records(use_atom_names)
        data_name = self._name.replace(" ", "_")

        with open(link, "w") as file_out:
            file_out.write(f"data_{data_name}\n")
            file_out.write("#\n")
            file_out.write("_symmetry.space_group_name_H-M 'P 1'\n")
            file_out.write("_symmetry.Int_Tables_number 1\n")
            file_out.write(f"_cell.length_a {self._box[0] * 10:.3f}\n")
            file_out.write(f"_cell.length_b {self._box[1] * 10:.3f}\n")
            file_out.write(f"_cell.length_c {self._box[2] * 10:.3f}\n")
            file_out.write("_cell.angle_alpha 90.000\n")
            file_out.write("_cell.angle_beta 90.000\n")
            file_out.write("_cell.angle_gamma 90.000\n")
            file_out.write("#\n")
            file_out.write("loop_\n")
            file_out.write("_atom_site.group_PDB\n")
            file_out.write("_atom_site.id\n")
            file_out.write("_atom_site.type_symbol\n")
            file_out.write("_atom_site.label_atom_id\n")
            file_out.write("_atom_site.label_alt_id\n")
            file_out.write("_atom_site.label_comp_id\n")
            file_out.write("_atom_site.label_asym_id\n")
            file_out.write("_atom_site.label_entity_id\n")
            file_out.write("_atom_site.label_seq_id\n")
            file_out.write("_atom_site.pdbx_PDB_ins_code\n")
            file_out.write("_atom_site.Cartn_x\n")
            file_out.write("_atom_site.Cartn_y\n")
            file_out.write("_atom_site.Cartn_z\n")
            file_out.write("_atom_site.occupancy\n")
            file_out.write("_atom_site.B_iso_or_equiv\n")
            file_out.write("_atom_site.pdbx_formal_charge\n")
            file_out.write("_atom_site.auth_seq_id\n")
            file_out.write("_atom_site.auth_comp_id\n")
            file_out.write("_atom_site.auth_asym_id\n")
            file_out.write("_atom_site.auth_atom_id\n")
            file_out.write("_atom_site.pdbx_PDB_model_num\n")

            for record in atom_records:
                file_out.write(
                    " ".join(
                        [
                            "HETATM",
                            str(record.serial),
                            record.atom_type,
                            record.atom_name,
                            ".",
                            record.residue_short,
                            "A",
                            "1",
                            str(record.residue_id),
                            "?",
                            f"{record.position[0] * 10:.3f}",
                            f"{record.position[1] * 10:.3f}",
                            f"{record.position[2] * 10:.3f}",
                            "1.00",
                            "0.00",
                            "?",
                            str(record.residue_id),
                            record.residue_short,
                            "A",
                            record.atom_name,
                            "1",
                        ]
                    )
                    + "\n"
                )

            if write_bonds:
                self._write_cif_struct_conn_loop(file_out, atom_records, molecule_serials)

            file_out.write("#\n")

    def pdb(self, name="", use_atom_names=False, write_conect=True):
        """Write the current structure in PDB format.

        Parameters
        ----------
        name : str, optional
            Output filename. Defaults to ``<name>.pdb``.
        use_atom_names : bool, optional
            True to preserve explicit atom names when available. False to
            enumerate atom names from atom types.
        write_conect : bool, optional
            When ``True`` (the default), also emit inspection-oriented
            ``CONECT`` records for the active silica scaffold and the built-in
            ``SL``/``SLG`` surface residues. Ligand-internal connectivity is
            not reconstructed here.
        """
        # Initialize
        link = self._link
        link += name if name else self._name+".pdb"
        atom_records, molecule_serials = self._collect_structure_records(use_atom_names)
        self._warn_if_pdb_limits_exceeded(atom_records)

        # Open file
        with open(link, "w") as file_out:
            for record in atom_records:
                out_string = "HETATM"                       #  1- 6 (6)    Record name
                out_string += f"{record.serial:5d}"        #  7-11 (5)    Atom serial number
                out_string += " "                          # 12    (1)    -
                out_string += f"{record.atom_name:>4s}"    # 13-16 (4)    Atom name
                out_string += " "                          # 17    (1)    Alternate location indicator
                out_string += f"{record.residue_short:>3s}"# 18-20 (3)    Residue name
                out_string += " "                          # 21    (1)    -
                out_string += "A"                          # 22    (1)    Chain identifier
                out_string += f"{record.residue_id:4d}"    # 23-26 (4)    Residue sequence number
                out_string += " "                          # 27    (1)    Code for insertion of residues
                out_string += "   "                        # 28-30 (3)    -
                for coord in record.position:              # 31-54 (3*8)  Coordinates
                    out_string += f"{coord*10:8.3f}"
                out_string += f"{1:6.2f}"                  # 55-60 (6)    Occupancy
                out_string += f"{0:6.2f}"                  # 61-66 (6)    Temperature factor
                out_string += "          "                 # 67-76 (10)   -
                out_string += f"{record.atom_type:>2s}"    # 77-78 (2)    Element symbol
                out_string += "  "                         # 79-80 (2)    Charge on the atom

                file_out.write(out_string+"\n")

            if write_conect:
                self._write_pdb_conect_records(
                    file_out,
                    self._pdb_conect_pairs(atom_records, molecule_serials),
                )

            # End statement
            file_out.write("TER\nEND\n")

    def gro(self, name="", use_atom_names=False):
        """Write the current structure in GROMACS GRO format.

        Parameters
        ----------
        name : str, optional
            Output filename. Defaults to ``<name>.gro``.
        use_atom_names : bool, optional
            True to preserve explicit atom names when available. False to
            enumerate atom names from atom types.
        """
        # Initialize
        link = self._link
        link += name if name else self._name+".gro"

        # Open file
        with open(link, "w") as file_out:
            # Set title
            file_out.write("Molecule generated using the PoreMS package\n")

            # Number of atoms
            file_out.write("%i" % sum([x.get_num() for x in self._mols])+"\n")

            # Set counter
            num_a = 1
            num_m = 1

            # Run through molecules
            for mol in self._mols:
                atom_types = {}
                temp_res_id = 0
                # Run through atoms
                for atom in mol.get_atom_list():
                    # Process residue index
                    if not atom.get_residue() == temp_res_id:
                        num_m = num_m+1 if num_m < 99999 else 0
                        temp_res_id = atom.get_residue()

                    # Get atom type
                    atom_type = atom.get_atom_type()

                    # Create type dictionary
                    if atom_type not in atom_types:
                        atom_types[atom_type] = 1

                    # Set atom name
                    if use_atom_names and atom.get_name():
                        atom_name = atom.get_name()
                    else:
                        atom_name = atom_type+str(atom_types[atom_type])

                    # Write file
                    out_string = "%5i" % num_m              #  1- 5 (5)    Residue number
                    out_string += "%-5s" % mol.get_short()  #  6-10 (5)    Residue short name
                    out_string += "%5s" % atom_name         # 11-15 (5)    Atom name
                    out_string += "%5i" % num_a             # 16-20 (5)    Atom number
                    for i in range(self._dim):                    # 21-44 (3*8)  Coordinates
                        out_string += "%8.3f" % atom.get_pos()[i]

                    file_out.write(out_string+"\n")

                    # Process counter
                    num_a = num_a+1 if num_a < 99999 else 0
                    atom_types[atom_type] = atom_types[atom_type]+1 if atom_types[atom_type] < 999 else 0
                num_m = num_m+1 if num_m < 99999 else 0

            # Box
            out_string = ""
            for i in range(self._dim):
                out_string += "%.3f" % self._box[i]
                out_string += " " if i < self._dim-1 else "\n"

            file_out.write(out_string)

    def xyz(self, name="", use_atom_names=False):
        """Write the current structure in XYZ format.

        Parameters
        ----------
        name : str, optional
            Output filename. Defaults to ``<name>.xyz``.
        use_atom_names : bool, optional
            Retained for API compatibility. XYZ output always uses atom types as
            element labels.
        """
        # Initialize
        link = self._link
        link += name if name else self._name+".xyz"

        # Open output file and set title
        with open(link, "w") as file_out:
            # Header
            file_out.write("%i" % sum([x.get_num() for x in self._mols])+"\n"+"Energy = \n")

            # Run through molecules
            for mol in self._mols:
                # Run through atoms
                for atom in mol.get_atom_list():
                    # Write file
                    out_string = "%-2s" % atom.get_atom_type()  # 1- 2 (2)     Atom name
                    for i in range(self._dim):                  # 3-41 (3*13)  Coordinates
                        out_string += "%13.7f" % (atom.get_pos()[i]*10)

                    file_out.write(out_string+"\n")

    def lmp(self, name=""):
        """Write the current structure in LAMMPS data format.

        Coordinates are written in Angstrom assuming ``real`` units.

        Parameters
        ----------
        name : str, optional
            Output filename. Defaults to ``<name>.lmp``.
        """
        # Initialize
        link = self._link
        link += name if name else self._name+".lmp"

        # Atom types
        atom_types = list(set(sum([[x.get_atom_type(i) for i in range(x.get_num())] for x in self._mols], [])))

        # Open file
        with open(link, "w") as file_out:
            # Set title
            file_out.write("Molecule generated using the PoreMS package\n\n")

            # Porperties section
            file_out.write("%i" % sum([x.get_num() for x in self._mols])+" atoms\n")
            file_out.write("%i" % len(atom_types)+" atom types\n")
            file_out.write("\n")

            # Box size - Periodic boundary conditions
            file_out.write("0.000 "+"%.3f" % (self._box[0]*10)+" xlo xhi\n")
            file_out.write("0.000 "+"%.3f" % (self._box[1]*10)+" ylo yhi\n")
            file_out.write("0.000 "+"%.3f" % (self._box[2]*10)+" zlo zhi\n")
            file_out.write("\n")

            # Masses
            file_out.write("Masses\n\n")
            for i, at in enumerate(atom_types):
                file_out.write("%i"%(i+1)+" "+"%8.3f"%db.get_mass(at)+"\n")
            file_out.write("\n")

            # Atoms
            file_out.write("Atoms\n\n")

            # Set counter
            num_a = 1
            num_m = 1

            # Run through molecules
            for mol in self._mols:
                temp_res_id = 0
                # Run through atoms
                for atom in mol.get_atom_list():
                    # Process residue index
                    if not atom.get_residue() == temp_res_id:
                        num_m = num_m+1
                        temp_res_id = atom.get_residue()

                    # Get atom type
                    atom_type_id = atom_types.index(atom.get_atom_type())+1

                    # Write atom line
                    out_string  = "%5i" % num_a + " "        #  Atom number
                    out_string += "%5i" % num_m + " "        #  Residue number
                    out_string += "%3i" % atom_type_id + " " #  Atom type
                    out_string += "%5i" % 0 + " "            #  Charge
                    for i in range(self._dim):               #  Coordinates
                        out_string += "%8.3f" % (atom.get_pos()[i]*10)
                        out_string += " " if i<self._dim-1 else ""
                    file_out.write(out_string+"\n")

                    # Process counter
                    num_a = num_a+1
                num_m = num_m+1


    ############
    # Topology #
    ############
    def top(self, name=""):
        """Write the main GROMACS topology for a pore object.

        The generated ``.top`` file includes molecule ``.itp`` files and the
        final molecule counts stored in the wrapped pore.

        Parameters
        ----------
        name : str, optional
            Output filename. Defaults to ``<name>.top``.

        Raises
        ------
        TypeError
            Raised when topology generation is requested for a non-pore input.
        """
        # Check input type
        if not isinstance(self._inp, Pore):
            raise TypeError("Store: Unsupported input type for topology creation...")

        # Initialize
        link = self._link
        link += name if name else self._name+".top"

        # Copy master topology file
        utils.copy(os.path.split(__file__)[0]+"/templates/topol.top", link)

        # Open file
        with open(link, "a") as file_out:
            # Include topology
            for mol_short in self._short_list:
                if mol_short not in ["SI", "OM", "SL", "SLG", "SLX"]:
                    file_out.write("#include \""+mol_short+".itp\"\n")

            file_out.write("\n")
            file_out.write("[ system ]\n")
            file_out.write("Pore-System Generated by the PoreMS Package\n")

            file_out.write("\n")
            file_out.write("[ molecules ]\n")

            # Number of atoms
            for mol_short in self._short_list:
                file_out.write(mol_short+" "+str(len(self._inp.get_mol_dict()[mol_short]))+"\n")

    def grid(self, name="", charges=None):
        """Write the ``grid.itp`` topology template for silica grid atoms.

        Parameters
        ----------
        name : str, optional
            Output filename. Defaults to ``grid.itp``.
        charges : dict, optional
            Charge mapping containing ``"si"`` and ``"om"`` entries for the
            silicon and oxygen grid atoms.
        """
        charges = {"si": 1.28, "om": -0.64} if charges is None else charges

        # Initialize
        link = self._link
        link += name if name else "grid.itp"

        # Copy grid file
        utils.copy(os.path.split(__file__)[0]+"/templates/grid.itp", link)

        # Replace charges
        utils.replace(link, "CHARGEO", "%8.6f" % charges["om"])
        utils.replace(link, "CHARGESI", "%8.6f" % charges["si"])
