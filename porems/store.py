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

from porems.connectivity import (
    AssembledStructureGraph,
    ConnectivityValidationFinding,
    ConnectivityValidationReport,
    GraphBond,
)
from porems.molecule import Molecule
from porems.pore import Pore
from porems.topology import (
    GromacsAngle,
    GromacsAtom,
    GromacsAtomType,
    GromacsBond,
    GromacsDihedral,
    GromacsMoleculeType,
    GromacsPair,
    SilicaTopologyModel,
    default_silica_topology,
    parse_flat_itp,
    render_itp,
    render_top,
)


_HYBRID36_DIGITS_UPPER = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_HYBRID36_DIGITS_LOWER = "0123456789abcdefghijklmnopqrstuvwxyz"
_PDB_CHAIN_ID = "A"
_CIF_LABEL_SEQ_ID = "1"
_FULL_SLIT_NREXCL = 3


@dataclass(frozen=True)
class _PdbResidueAliasRecord:
    """Mapping from one native residue name to its PDB-safe alias.

    Parameters
    ----------
    full_name : str
        Full residue identifier used internally and in mmCIF.
    pdb_name : str
        Three-character residue alias written to the PDB residue-name field.
    """

    full_name: str
    pdb_name: str


@dataclass(frozen=True)
class _CifEntityRecord:
    """One mmCIF entity row used by the structure writer.

    Parameters
    ----------
    entity_id : str
        mmCIF entity identifier referenced from ``_atom_site``.
    residue_name : str
        Full residue identifier represented by the entity.
    entity_type : str, optional
        mmCIF entity type. Functionalized slit exports use ``"non-polymer"``.
    """

    entity_id: str
    residue_name: str
    entity_type: str = "non-polymer"


@dataclass(frozen=True)
class _CifStructAsymRecord:
    """One mmCIF structural-asymmetry row used by the writer.

    Parameters
    ----------
    asym_id : str
        Unique label asymmetry identifier for one residue instance.
    entity_id : str
        Entity identifier referenced by ``asym_id``.
    residue_id : int
        One-based residue identifier in writer order.
    residue_name : str
        Full residue identifier represented by the asymmetry row.
    """

    asym_id: str
    entity_id: str
    residue_id: int
    residue_name: str


@dataclass(frozen=True)
class _ResidueExportIdentifiers:
    """Format-specific identifiers assigned to one residue instance.

    Parameters
    ----------
    residue_id : int
        One-based residue identifier in writer order.
    residue_name : str
        Full residue identifier preserved in mmCIF.
    pdb_residue_name : str
        Three-character PDB-safe residue alias.
    pdb_chain_id : str
        PDB chain identifier written to column 22.
    pdb_residue_id_token : str
        Four-character hybrid-36 residue-sequence token.
    cif_entity_id : str
        Entity identifier referenced from ``_atom_site.label_entity_id``.
    cif_asym_id : str
        Unique asymmetry identifier for ``_atom_site.label_asym_id``.
    cif_label_seq_id : str
        Label sequence token used by the mmCIF writer.
    """

    residue_id: int
    residue_name: str
    pdb_residue_name: str
    pdb_chain_id: str
    pdb_residue_id_token: str
    cif_entity_id: str
    cif_asym_id: str
    cif_label_seq_id: str


def _encode_pure(digits, value, width):
    """Encode one non-negative integer in a fixed-width positional alphabet.

    Parameters
    ----------
    digits : str
        Digits used by the positional numeral system.
    value : int
        Non-negative integer to encode.
    width : int
        Output width in characters.

    Returns
    -------
    token : str
        Encoded value padded to ``width`` characters.

    Raises
    ------
    ValueError
        Raised when ``value`` is negative or does not fit in ``width``
        characters for the selected alphabet.
    """
    if value < 0:
        raise ValueError("Encoded values must be non-negative.")

    base = len(digits)
    buffer = ["0"] * width
    remainder = value
    for index in range(width - 1, -1, -1):
        buffer[index] = digits[remainder % base]
        remainder //= base

    if remainder != 0:
        raise ValueError(f"Value {value} does not fit in width {width}.")

    return "".join(buffer)


def _decode_pure(digits, token):
    """Decode one positional token produced by :func:`_encode_pure`.

    Parameters
    ----------
    digits : str
        Digits used by the positional numeral system.
    token : str
        Fixed-width encoded token.

    Returns
    -------
    value : int
        Decoded integer value.

    Raises
    ------
    ValueError
        Raised when ``token`` contains a character outside ``digits``.
    """
    base = len(digits)
    value = 0
    for char in token:
        try:
            digit = digits.index(char)
        except ValueError as exc:
            raise ValueError(f"Invalid digit {char!r} in token {token!r}.") from exc
        value = value * base + digit
    return value


def _hybrid36_max_value(width):
    """Return the maximum non-negative integer representable in hybrid-36.

    Parameters
    ----------
    width : int
        Field width in characters.

    Returns
    -------
    value : int
        Largest representable non-negative integer.
    """
    return (10 ** width) + (2 * 26 * (36 ** (width - 1))) - 1


def _encode_hybrid36(width, value):
    """Encode one non-negative integer using canonical hybrid-36.

    Parameters
    ----------
    width : int
        Field width in characters. PDB uses ``4`` for residue ids and ``5``
        for atom serials.
    value : int
        Non-negative integer to encode.

    Returns
    -------
    token : str
        Fixed-width hybrid-36 token.

    Raises
    ------
    ValueError
        Raised when ``value`` is negative or exceeds the hybrid-36 range for
        ``width``.
    """
    if value < 0:
        raise ValueError("Hybrid-36 values must be non-negative.")

    decimal_limit = 10 ** width
    if value < decimal_limit:
        return f"{value:>{width}d}"

    base36_block = 26 * (36 ** (width - 1))
    offset = value - decimal_limit
    if offset < base36_block:
        return _encode_pure(
            _HYBRID36_DIGITS_UPPER,
            offset + (10 * (36 ** (width - 1))),
            width,
        )

    offset -= base36_block
    if offset < base36_block:
        return _encode_pure(
            _HYBRID36_DIGITS_LOWER,
            offset + (10 * (36 ** (width - 1))),
            width,
        )

    raise ValueError(
        f"Value {value} exceeds the hybrid-36 range for width {width} "
        f"(max {_hybrid36_max_value(width)})."
    )


def _decode_hybrid36(width, token):
    """Decode one canonical hybrid-36 token.

    Parameters
    ----------
    width : int
        Field width in characters.
    token : str
        Fixed-width hybrid-36 token.

    Returns
    -------
    value : int
        Decoded non-negative integer.

    Raises
    ------
    ValueError
        Raised when ``token`` does not match a supported hybrid-36 pattern.
    """
    if len(token) != width:
        raise ValueError(
            f"Hybrid-36 token {token!r} does not match width {width}."
        )

    first = token[0]
    if first == " " or first.isdigit():
        return int(token)
    if first.isupper():
        return (
            _decode_pure(_HYBRID36_DIGITS_UPPER, token)
            - (10 * (36 ** (width - 1)))
            + (10 ** width)
        )
    if first.islower():
        return (
            _decode_pure(_HYBRID36_DIGITS_LOWER, token)
            + (10 ** width)
            + (16 * (36 ** (width - 1)))
        )

    raise ValueError(f"Unsupported hybrid-36 token {token!r}.")


def _normalize_pdb_identifier(value, fallback):
    """Return an uppercase alphanumeric identifier for PDB-safe aliases.

    Parameters
    ----------
    value : str
        Identifier to normalize.
    fallback : str
        Fallback token used when normalization removes all characters.

    Returns
    -------
    token : str
        Uppercase alphanumeric identifier.
    """
    token = "".join(
        character
        for character in str(value).upper()
        if character.isascii() and character.isalnum()
    )
    return token or fallback


def _cif_token(value):
    """Return one mmCIF-safe token for a simple loop writer.

    Parameters
    ----------
    value : object
        Value written to the mmCIF file.

    Returns
    -------
    token : str
        Plain token or single-quoted value when quoting is required.
    """
    token = str(value)
    if not token:
        return "."
    if any(character.isspace() for character in token) or "'" in token:
        return "'" + token.replace("'", "''") + "'"
    return token


def _sanitize_pdb_token(value, width, fallback=""):
    """Return an ASCII token that cannot overflow a fixed-width PDB field.

    Parameters
    ----------
    value : str
        Token to sanitize.
    width : int
        Maximum field width.
    fallback : str, optional
        Replacement token used when sanitization removes all characters.

    Returns
    -------
    token : str
        Sanitized token truncated to ``width`` characters.
    """
    token = "".join(
        character
        for character in str(value)
        if character.isascii() and character.isprintable() and not character.isspace()
    )
    token = token or fallback
    return token[:width]


def _format_decimal_token(value, places=6):
    """Return one fixed-precision decimal token.

    Parameters
    ----------
    value : float or str
        Value to format.
    places : int, optional
        Number of decimal places used for float inputs.

    Returns
    -------
    token : str
        String token suitable for topology output.
    """
    if isinstance(value, str):
        return value
    return f"{value:.{places}f}"


def _sanitize_gromacs_identifier(value, fallback="SLIT"):
    """Return one simple GROMACS-safe identifier token.

    Parameters
    ----------
    value : str
        Candidate identifier.
    fallback : str, optional
        Fallback token used when sanitization removes every character.

    Returns
    -------
    token : str
        Uppercase alphanumeric identifier allowing underscores.
    """
    token = "".join(
        character
        for character in str(value)
        if character.isascii() and (character.isalnum() or character == "_")
    ).upper()
    return token or fallback


def _silica_atomtypes_in_order(silica_topology):
    """Return silica atom types in the exported deterministic order.

    Parameters
    ----------
    silica_topology : SilicaTopologyModel
        Resolved silica topology model.

    Returns
    -------
    atomtypes : tuple[GromacsAtomType, ...]
        Silica atom types converted into immutable GROMACS records.
    """
    return (
        silica_topology.atomtypes.framework_silicon.to_gromacs_atomtype(),
        silica_topology.atomtypes.framework_oxygen.to_gromacs_atomtype(),
        silica_topology.atomtypes.silanol_oxygen.to_gromacs_atomtype(),
        silica_topology.atomtypes.silanol_hydrogen.to_gromacs_atomtype(),
    )


def _silica_atomtype_lookup(silica_topology):
    """Return silica atom types keyed by their exported atom-type names.

    Parameters
    ----------
    silica_topology : SilicaTopologyModel
        Resolved silica topology model.

    Returns
    -------
    atomtypes_by_name : dict[str, GromacsAtomType]
        Mapping from atom-type name to immutable GROMACS atom-type record.

    Raises
    ------
    ValueError
        Raised when the silica model contains duplicate atom-type names.
    """
    atomtypes_by_name = {}
    for atomtype in _silica_atomtypes_in_order(silica_topology):
        if atomtype.name in atomtypes_by_name:
            raise ValueError(
                "Silica topology model defines duplicate atomtype name "
                f"{atomtype.name!r}."
            )
        atomtypes_by_name[atomtype.name] = atomtype
    return atomtypes_by_name


@dataclass(frozen=True)
class _StructureAtomRecord:
    """One serialized atom record used by structure writers.

    Parameters
    ----------
    serial : int
        One-based atom serial number in writer order.
    pdb_serial_token : str
        Five-character hybrid-36 atom-serial token for PDB output.
    molecule_index : int
        Zero-based molecule index in the writer output order.
    local_atom_index : int
        Zero-based atom index inside the source molecule.
    residue_name : str
        Full residue identifier of the source molecule.
    pdb_residue_name : str
        Three-character PDB-safe residue alias.
    residue_id : int
        One-based residue identifier in writer order.
    pdb_residue_id_token : str
        Four-character hybrid-36 residue-sequence token for PDB output.
    atom_name : str
        Final atom name written by the structure writer.
    atom_type : str
        Element or atom-type token of the source atom.
    position : tuple[float, float, float]
        Cartesian position in nanometers.
    source_id : int or None
        Optional source atom identifier from the originating pore block.
    pdb_chain_id : str
        Single-character PDB chain identifier.
    cif_entity_id : str
        Entity identifier referenced by ``_atom_site.label_entity_id``.
    cif_asym_id : str
        Asymmetry identifier referenced by ``_atom_site.label_asym_id``.
    cif_label_seq_id : str
        Label sequence identifier referenced by ``_atom_site.label_seq_id``.
    """

    serial: int
    pdb_serial_token: str
    molecule_index: int
    local_atom_index: int
    residue_name: str
    pdb_residue_name: str
    residue_id: int
    pdb_residue_id_token: str
    atom_name: str
    atom_type: str
    position: tuple[float, float, float]
    source_id: int | None
    pdb_chain_id: str
    cif_entity_id: str
    cif_asym_id: str
    cif_label_seq_id: str


@dataclass
class _StructureExportCache:
    """Cached assembled export data for one atom-name mode.

    Parameters
    ----------
    atom_records : list[_StructureAtomRecord]
        Serialized atom metadata in structure-writer order.
    molecule_serials : list[list[int]]
        Atom serial numbers grouped by written molecule.
    residue_alias_records : list[_PdbResidueAliasRecord]
        Full-name to PDB-alias mappings used by the current export.
    entity_records : list[_CifEntityRecord]
        Declared mmCIF entity rows referenced by ``atom_records``.
    struct_asym_records : list[_CifStructAsymRecord]
        Declared mmCIF structural-asymmetry rows referenced by ``atom_records``.
    graph : AssembledStructureGraph or None, optional
        Cached assembled bond graph matching ``atom_records``.
    validation_report : ConnectivityValidationReport or None, optional
        Cached connectivity-validation report for the assembled structure.
    """

    atom_records: list[_StructureAtomRecord]
    molecule_serials: list[list[int]]
    residue_alias_records: list[_PdbResidueAliasRecord]
    entity_records: list[_CifEntityRecord]
    struct_asym_records: list[_CifStructAsymRecord]
    graph: AssembledStructureGraph | None = None
    validation_report: ConnectivityValidationReport | None = None


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
        self._structure_export_cache = {}


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

    def _residue_names_in_order(self):
        """Return residue identifiers in first-appearance order.

        Returns
        -------
        residue_names : list[str]
            Unique residue identifiers encountered while iterating over the
            serialized molecule list in writer order.
        """
        residue_names = []
        seen = set()
        for mol in self._mols:
            residue_name = mol.get_short()
            if residue_name in seen:
                continue
            residue_names.append(residue_name)
            seen.add(residue_name)
        return residue_names

    def _pdb_residue_alias_records(self, residue_names):
        """Return deterministic PDB-safe aliases for residue identifiers.

        Parameters
        ----------
        residue_names : list[str]
            Unique residue identifiers in first-appearance order.

        Returns
        -------
        alias_records : list[_PdbResidueAliasRecord]
            Full-name to three-character alias mappings.

        Raises
        ------
        ValueError
            Raised when more than ``36**2`` colliding long-name aliases would
            be required for the same initial letter.
        """
        alias_records = []
        used_aliases = set()

        for residue_name in residue_names:
            normalized = _normalize_pdb_identifier(residue_name, fallback="UNK")
            alias = None

            if len(normalized) <= 3 and normalized not in used_aliases:
                alias = normalized
            else:
                for candidate in (
                    (normalized[:3] if len(normalized) >= 3 else (normalized + "XXX")[:3]),
                    (normalized[:2] + normalized[-1]) if len(normalized) >= 3 else None,
                    (normalized[0] + normalized[-2:]) if len(normalized) >= 3 else None,
                ):
                    if candidate is None:
                        continue
                    candidate = candidate[:3]
                    if candidate not in used_aliases:
                        alias = candidate
                        break

            if alias is None:
                prefix = normalized[0]
                for suffix_value in range(36 ** 2):
                    candidate = prefix + _encode_pure(
                        _HYBRID36_DIGITS_UPPER,
                        suffix_value,
                        2,
                    )
                    if candidate not in used_aliases:
                        alias = candidate
                        break
                if alias is None:
                    raise ValueError(
                        "Could not assign a unique 3-character PDB residue "
                        f"alias for residue {residue_name!r}."
                    )

            used_aliases.add(alias)
            alias_records.append(
                _PdbResidueAliasRecord(
                    full_name=residue_name,
                    pdb_name=alias,
                )
            )

        return alias_records

    def _pdb_alias_remark_lines(self, residue_alias_records):
        """Return PDB remark lines describing residue-name aliasing.

        Parameters
        ----------
        residue_alias_records : list[_PdbResidueAliasRecord]
            Full-name to PDB-alias mappings for the current export.

        Returns
        -------
        lines : list[str]
            ``REMARK`` lines describing aliases that differ from the original
            residue name.
        """
        lines = []
        for record in residue_alias_records:
            if record.full_name == record.pdb_name:
                continue
            lines.append(
                f"REMARK 250 RESIDUE_ALIAS {record.full_name} -> {record.pdb_name}\n"
            )
        return lines

    def _pdb_atom_name_field(self, atom_name, atom_type):
        """Return one fixed-width PDB atom-name field.

        Parameters
        ----------
        atom_name : str
            Atom name to serialize.
        atom_type : str
            Atom type used to infer the element alignment rule.

        Returns
        -------
        field : str
            Exactly four characters suitable for columns 13-16 of a PDB
            ``ATOM`` or ``HETATM`` record.
        """
        sanitized = _sanitize_pdb_token(atom_name, width=4, fallback="X")
        try:
            element = db.get_element(atom_type)
        except ValueError:
            element = atom_type

        if len(sanitized) == 4:
            return sanitized
        if sanitized[0].isdigit() or len(element) == 2:
            return f"{sanitized:<4s}"
        return f"{sanitized:>4s}"

    def _pdb_cryst1_record(self):
        """Return the CRYST1 record for the current periodic box.

        Returns
        -------
        line : str
            PDB ``CRYST1`` record or an empty string when no periodic box is
            available.
        """
        if not any(self._box):
            return ""

        return (
            "CRYST1"
            f"{self._box[0] * 10:9.3f}"
            f"{self._box[1] * 10:9.3f}"
            f"{self._box[2] * 10:9.3f}"
            f"{90.0:7.2f}"
            f"{90.0:7.2f}"
            f"{90.0:7.2f}"
            " P 1           1\n"
        )

    def _collect_structure_records(self, use_atom_names=False):
        """Collect structure-writer metadata in the current writer order.

        Parameters
        ----------
        use_atom_names : bool, optional
            True to preserve explicit atom names when available. False to
            enumerate atom names from atom types.

        Returns
        -------
        cache : _StructureExportCache
            Structure-export metadata, alias tables, and mmCIF support tables
            in writer order.
        """
        residue_names = self._residue_names_in_order()
        residue_alias_records = self._pdb_residue_alias_records(residue_names)
        pdb_alias_by_name = {
            record.full_name: record.pdb_name
            for record in residue_alias_records
        }
        entity_records = [
            _CifEntityRecord(entity_id=str(entity_index), residue_name=residue_name)
            for entity_index, residue_name in enumerate(residue_names, start=1)
        ]
        entity_id_by_name = {
            record.residue_name: record.entity_id
            for record in entity_records
        }

        atom_records = []
        molecule_serials = []
        struct_asym_records = []
        atom_serial = 1
        residue_serial = 1

        for molecule_index, mol in enumerate(self._mols):
            atom_types = {}
            molecule_serial = []
            residue_marker = None
            residue_identifiers = None

            for local_atom_index, atom in enumerate(mol.get_atom_list()):
                if residue_identifiers is None or atom.get_residue() != residue_marker:
                    if residue_identifiers is not None:
                        residue_serial += 1
                    residue_marker = atom.get_residue()
                    residue_name = mol.get_short()
                    residue_identifiers = _ResidueExportIdentifiers(
                        residue_id=residue_serial,
                        residue_name=residue_name,
                        pdb_residue_name=pdb_alias_by_name[residue_name],
                        pdb_chain_id=_PDB_CHAIN_ID,
                        pdb_residue_id_token=_encode_hybrid36(4, residue_serial),
                        cif_entity_id=entity_id_by_name[residue_name],
                        cif_asym_id=f"A{residue_serial}",
                        cif_label_seq_id=_CIF_LABEL_SEQ_ID,
                    )
                    struct_asym_records.append(
                        _CifStructAsymRecord(
                            asym_id=residue_identifiers.cif_asym_id,
                            entity_id=residue_identifiers.cif_entity_id,
                            residue_id=residue_identifiers.residue_id,
                            residue_name=residue_identifiers.residue_name,
                        )
                    )

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
                        pdb_serial_token=_encode_hybrid36(5, atom_serial),
                        molecule_index=molecule_index,
                        local_atom_index=local_atom_index,
                        residue_name=residue_identifiers.residue_name,
                        pdb_residue_name=residue_identifiers.pdb_residue_name,
                        residue_id=residue_identifiers.residue_id,
                        pdb_residue_id_token=residue_identifiers.pdb_residue_id_token,
                        atom_name=atom_name,
                        atom_type=atom_type,
                        position=tuple(atom.get_pos()),
                        source_id=atom.get_source_id(),
                        pdb_chain_id=residue_identifiers.pdb_chain_id,
                        cif_entity_id=residue_identifiers.cif_entity_id,
                        cif_asym_id=residue_identifiers.cif_asym_id,
                        cif_label_seq_id=residue_identifiers.cif_label_seq_id,
                    )
                )
                molecule_serial.append(atom_serial)

                atom_serial += 1
                atom_types[atom_type] = atom_types[atom_type] + 1 if atom_types[atom_type] < 99 else 1

            molecule_serials.append(molecule_serial)
            if residue_identifiers is not None:
                residue_serial += 1

        return _StructureExportCache(
            atom_records=atom_records,
            molecule_serials=molecule_serials,
            residue_alias_records=residue_alias_records,
            entity_records=entity_records,
            struct_asym_records=struct_asym_records,
        )

    def _export_cache(self, use_atom_names=False):
        """Return cached structure-export data for one atom-name mode.

        Parameters
        ----------
        use_atom_names : bool, optional
            True to preserve explicit atom names when available. False to
            enumerate atom names from atom types.

        Returns
        -------
        cache : _StructureExportCache
            Cached structure records and lazily populated graph/validation
            data for the requested atom-name mode.
        """
        cache = self._structure_export_cache.get(use_atom_names)
        if cache is not None:
            return cache

        cache = self._collect_structure_records(use_atom_names)
        self._structure_export_cache[use_atom_names] = cache
        return cache

    def _export_graph(self, use_atom_names=False):
        """Return the cached assembled bond graph for one atom-name mode.

        Parameters
        ----------
        use_atom_names : bool, optional
            True to preserve explicit atom names when available. False to
            enumerate atom names from atom types.

        Returns
        -------
        graph : AssembledStructureGraph
            Assembled graph matching the current structure-writer order.
        """
        cache = self._export_cache(use_atom_names)
        if cache.graph is None:
            cache.graph = self._assembled_structure_graph(
                cache.atom_records,
                cache.molecule_serials,
            )
        return cache.graph

    def _validation_report(self, use_atom_names=False):
        """Return the cached connectivity-validation report.

        Parameters
        ----------
        use_atom_names : bool, optional
            True to preserve explicit atom names when available. False to
            enumerate atom names from atom types.

        Returns
        -------
        report : ConnectivityValidationReport
            Cached validation report for the assembled structure.
        """
        cache = self._export_cache(use_atom_names)
        if cache.validation_report is None:
            graph = self._export_graph(use_atom_names)
            cache.validation_report = ConnectivityValidationReport(
                atom_count=len(cache.atom_records),
                bond_count=len(graph.bonds),
                findings=self._connectivity_validation_findings(cache.atom_records, graph),
            )
        return cache.validation_report

    def _validate_pdb_hybrid36_limits(self, atom_records):
        """Validate that PDB identifiers fit inside hybrid-36 fields.

        Parameters
        ----------
        atom_records : list[_StructureAtomRecord]
            Serialized atom metadata in file order.

        Raises
        ------
        ValueError
            Raised when any atom serial or residue identifier exceeds the
            hybrid-36 representable range of the corresponding PDB field.
        """
        if not atom_records:
            return

        max_atom_serial = max(record.serial for record in atom_records)
        max_residue_serial = max(record.residue_id for record in atom_records)

        if max_atom_serial > _hybrid36_max_value(5):
            raise ValueError(
                "PDB atom serials exceed the hybrid-36 range "
                f"(max atom serial={max_atom_serial}, max {_hybrid36_max_value(5)})."
            )
        if max_residue_serial > _hybrid36_max_value(4):
            raise ValueError(
                "PDB residue ids exceed the hybrid-36 range "
                f"(max residue id={max_residue_serial}, max {_hybrid36_max_value(4)})."
            )

    def _write_cif_entity_loop(self, file_out, entity_records):
        """Write the mmCIF entity loop for the current export.

        Parameters
        ----------
        file_out : TextIO
            Open output stream.
        entity_records : list[_CifEntityRecord]
            Entity rows declared for the current export.
        """
        if not entity_records:
            return

        file_out.write("loop_\n")
        file_out.write("_entity.id\n")
        file_out.write("_entity.type\n")
        file_out.write("_entity.pdbx_description\n")
        for record in entity_records:
            file_out.write(
                " ".join(
                    (
                        _cif_token(record.entity_id),
                        _cif_token(record.entity_type),
                        _cif_token(record.residue_name),
                    )
                )
                + "\n"
            )
        file_out.write("#\n")

    def _write_cif_struct_asym_loop(self, file_out, struct_asym_records):
        """Write the mmCIF structural-asymmetry loop for the export.

        Parameters
        ----------
        file_out : TextIO
            Open output stream.
        struct_asym_records : list[_CifStructAsymRecord]
            Asymmetry rows declared for the current export.
        """
        if not struct_asym_records:
            return

        file_out.write("loop_\n")
        file_out.write("_struct_asym.id\n")
        file_out.write("_struct_asym.entity_id\n")
        for record in struct_asym_records:
            file_out.write(
                f"{_cif_token(record.asym_id)} {_cif_token(record.entity_id)}\n"
            )
        file_out.write("#\n")

    def _assembled_structure_graph(self, atom_records, molecule_serials):
        """Build a bonded graph for the serialized structure.

        Parameters
        ----------
        atom_records : list[_StructureAtomRecord]
            All atom records written to the current structure file.
        molecule_serials : list[list[int]]
            Atom serial numbers grouped per written molecule.

        Returns
        -------
        graph : AssembledStructureGraph
            Assembled bond graph in writer atom order.
        """
        bonds = []

        source_serials = {
            record.source_id: record.serial
            for record in atom_records
            if record.source_id is not None
        }
        if isinstance(self._inp, Pore):
            inserted_bridge_oxygen = {
                record.atom_id
                for record in self._inp.get_surface_edit_history()
                if record.reason == "inserted_bridge_oxygen"
            }
            for atom_id, props in self._inp._matrix.get_matrix().items():
                if atom_id not in source_serials:
                    continue
                for neighbor_id in props["atoms"]:
                    if neighbor_id not in source_serials or neighbor_id < atom_id:
                        continue
                    provenance = (
                        "siloxane_bridge"
                        if atom_id in inserted_bridge_oxygen or neighbor_id in inserted_bridge_oxygen
                        else "scaffold"
                    )
                    bonds.append(
                        GraphBond(
                            source_serials[atom_id],
                            source_serials[neighbor_id],
                            provenance,
                        )
                    )

        for molecule_index, serials in enumerate(molecule_serials):
            mol = self._mols[molecule_index]
            explicit_bonds = mol.get_bonds()
            for atom_a, atom_b in explicit_bonds:
                bonds.append(
                    GraphBond(serials[atom_a], serials[atom_b], "ligand_explicit")
                )
            if not explicit_bonds:
                for atom_a, atom_b in mol.infer_bonds():
                    bonds.append(
                        GraphBond(serials[atom_a], serials[atom_b], "ligand_inferred")
                    )

        if isinstance(self._inp, Pore):
            molecule_serial_map = {
                id(mol): serials
                for mol, serials in zip(self._mols, molecule_serials)
            }
            for record in self._inp.get_attachment_records():
                serials = molecule_serial_map.get(id(record.molecule))
                if not serials:
                    continue
                mount_serial = serials[record.mount_atom_local_id]
                for oxygen_source_id in record.scaffold_oxygen_source_ids:
                    if oxygen_source_id in source_serials:
                        bonds.append(
                            GraphBond(
                                mount_serial,
                                source_serials[oxygen_source_id],
                                "graft_junction",
                            )
                        )

        return AssembledStructureGraph.from_bonds(
            (record.serial for record in atom_records),
            bonds,
        )

    def assembled_graph(self, use_atom_names=False):
        """Return the assembled bonded graph in structure-writer atom order.

        Parameters
        ----------
        use_atom_names : bool, optional
            True to preserve explicit atom names when available while matching
            the graph to the same atom order used by structure writers.

        Returns
        -------
        graph : AssembledStructureGraph
            Bond graph and derived angle hooks for the current object.
        """
        return self._export_graph(use_atom_names)

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
                    + _encode_hybrid36(5, serial)
                    + "".join(_encode_hybrid36(5, neighbor) for neighbor in chunk)
                    + "\n"
                )

    def _write_cif_struct_conn_loop(self, file_out, atom_records, graph):
        """Write an mmCIF bond loop for the assembled bonded structure.

        Parameters
        ----------
        file_out : TextIO
            Open output stream.
        atom_records : list[_StructureAtomRecord]
            Serialized atom metadata in file order.
        graph : AssembledStructureGraph
            Assembled graph matching ``atom_records``.
        """
        if not graph.bonds:
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

        for conn_index, bond in enumerate(graph.bonds, start=1):
            record_a = record_by_serial[bond.atom_a]
            record_b = record_by_serial[bond.atom_b]
            file_out.write(
                " ".join(
                    (
                        _cif_token(f"conn{conn_index}"),
                        "covale",
                        _cif_token(record_a.cif_asym_id),
                        _cif_token(record_a.residue_name),
                        _cif_token(record_a.cif_label_seq_id),
                        _cif_token(record_a.atom_name),
                        _cif_token(record_b.cif_asym_id),
                        _cif_token(record_b.residue_name),
                        _cif_token(record_b.cif_label_seq_id),
                        _cif_token(record_b.atom_name),
                    )
                )
                + "\n"
            )

    def _connectivity_validation_neighbors(self, graph):
        """Build an adjacency map for one assembled graph.

        Parameters
        ----------
        graph : AssembledStructureGraph
            Graph whose neighbor lists should be assembled.

        Returns
        -------
        neighbors : dict[int, set[int]]
            One-based neighbor ids grouped by central atom id.
        """
        neighbors = {atom_id: set() for atom_id in graph.atom_ids}
        for bond in graph.bonds:
            if bond.atom_a in neighbors:
                neighbors[bond.atom_a].add(bond.atom_b)
            if bond.atom_b in neighbors:
                neighbors[bond.atom_b].add(bond.atom_a)
        return neighbors

    def _connectivity_validation_findings(self, atom_records, graph):
        """Collect connectivity-validation findings for one assembled graph.

        Parameters
        ----------
        atom_records : list[_StructureAtomRecord]
            Serialized atom metadata in writer order.
        graph : AssembledStructureGraph
            Assembled bond graph matching ``atom_records``.

        Returns
        -------
        findings : tuple[ConnectivityValidationFinding, ...]
            Sorted validation findings for the assembled structure.
        """
        record_by_serial = {record.serial: record for record in atom_records}
        neighbors = self._connectivity_validation_neighbors(graph)
        findings = {}
        is_finalized_pore = not isinstance(self._inp, Pore) or self._inp.is_finalized()

        allowed_degrees = {
            "H": {1},
            "B": {3, 4},
            "C": {1, 2, 3, 4},
            "N": {1, 2, 3, 4},
            "O": {1, 2},
            "F": {1},
            "P": {1, 2, 3, 4, 5},
            "S": {1, 2, 3, 4, 5, 6},
            "Cl": {1},
            "Br": {1},
            "I": {1},
            "Si": {1, 2, 3, 4},
        }

        def add_finding(code, message, atom_ids, is_error=True):
            atom_ids = tuple(atom_ids)
            key = (code, atom_ids)
            if key in findings:
                return
            findings[key] = ConnectivityValidationFinding(
                code=code,
                message=message,
                atom_ids=atom_ids,
                atom_types=tuple(record_by_serial[atom_id].atom_type for atom_id in atom_ids if atom_id in record_by_serial),
                residue_shorts=tuple(record_by_serial[atom_id].residue_name for atom_id in atom_ids if atom_id in record_by_serial),
                degrees=tuple(len(neighbors.get(atom_id, ())) for atom_id in atom_ids),
                is_error=is_error,
            )

        for atom_id in graph.atom_ids:
            if atom_id not in record_by_serial:
                continue
            record = record_by_serial[atom_id]
            degree = len(neighbors.get(atom_id, ()))
            is_pre_finalized_scaffold = (
                isinstance(self._inp, Pore)
                and not is_finalized_pore
                and record.residue_name in {"OM", "SI"}
            )
            try:
                element = db.get_element(record.atom_type)
            except ValueError:
                add_finding(
                    "unknown_element",
                    f"Atom {atom_id} has unsupported atom type '{record.atom_type}' for connectivity validation.",
                    (atom_id,),
                )
                continue

            if (
                not is_pre_finalized_scaffold
                and element in allowed_degrees
                and degree not in allowed_degrees[element]
            ):
                add_finding(
                    "unexpected_degree",
                    f"Atom {atom_id} ({record.atom_type}) has degree {degree}, which is outside the supported range for {element}.",
                    (atom_id,),
                )

            neighbor_elements = []
            for neighbor_id in sorted(neighbors.get(atom_id, ())):
                try:
                    neighbor_elements.append(db.get_element(record_by_serial[neighbor_id].atom_type))
                except (KeyError, ValueError):
                    neighbor_elements.append(record_by_serial[neighbor_id].atom_type)

            if element == "H":
                if degree != 1:
                    add_finding(
                        "hydrogen_degree",
                        f"Hydrogen atom {atom_id} must have degree 1, found {degree}.",
                        (atom_id,),
                    )
                elif neighbor_elements and neighbor_elements[0] == "H":
                    add_finding(
                        "hydrogen_neighbor",
                        f"Hydrogen atom {atom_id} is bonded to another hydrogen atom.",
                        (atom_id, next(iter(sorted(neighbors[atom_id])))),
                    )

            if is_pre_finalized_scaffold:
                continue

            if record.residue_name == "OM":
                if degree != 2 or sorted(neighbor_elements) != ["Si", "Si"]:
                    add_finding(
                        "framework_oxygen_environment",
                        f"Framework oxygen atom {atom_id} must be bonded to exactly two silicon atoms.",
                        (atom_id, *sorted(neighbors.get(atom_id, ()))),
                    )

            if record.residue_name == "SI":
                if degree not in {4}:
                    add_finding(
                        "framework_silicon_environment",
                        f"Framework silicon atom {atom_id} must have degree 4 after export.",
                        (atom_id,),
                    )

        for bond in graph.bonds:
            if bond.atom_a not in record_by_serial or bond.atom_b not in record_by_serial:
                continue
            try:
                element_a = db.get_element(record_by_serial[bond.atom_a].atom_type)
                element_b = db.get_element(record_by_serial[bond.atom_b].atom_type)
            except ValueError:
                continue

            if bond.provenance in {"scaffold", "siloxane_bridge", "graft_junction"}:
                if sorted((element_a, element_b)) != ["O", "Si"]:
                    add_finding(
                        "invalid_silica_bond",
                        f"{bond.provenance.replace('_', ' ')} bond {bond.atom_a}-{bond.atom_b} must connect silicon and oxygen atoms.",
                        (bond.atom_a, bond.atom_b),
                    )

        return tuple(sorted(findings.values(), key=lambda finding: (finding.code, finding.atom_ids)))

    def validate_connectivity(self, use_atom_names=False):
        """Validate the assembled bond graph against simple chemistry rules.

        Parameters
        ----------
        use_atom_names : bool, optional
            True to preserve explicit atom names while matching the validation
            atom order to the same order used by structure writers.

        Returns
        -------
        report : ConnectivityValidationReport
            Structured validation report for the current assembled structure.
            Full framework-oxygen checks are applied only to finalized pore
            states.
        """
        return self._validation_report(use_atom_names)

    def _handle_connectivity_validation(self, use_atom_names, validate_connectivity):
        """Validate one structure before writing and handle the chosen mode.

        Parameters
        ----------
        use_atom_names : bool
            Forwarded atom-name handling mode used for structure serialization.
        validate_connectivity : str
            Validation mode: ``"off"``, ``"warn"``, or ``"strict"``.

        Returns
        -------
        report : ConnectivityValidationReport or None
            Validation report when the mode is not ``"off"``, otherwise
            ``None``.

        Raises
        ------
        ValueError
            Raised when ``validate_connectivity`` is unsupported or when the
            mode is ``"strict"`` and the validation report contains errors.
        """
        if validate_connectivity not in {"off", "warn", "strict"}:
            raise ValueError(
                "Unsupported connectivity validation mode. "
                "Expected one of: ['off', 'strict', 'warn']."
            )

        if validate_connectivity == "off":
            return None

        report = self.validate_connectivity(use_atom_names=use_atom_names)
        if report.is_valid:
            return report

        summary = (
            "Connectivity validation found "
            f"{report.error_count} error(s) and {report.warning_count} warning(s)."
        )
        preview = "; ".join(
            finding.message for finding in report.findings[:3]
        )
        message = summary + (" " + preview if preview else "")

        if validate_connectivity == "strict":
            raise ValueError(message)

        warnings.warn(message, UserWarning, stacklevel=3)
        return report

    def cif(
        self,
        name="",
        use_atom_names=False,
        write_bonds=True,
        validate_connectivity="warn",
    ):
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
            for the full assembled bond graph, including silica scaffold,
            siloxane bridges, ligand-internal bonds, and graft junctions.
            Full residue names are preserved in mmCIF, and the writer emits
            matching ``_entity`` and ``_struct_asym`` loops so that
            ``_atom_site.label_entity_id`` and ``_atom_site.label_asym_id``
            reference declared rows.
        validate_connectivity : str, optional
            Connectivity validation mode: ``"off"``, ``"warn"``, or
            ``"strict"``. The default warns on invalid assembled local
            chemistry before writing the file.
        """
        self._handle_connectivity_validation(use_atom_names, validate_connectivity)
        link = self._link
        link += name if name else self._name + ".cif"
        cache = self._export_cache(use_atom_names)
        atom_records = cache.atom_records
        entity_records = cache.entity_records
        struct_asym_records = cache.struct_asym_records
        graph = self._export_graph(use_atom_names)
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
            self._write_cif_entity_loop(file_out, entity_records)
            self._write_cif_struct_asym_loop(file_out, struct_asym_records)
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
                            _cif_token(record.serial),
                            _cif_token(db.get_element(record.atom_type)),
                            _cif_token(record.atom_name),
                            ".",
                            _cif_token(record.residue_name),
                            _cif_token(record.cif_asym_id),
                            _cif_token(record.cif_entity_id),
                            _cif_token(record.cif_label_seq_id),
                            "?",
                            _cif_token(f"{record.position[0] * 10:.3f}"),
                            _cif_token(f"{record.position[1] * 10:.3f}"),
                            _cif_token(f"{record.position[2] * 10:.3f}"),
                            "1.00",
                            "0.00",
                            "?",
                            _cif_token(record.residue_id),
                            _cif_token(record.residue_name),
                            _cif_token(record.pdb_chain_id),
                            _cif_token(record.atom_name),
                            "1",
                        ]
                    )
                    + "\n"
                )

            if write_bonds:
                self._write_cif_struct_conn_loop(file_out, atom_records, graph)

            file_out.write("#\n")

    def pdb(
        self,
        name="",
        use_atom_names=False,
        write_conect=True,
        validate_connectivity="warn",
    ):
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
            ``CONECT`` records for the full assembled bond graph, including
            silica scaffold, siloxane bridges, ligand-internal bonds, and
            graft junctions. Residue names longer than three characters are
            written as deterministic three-character aliases, atom serials and
            residue ids automatically switch to hybrid-36 on overflow, and a
            ``CRYST1`` record is emitted when periodic box lengths are known.
        validate_connectivity : str, optional
            Connectivity validation mode: ``"off"``, ``"warn"``, or
            ``"strict"``. The default warns on invalid assembled local
            chemistry before writing the file.
        """
        self._handle_connectivity_validation(use_atom_names, validate_connectivity)
        # Initialize
        link = self._link
        link += name if name else self._name+".pdb"
        cache = self._export_cache(use_atom_names)
        atom_records = cache.atom_records
        residue_alias_records = cache.residue_alias_records
        graph = self._export_graph(use_atom_names)
        self._validate_pdb_hybrid36_limits(atom_records)

        # Open file
        with open(link, "w") as file_out:
            for line in self._pdb_alias_remark_lines(residue_alias_records):
                file_out.write(line)

            cryst1_record = self._pdb_cryst1_record()
            if cryst1_record:
                file_out.write(cryst1_record)

            for record in atom_records:
                element = _sanitize_pdb_token(
                    db.get_element(record.atom_type),
                    width=2,
                    fallback="X",
                )
                out_string = "HETATM"                       #  1- 6 (6)    Record name
                out_string += record.pdb_serial_token      #  7-11 (5)    Atom serial number
                out_string += " "                          # 12    (1)    -
                out_string += self._pdb_atom_name_field(
                    record.atom_name,
                    record.atom_type,
                )                                          # 13-16 (4)    Atom name
                out_string += " "                          # 17    (1)    Alternate location indicator
                out_string += f"{record.pdb_residue_name:>3s}"# 18-20 (3)  Residue name
                out_string += " "                          # 21    (1)    -
                out_string += record.pdb_chain_id          # 22    (1)    Chain identifier
                out_string += record.pdb_residue_id_token  # 23-26 (4)    Residue sequence number
                out_string += " "                          # 27    (1)    Code for insertion of residues
                out_string += "   "                        # 28-30 (3)    -
                for coord in record.position:              # 31-54 (3*8)  Coordinates
                    out_string += f"{coord*10:8.3f}"
                out_string += f"{1:6.2f}"                  # 55-60 (6)    Occupancy
                out_string += f"{0:6.2f}"                  # 61-66 (6)    Temperature factor
                out_string += "          "                 # 67-76 (10)   -
                out_string += f"{element:>2s}"             # 77-78 (2)    Element symbol
                out_string += "  "                         # 79-80 (2)    Charge on the atom

                file_out.write(out_string+"\n")

            if write_conect:
                self._write_pdb_conect_records(
                    file_out,
                    [(bond.atom_a, bond.atom_b) for bond in graph.bonds],
                )

            # End statement
            file_out.write("TER\nEND\n")

    def gro(self, name="", use_atom_names=False, validate_connectivity="warn"):
        """Write the current structure in GROMACS GRO format.

        Parameters
        ----------
        name : str, optional
            Output filename. Defaults to ``<name>.gro``.
        use_atom_names : bool, optional
            True to preserve explicit atom names when available. False to
            enumerate atom names from atom types.
        validate_connectivity : str, optional
            Connectivity validation mode: ``"off"``, ``"warn"``, or
            ``"strict"``. The default warns on invalid assembled local
            chemistry before writing the file.
        """
        self._handle_connectivity_validation(use_atom_names, validate_connectivity)
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

    def full_slit_topology(
        self,
        name="",
        base_ligand_short="",
        silane_topology_config=None,
        silica_topology=None,
    ):
        """Write one self-contained full-slit GROMACS topology when supported.

        Parameters
        ----------
        name : str, optional
            Output topology filename. Defaults to ``<name>.top``.
        base_ligand_short : str, optional
            Base short name of the configured silane ligand, for example
            ``"TMS"``. Geminal variants are expected to use the same short
            name with a trailing ``"G"``.
        silane_topology_config : object or None, optional
            Optional silane topology configuration carrying ``itp_path``,
            ``moleculetype_name``, and ``junction_parameters`` attributes.
        silica_topology : SilicaTopologyModel or None, optional
            Resolved editable silica topology model used for scaffold and
            graft-junction terms. When omitted, the package defaults are used.

        Returns
        -------
        wrote_full_topology : bool
            True when a full-slit topology was written. False when the current
            finalized pore contains unsupported non-silica residues and no
            matching ligand topology input was available.

        Raises
        ------
        TypeError
            Raised when the wrapped input is not a :class:`porems.pore.Pore`.
        ValueError
            Raised when the pore is not finalized or when the available
            topology information is internally inconsistent.
        """
        if not isinstance(self._inp, Pore):
            raise TypeError("Store: Unsupported input type for full slit topology creation...")
        if not self._inp.is_finalized():
            raise ValueError("Store: Full slit topology export requires a finalized pore.")
        if silica_topology is None:
            silica_topology = default_silica_topology()
        elif not isinstance(silica_topology, SilicaTopologyModel):
            raise TypeError(
                "Store: silica_topology must be a SilicaTopologyModel instance "
                "or None."
            )

        bundle = None
        ligand_shorts = set()
        bundle_atoms_by_index = {}
        bundle_atomtypes_by_name = {}
        if silane_topology_config is not None:
            bundle = parse_flat_itp(
                silane_topology_config.itp_path,
                moleculetype_name=getattr(silane_topology_config, "moleculetype_name", ""),
            )
            if bundle.moleculetype.nrexcl != _FULL_SLIT_NREXCL:
                raise ValueError(
                    "Full slit topology export currently requires ligand "
                    f"nrexcl={_FULL_SLIT_NREXCL}. Parsed "
                    f"{bundle.moleculetype.nrexcl} from "
                    f"{silane_topology_config.itp_path!r}."
                )
            bundle_atoms_by_index = {
                atom.index: atom
                for atom in bundle.moleculetype.atoms
            }
            bundle_atomtypes_by_name = {
                atomtype.name: atomtype
                for atomtype in bundle.atomtypes
            }
            if base_ligand_short:
                ligand_shorts = {base_ligand_short, base_ligand_short + "G"}

        cache = self._collect_structure_records(use_atom_names=True)
        graph = self._assembled_structure_graph(
            cache.atom_records,
            cache.molecule_serials,
        )
        record_by_serial = {
            record.serial: record
            for record in cache.atom_records
        }

        unsupported_residues = sorted(
            {
                record.residue_name
                for record in cache.atom_records
                if record.residue_name not in {"OM", "SI", "SL", "SLG"}
                and record.residue_name not in ligand_shorts
            }
        )
        if unsupported_residues:
            return False

        silica_atomtypes = _silica_atomtypes_in_order(silica_topology)
        silica_atomtype_by_name = _silica_atomtype_lookup(silica_topology)

        def assignment_fields(assignment):
            atomtype = silica_atomtype_by_name.get(assignment.atom_type_name)
            if atomtype is None:
                raise ValueError(
                    "Silica topology assignment references unknown atomtype "
                    f"{assignment.atom_type_name!r}."
                )
            return assignment.atom_type_name, assignment.charge, assignment.mass

        def silica_atom_fields(record):
            if record.residue_name == "OM":
                return assignment_fields(silica_topology.atom_assignments.framework_oxygen)
            if record.residue_name == "SI":
                return assignment_fields(silica_topology.atom_assignments.framework_silicon)
            if record.residue_name == "SL":
                if record.atom_type == "Si":
                    return assignment_fields(silica_topology.atom_assignments.silanol_silicon)
                if record.atom_type == "O":
                    return assignment_fields(silica_topology.atom_assignments.silanol_oxygen)
                if record.atom_type == "H":
                    return assignment_fields(silica_topology.atom_assignments.silanol_hydrogen)
            if record.residue_name == "SLG":
                if record.atom_type == "Si":
                    return assignment_fields(silica_topology.atom_assignments.geminal_silicon)
                if record.atom_type == "O":
                    return assignment_fields(silica_topology.atom_assignments.geminal_oxygen)
                if record.atom_type == "H":
                    return assignment_fields(silica_topology.atom_assignments.geminal_hydrogen)
            raise ValueError(
                "Unsupported silica residue/atom combination for full slit "
                f"topology export: {(record.residue_name, record.atom_type, record.atom_name)!r}."
            )

        def ligand_atom_fields(record):
            if bundle is None or record.residue_name not in ligand_shorts:
                return None

            if bundle.has_atom_name(record.atom_name):
                bundle_atom = bundle.atom_by_name(record.atom_name)
                mass = bundle_atom.mass
                if mass is None:
                    atomtype = bundle_atomtypes_by_name.get(bundle_atom.atom_type)
                    if atomtype is None:
                        raise ValueError(
                            "Bundle atom mass is missing and no matching "
                            f"[ atomtypes ] definition exists for "
                            f"{bundle_atom.atom_type!r} in {bundle.source_path!r}."
                        )
                    mass = atomtype.mass
                return bundle_atom.atom_type, bundle_atom.charge, mass

            if record.residue_name == base_ligand_short + "G":
                if record.atom_type == "O":
                    return assignment_fields(silica_topology.atom_assignments.geminal_oxygen)
                if record.atom_type == "H":
                    return assignment_fields(silica_topology.atom_assignments.geminal_hydrogen)

            raise ValueError(
                "Finalized ligand residue contains atoms that cannot be mapped "
                "to the supplied base ligand topology bundle: "
                f"{(record.residue_name, record.atom_name)!r}."
            )

        atoms = []
        atomtype_order = list(silica_atomtypes)
        atomtype_by_name = {
            atomtype.name: atomtype
            for atomtype in atomtype_order
        }
        if bundle is not None:
            for atomtype in bundle.atomtypes:
                existing = atomtype_by_name.get(atomtype.name)
                if existing is not None:
                    if not existing.is_compatible_with(atomtype):
                        raise ValueError(
                            "Conflicting duplicate atomtype definition for "
                            f"{atomtype.name!r} between the internal silica "
                            "model and the supplied ligand topology bundle."
                        )
                    continue
                atomtype_by_name[atomtype.name] = atomtype
                atomtype_order.append(atomtype)

        for record in cache.atom_records:
            atom_fields = ligand_atom_fields(record)
            if atom_fields is None:
                atom_fields = silica_atom_fields(record)
            atom_type_name, charge_token, mass_token = atom_fields
            atoms.append(
                GromacsAtom(
                    index=record.serial,
                    atom_type=atom_type_name,
                    residue_number=record.residue_id,
                    residue_name=record.residue_name,
                    atom_name=record.atom_name,
                    charge_group=record.serial,
                    charge=_format_decimal_token(charge_token),
                    mass=_format_decimal_token(mass_token, places=4),
                )
            )

        bonds = []
        for bond in graph.bonds:
            record_a = record_by_serial[bond.atom_a]
            record_b = record_by_serial[bond.atom_b]
            bond_parameters = None

            if (
                bundle is not None
                and record_a.molecule_index == record_b.molecule_index
                and record_a.residue_name in ligand_shorts
                and record_b.residue_name in ligand_shorts
            ):
                bundle_bond = bundle.bond_by_names(
                    record_a.atom_name,
                    record_b.atom_name,
                )
                if bundle_bond is not None:
                    bond_parameters = bundle_bond.parameters

            if bond_parameters is None:
                elements = sorted(
                    (
                        db.get_element(record_a.atom_type),
                        db.get_element(record_b.atom_type),
                    )
                )
                if elements == ["H", "O"]:
                    bond_parameters = (
                        silica_topology
                        .bond_terms
                        .silanol_o_h
                        .to_gromacs_parameters()
                    )
                elif elements == ["O", "Si"]:
                    if bond.provenance == "graft_junction":
                        bond_parameters = (
                            silica_topology
                            .bond_terms
                            .graft_mount_scaffold_si_o
                            .to_gromacs_parameters()
                        )
                    else:
                        bond_parameters = (
                            silica_topology
                            .bond_terms
                            .framework_si_o
                            .to_gromacs_parameters()
                        )
                else:
                    raise ValueError(
                        "Unsupported bond environment for full slit topology "
                        f"export: {(record_a.residue_name, record_a.atom_name, record_b.residue_name, record_b.atom_name)!r}."
                    )

            bonds.append(
                GromacsBond(
                    atom_a=bond.atom_a,
                    atom_b=bond.atom_b,
                    parameters=bond_parameters,
                )
            )

        angles = []
        for angle in graph.angles:
            record_a = record_by_serial[angle.atom_a]
            record_b = record_by_serial[angle.atom_b]
            record_c = record_by_serial[angle.atom_c]
            angle_parameters = None

            if (
                bundle is not None
                and record_a.molecule_index == record_b.molecule_index == record_c.molecule_index
                and record_a.residue_name in ligand_shorts
                and record_b.residue_name in ligand_shorts
                and record_c.residue_name in ligand_shorts
            ):
                bundle_angle = bundle.angle_by_names(
                    record_a.atom_name,
                    record_b.atom_name,
                    record_c.atom_name,
                )
                if bundle_angle is not None:
                    angle_parameters = bundle_angle.parameters

            if angle_parameters is None:
                center_element = db.get_element(record_b.atom_type)
                outer_elements = sorted(
                    (
                        db.get_element(record_a.atom_type),
                        db.get_element(record_c.atom_type),
                    )
                )

                if center_element == "O" and outer_elements == ["H", "Si"]:
                    angle_parameters = (
                        silica_topology
                        .angle_terms
                        .silanol_si_o_h
                        .to_gromacs_parameters()
                    )
                elif center_element == "O" and outer_elements == ["Si", "Si"]:
                    if (
                        record_b.residue_name == "OM"
                        and (
                            (record_a.residue_name == "SI" and record_c.residue_name in ligand_shorts)
                            or (record_c.residue_name == "SI" and record_a.residue_name in ligand_shorts)
                        )
                    ):
                        angle_parameters = (
                            silica_topology
                            .angle_terms
                            .graft_scaffold_si_scaffold_o_mount
                            .to_gromacs_parameters()
                        )
                    else:
                        angle_parameters = (
                            silica_topology
                            .angle_terms
                            .framework_si_o_si
                            .to_gromacs_parameters()
                        )
                elif center_element == "Si" and outer_elements == ["O", "O"]:
                    if record_b.residue_name in ligand_shorts:
                        angle_parameters = (
                            silica_topology
                            .angle_terms
                            .graft_oxygen_mount_oxygen
                            .to_gromacs_parameters()
                        )
                    else:
                        angle_parameters = (
                            silica_topology
                            .angle_terms
                            .silanol_o_si_o
                            .to_gromacs_parameters()
                        )
                else:
                    raise ValueError(
                        "Unsupported angle environment for full slit topology "
                        f"export: {(record_a.residue_name, record_a.atom_name, record_b.residue_name, record_b.atom_name, record_c.residue_name, record_c.atom_name)!r}."
                    )

            angles.append(
                GromacsAngle(
                    atom_a=angle.atom_a,
                    atom_b=angle.atom_b,
                    atom_c=angle.atom_c,
                    parameters=angle_parameters,
                )
            )

        pairs = []
        dihedrals = []
        if bundle is not None:
            for mol, serials in zip(self._mols, cache.molecule_serials):
                if mol.get_short() not in ligand_shorts:
                    continue

                local_name_by_serial = {
                    serial: record_by_serial[serial].atom_name
                    for serial in serials
                }
                serial_by_bundle_name = {
                    atom_name: serial
                    for serial, atom_name in local_name_by_serial.items()
                    if bundle.has_atom_name(atom_name)
                }
                missing_bundle_names = sorted(
                    name
                    for name in bundle.atom_index_by_name
                    if name not in serial_by_bundle_name
                )
                if missing_bundle_names:
                    raise ValueError(
                        "Finalized ligand molecule is missing atom names "
                        f"required by the supplied topology bundle: "
                        f"{missing_bundle_names}."
                    )

                for pair in bundle.moleculetype.pairs:
                    atom_name_a = bundle_atoms_by_index[pair.atom_a].atom_name
                    atom_name_b = bundle_atoms_by_index[pair.atom_b].atom_name
                    pairs.append(
                        GromacsPair(
                            atom_a=serial_by_bundle_name[atom_name_a],
                            atom_b=serial_by_bundle_name[atom_name_b],
                            function=pair.function,
                            parameters=pair.parameters,
                        )
                    )

                for dihedral in bundle.moleculetype.dihedrals:
                    atom_name_a = bundle_atoms_by_index[dihedral.atom_a].atom_name
                    atom_name_b = bundle_atoms_by_index[dihedral.atom_b].atom_name
                    atom_name_c = bundle_atoms_by_index[dihedral.atom_c].atom_name
                    atom_name_d = bundle_atoms_by_index[dihedral.atom_d].atom_name
                    dihedrals.append(
                        GromacsDihedral(
                            atom_a=serial_by_bundle_name[atom_name_a],
                            atom_b=serial_by_bundle_name[atom_name_b],
                            atom_c=serial_by_bundle_name[atom_name_c],
                            atom_d=serial_by_bundle_name[atom_name_d],
                            function=dihedral.function,
                            parameters=dihedral.parameters,
                        )
                    )

        top_filename = name if name else self._name + ".top"
        if not top_filename.endswith(".top"):
            top_filename = top_filename + ".top"
        itp_filename = os.path.splitext(top_filename)[0] + ".itp"
        top_link = self._link + top_filename
        itp_link = self._link + os.path.basename(itp_filename)

        molecule_name = _sanitize_gromacs_identifier(
            os.path.splitext(os.path.basename(itp_filename))[0],
            fallback="SLIT",
        )
        molecule_type = GromacsMoleculeType(
            name=molecule_name,
            nrexcl=_FULL_SLIT_NREXCL,
            atoms=tuple(atoms),
            bonds=tuple(sorted(bonds, key=lambda item: (item.atom_a, item.atom_b))),
            pairs=tuple(sorted(pairs, key=lambda item: (item.atom_a, item.atom_b, item.function))),
            angles=tuple(sorted(angles, key=lambda item: (item.atom_b, item.atom_a, item.atom_c))),
            dihedrals=tuple(
                sorted(
                    dihedrals,
                    key=lambda item: (item.atom_b, item.atom_c, item.atom_a, item.atom_d, item.function),
                )
            ),
        )

        with open(itp_link, "w") as file_out:
            file_out.write(render_itp(atomtype_order, molecule_type))

        with open(top_link, "w") as file_out:
            file_out.write(
                render_top(
                    include_filename=os.path.basename(itp_link),
                    system_name="Pore-System Generated by the PoreMS Package",
                    molecule_name=molecule_type.name,
                )
            )

        return True

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
