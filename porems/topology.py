################################################################################
# GROMACS Topology Models                                                      #
#                                                                              #
"""Dataclasses and helpers for flat GROMACS topology parsing and writing."""
################################################################################


from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GromacsBondParameters:
    """Parameter payload for one GROMACS bond definition.

    Parameters
    ----------
    function : int
        GROMACS bond function type.
    parameters : tuple[str, ...]
        Raw parameter tokens written after ``function``.
    """

    function: int
    parameters: tuple[str, ...]

    @classmethod
    def harmonic(cls, length_nm, force_constant):
        """Build one harmonic bond-parameter record.

        Parameters
        ----------
        length_nm : float
            Equilibrium bond length in nanometers.
        force_constant : float
            Harmonic force constant in the GROMACS bond units.

        Returns
        -------
        parameters : GromacsBondParameters
            Harmonic bond-parameter record.
        """
        return cls(
            function=1,
            parameters=(f"{length_nm:.5f}", f"{force_constant:.6f}"),
        )


@dataclass(frozen=True)
class GromacsAngleParameters:
    """Parameter payload for one GROMACS angle definition.

    Parameters
    ----------
    function : int
        GROMACS angle function type.
    parameters : tuple[str, ...]
        Raw parameter tokens written after ``function``.
    """

    function: int
    parameters: tuple[str, ...]

    @classmethod
    def harmonic(cls, angle_deg, force_constant):
        """Build one harmonic angle-parameter record.

        Parameters
        ----------
        angle_deg : float
            Equilibrium angle in degrees.
        force_constant : float
            Harmonic force constant in the GROMACS angle units.

        Returns
        -------
        parameters : GromacsAngleParameters
            Harmonic angle-parameter record.
        """
        return cls(
            function=1,
            parameters=(f"{angle_deg:.5f}", f"{force_constant:.6f}"),
        )


@dataclass(frozen=True)
class GromacsAtomType:
    """One ``[ atomtypes ]`` row.

    Parameters
    ----------
    name : str
        GROMACS atom-type identifier.
    atomic_number : int or None
        Optional atomic number token.
    mass : str
        Mass token written in the atom-type row.
    charge : str
        Charge token written in the atom-type row.
    particle_type : str
        Particle-type token such as ``"A"``.
    sigma : str
        Lennard-Jones sigma token.
    epsilon : str
        Lennard-Jones epsilon token.
    """

    name: str
    atomic_number: int | None
    mass: str
    charge: str
    particle_type: str
    sigma: str
    epsilon: str

    def is_compatible_with(self, other):
        """Return whether two atom-type definitions are identical.

        Parameters
        ----------
        other : GromacsAtomType
            Atom-type definition to compare.

        Returns
        -------
        is_compatible : bool
            True when all fields except object identity match exactly.
        """
        return (
            self.name == other.name
            and self.atomic_number == other.atomic_number
            and self.mass == other.mass
            and self.charge == other.charge
            and self.particle_type == other.particle_type
            and self.sigma == other.sigma
            and self.epsilon == other.epsilon
        )


@dataclass(frozen=True)
class GromacsAtom:
    """One ``[ atoms ]`` row inside a ``moleculetype``.

    Parameters
    ----------
    index : int
        One-based atom index inside the molecule type.
    atom_type : str
        GROMACS atom-type identifier.
    residue_number : int
        One-based residue number written in the atom row.
    residue_name : str
        Residue name written in the atom row.
    atom_name : str
        Atom name written in the atom row.
    charge_group : int
        Charge-group identifier.
    charge : str
        Charge token written in the atom row.
    mass : str or None
        Optional mass token.
    """

    index: int
    atom_type: str
    residue_number: int
    residue_name: str
    atom_name: str
    charge_group: int
    charge: str
    mass: str | None = None


@dataclass(frozen=True)
class GromacsBond:
    """One ``[ bonds ]`` row.

    Parameters
    ----------
    atom_a : int
        One-based first atom index.
    atom_b : int
        One-based second atom index.
    parameters : GromacsBondParameters
        Bond function and parameter tokens.
    """

    atom_a: int
    atom_b: int
    parameters: GromacsBondParameters


@dataclass(frozen=True)
class GromacsPair:
    """One ``[ pairs ]`` row.

    Parameters
    ----------
    atom_a : int
        One-based first atom index.
    atom_b : int
        One-based second atom index.
    function : int
        GROMACS pair function type.
    parameters : tuple[str, ...], optional
        Optional parameter tokens written after ``function``.
    """

    atom_a: int
    atom_b: int
    function: int
    parameters: tuple[str, ...] = ()


@dataclass(frozen=True)
class GromacsAngle:
    """One ``[ angles ]`` row.

    Parameters
    ----------
    atom_a : int
        One-based first outer atom index.
    atom_b : int
        One-based central atom index.
    atom_c : int
        One-based second outer atom index.
    parameters : GromacsAngleParameters
        Angle function and parameter tokens.
    """

    atom_a: int
    atom_b: int
    atom_c: int
    parameters: GromacsAngleParameters


@dataclass(frozen=True)
class GromacsDihedral:
    """One ``[ dihedrals ]`` row.

    Parameters
    ----------
    atom_a : int
        One-based first atom index.
    atom_b : int
        One-based second atom index.
    atom_c : int
        One-based third atom index.
    atom_d : int
        One-based fourth atom index.
    function : int
        GROMACS dihedral function type.
    parameters : tuple[str, ...], optional
        Optional parameter tokens written after ``function``.
    """

    atom_a: int
    atom_b: int
    atom_c: int
    atom_d: int
    function: int
    parameters: tuple[str, ...] = ()


@dataclass(frozen=True)
class GromacsMoleculeType:
    """Parsed or generated GROMACS molecule-type payload.

    Parameters
    ----------
    name : str
        Molecule-type name.
    nrexcl : int
        ``nrexcl`` value written in ``[ moleculetype ]``.
    atoms : tuple[GromacsAtom, ...]
        Atom rows in molecule-local order.
    bonds : tuple[GromacsBond, ...], optional
        Bond rows.
    pairs : tuple[GromacsPair, ...], optional
        Pair rows.
    angles : tuple[GromacsAngle, ...], optional
        Angle rows.
    dihedrals : tuple[GromacsDihedral, ...], optional
        Dihedral rows.
    """

    name: str
    nrexcl: int
    atoms: tuple[GromacsAtom, ...]
    bonds: tuple[GromacsBond, ...] = ()
    pairs: tuple[GromacsPair, ...] = ()
    angles: tuple[GromacsAngle, ...] = ()
    dihedrals: tuple[GromacsDihedral, ...] = ()


@dataclass(frozen=True)
class ParsedTopologyBundle:
    """One parsed self-contained flat GROMACS topology bundle.

    Parameters
    ----------
    source_path : str
        Source path from which the topology was parsed.
    atomtypes : tuple[GromacsAtomType, ...]
        Parsed atom-type definitions.
    moleculetype : GromacsMoleculeType
        Parsed molecule-type payload.
    """

    source_path: str
    atomtypes: tuple[GromacsAtomType, ...]
    moleculetype: GromacsMoleculeType
    atom_index_by_name: dict[str, int] = field(init=False, repr=False)
    bond_lookup: dict[tuple[str, str], GromacsBond] = field(init=False, repr=False)
    angle_lookup: dict[tuple[str, str, str], GromacsAngle] = field(init=False, repr=False)

    def __post_init__(self):
        """Build local lookup tables and validate unique atom names.

        Raises
        ------
        ValueError
            Raised when atom names are duplicated inside the parsed molecule.
        """
        atom_index_by_name = {}
        for atom in self.moleculetype.atoms:
            if atom.atom_name in atom_index_by_name:
                raise ValueError(
                    "Flat ligand topologies require unique atom names. "
                    f"Found duplicate atom name {atom.atom_name!r} in "
                    f"{self.source_path!r}."
                )
            atom_index_by_name[atom.atom_name] = atom.index
        object.__setattr__(self, "atom_index_by_name", atom_index_by_name)

        atoms_by_index = {
            atom.index: atom
            for atom in self.moleculetype.atoms
        }

        bond_lookup = {}
        for bond in self.moleculetype.bonds:
            name_a = atoms_by_index[bond.atom_a].atom_name
            name_b = atoms_by_index[bond.atom_b].atom_name
            bond_lookup[tuple(sorted((name_a, name_b)))] = bond
        object.__setattr__(self, "bond_lookup", bond_lookup)

        angle_lookup = {}
        for angle in self.moleculetype.angles:
            name_a = atoms_by_index[angle.atom_a].atom_name
            name_b = atoms_by_index[angle.atom_b].atom_name
            name_c = atoms_by_index[angle.atom_c].atom_name
            key = (
                name_a if name_a <= name_c else name_c,
                name_b,
                name_c if name_a <= name_c else name_a,
            )
            angle_lookup[key] = angle
        object.__setattr__(self, "angle_lookup", angle_lookup)

    def atom_by_name(self, atom_name):
        """Return one parsed atom row by atom name.

        Parameters
        ----------
        atom_name : str
            Atom name to look up.

        Returns
        -------
        atom : GromacsAtom
            Matching parsed atom row.

        Raises
        ------
        KeyError
            Raised when ``atom_name`` is not present in the bundle.
        """
        atom_index = self.atom_index_by_name[atom_name]
        for atom in self.moleculetype.atoms:
            if atom.index == atom_index:
                return atom
        raise KeyError(atom_name)

    def has_atom_name(self, atom_name):
        """Return whether the bundle contains one atom name.

        Parameters
        ----------
        atom_name : str
            Atom name to check.

        Returns
        -------
        contains : bool
            True when ``atom_name`` is present in the bundle.
        """
        return atom_name in self.atom_index_by_name

    def bond_by_names(self, atom_name_a, atom_name_b):
        """Return one bond definition by atom names.

        Parameters
        ----------
        atom_name_a : str
            First atom name.
        atom_name_b : str
            Second atom name.

        Returns
        -------
        bond : GromacsBond or None
            Matching bond row when present.
        """
        return self.bond_lookup.get(tuple(sorted((atom_name_a, atom_name_b))))

    def angle_by_names(self, atom_name_a, atom_name_b, atom_name_c):
        """Return one angle definition by atom names.

        Parameters
        ----------
        atom_name_a : str
            First outer atom name.
        atom_name_b : str
            Central atom name.
        atom_name_c : str
            Second outer atom name.

        Returns
        -------
        angle : GromacsAngle or None
            Matching angle row when present.
        """
        key = (
            atom_name_a if atom_name_a <= atom_name_c else atom_name_c,
            atom_name_b,
            atom_name_c if atom_name_a <= atom_name_c else atom_name_a,
        )
        return self.angle_lookup.get(key)


def _strip_comment(line):
    """Return one topology line without trailing semicolon comments.

    Parameters
    ----------
    line : str
        Raw topology line.

    Returns
    -------
    stripped : str
        Line content before the first semicolon.
    """
    return line.split(";", 1)[0].strip()


def _read_section_rows(path):
    """Parse a flat topology document into section-token rows.

    Parameters
    ----------
    path : str or Path
        Topology file path.

    Returns
    -------
    sections : dict[str, list[list[str]]]
        Mapping from lowercase section names to tokenized rows.

    Raises
    ------
    ValueError
        Raised when unsupported preprocessor lines are encountered or when a
        data row appears before the first section header.
    """
    sections = {}
    section_name = None
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = _strip_comment(raw_line)
        if not stripped:
            continue

        if stripped.startswith("#"):
            raise ValueError(
                "Flat slit ligand topology input does not support "
                f"preprocessor lines. Found {stripped!r} in {path!r} "
                f"at line {line_number}."
            )

        if stripped.startswith("[") and stripped.endswith("]"):
            section_name = stripped[1:-1].strip().lower()
            sections.setdefault(section_name, [])
            continue

        if section_name is None:
            raise ValueError(
                f"Topology row {stripped!r} in {path!r} appears before any "
                "section header."
            )

        sections[section_name].append(stripped.split())

    return sections


def _parse_atomtypes(path, rows):
    """Parse ``[ atomtypes ]`` rows.

    Parameters
    ----------
    path : str or Path
        Source path used for error messages.
    rows : list[list[str]]
        Tokenized atom-type rows.

    Returns
    -------
    atomtypes : tuple[GromacsAtomType, ...]
        Parsed atom-type definitions.
    """
    atomtypes = []
    for row in rows:
        if len(row) == 6:
            name, mass, charge, particle_type, sigma, epsilon = row
            atomtypes.append(
                GromacsAtomType(
                    name=name,
                    atomic_number=None,
                    mass=mass,
                    charge=charge,
                    particle_type=particle_type,
                    sigma=sigma,
                    epsilon=epsilon,
                )
            )
            continue
        if len(row) >= 7:
            atomtypes.append(
                GromacsAtomType(
                    name=row[0],
                    atomic_number=int(row[1]),
                    mass=row[2],
                    charge=row[3],
                    particle_type=row[4],
                    sigma=row[5],
                    epsilon=row[6],
                )
            )
            continue

        raise ValueError(
            f"Unsupported [ atomtypes ] row {row!r} in {path!r}. "
            "Expected 6 or 7 tokens."
        )

    return tuple(atomtypes)


def _parse_atoms(path, rows):
    """Parse ``[ atoms ]`` rows.

    Parameters
    ----------
    path : str or Path
        Source path used for error messages.
    rows : list[list[str]]
        Tokenized atom rows.

    Returns
    -------
    atoms : tuple[GromacsAtom, ...]
        Parsed atom rows.
    """
    atoms = []
    for row in rows:
        if len(row) < 7:
            raise ValueError(
                f"Unsupported [ atoms ] row {row!r} in {path!r}. "
                "Expected at least 7 tokens."
            )
        atoms.append(
            GromacsAtom(
                index=int(row[0]),
                atom_type=row[1],
                residue_number=int(row[2]),
                residue_name=row[3],
                atom_name=row[4],
                charge_group=int(row[5]),
                charge=row[6],
                mass=row[7] if len(row) >= 8 else None,
            )
        )
    return tuple(atoms)


def _parse_bonds(path, rows):
    """Parse ``[ bonds ]`` rows.

    Parameters
    ----------
    path : str or Path
        Source path used for error messages.
    rows : list[list[str]]
        Tokenized bond rows.

    Returns
    -------
    bonds : tuple[GromacsBond, ...]
        Parsed bond rows.
    """
    bonds = []
    for row in rows:
        if len(row) < 3:
            raise ValueError(
                f"Unsupported [ bonds ] row {row!r} in {path!r}. "
                "Expected at least 3 tokens."
            )
        bonds.append(
            GromacsBond(
                atom_a=int(row[0]),
                atom_b=int(row[1]),
                parameters=GromacsBondParameters(
                    function=int(row[2]),
                    parameters=tuple(row[3:]),
                ),
            )
        )
    return tuple(bonds)


def _parse_pairs(path, rows):
    """Parse ``[ pairs ]`` rows.

    Parameters
    ----------
    path : str or Path
        Source path used for error messages.
    rows : list[list[str]]
        Tokenized pair rows.

    Returns
    -------
    pairs : tuple[GromacsPair, ...]
        Parsed pair rows.
    """
    pairs = []
    for row in rows:
        if len(row) < 3:
            raise ValueError(
                f"Unsupported [ pairs ] row {row!r} in {path!r}. "
                "Expected at least 3 tokens."
            )
        pairs.append(
            GromacsPair(
                atom_a=int(row[0]),
                atom_b=int(row[1]),
                function=int(row[2]),
                parameters=tuple(row[3:]),
            )
        )
    return tuple(pairs)


def _parse_angles(path, rows):
    """Parse ``[ angles ]`` rows.

    Parameters
    ----------
    path : str or Path
        Source path used for error messages.
    rows : list[list[str]]
        Tokenized angle rows.

    Returns
    -------
    angles : tuple[GromacsAngle, ...]
        Parsed angle rows.
    """
    angles = []
    for row in rows:
        if len(row) < 4:
            raise ValueError(
                f"Unsupported [ angles ] row {row!r} in {path!r}. "
                "Expected at least 4 tokens."
            )
        angles.append(
            GromacsAngle(
                atom_a=int(row[0]),
                atom_b=int(row[1]),
                atom_c=int(row[2]),
                parameters=GromacsAngleParameters(
                    function=int(row[3]),
                    parameters=tuple(row[4:]),
                ),
            )
        )
    return tuple(angles)


def _parse_dihedrals(path, rows):
    """Parse ``[ dihedrals ]`` rows.

    Parameters
    ----------
    path : str or Path
        Source path used for error messages.
    rows : list[list[str]]
        Tokenized dihedral rows.

    Returns
    -------
    dihedrals : tuple[GromacsDihedral, ...]
        Parsed dihedral rows.
    """
    dihedrals = []
    for row in rows:
        if len(row) < 5:
            raise ValueError(
                f"Unsupported [ dihedrals ] row {row!r} in {path!r}. "
                "Expected at least 5 tokens."
            )
        dihedrals.append(
            GromacsDihedral(
                atom_a=int(row[0]),
                atom_b=int(row[1]),
                atom_c=int(row[2]),
                atom_d=int(row[3]),
                function=int(row[4]),
                parameters=tuple(row[5:]),
            )
        )
    return tuple(dihedrals)


def parse_flat_itp(path, moleculetype_name=""):
    """Parse one simple self-contained flat GROMACS ``.itp`` file.

    Parameters
    ----------
    path : str or Path
        Input ``.itp`` file path.
    moleculetype_name : str, optional
        Optional explicit molecule-type name. When provided, the parsed
        ``[ moleculetype ]`` section must match this name.

    Returns
    -------
    bundle : ParsedTopologyBundle
        Parsed self-contained topology bundle.

    Raises
    ------
    ValueError
        Raised when required sections are missing, unsupported sections are
        present, or the flat-input constraints are violated.
    """
    sections = _read_section_rows(path)

    unsupported_sections = sorted(
        section_name
        for section_name in sections
        if section_name not in {
            "atomtypes",
            "moleculetype",
            "atoms",
            "bonds",
            "pairs",
            "angles",
            "dihedrals",
        }
    )
    if unsupported_sections:
        raise ValueError(
            "Flat slit ligand topology input supports only [ atomtypes ], "
            "[ moleculetype ], [ atoms ], [ bonds ], [ pairs ], [ angles ], "
            f"and [ dihedrals ]. Unsupported sections in {path!r}: "
            f"{unsupported_sections}."
        )

    if "moleculetype" not in sections or not sections["moleculetype"]:
        raise ValueError(f"Missing [ moleculetype ] section in {path!r}.")
    if "atoms" not in sections or not sections["atoms"]:
        raise ValueError(f"Missing [ atoms ] section in {path!r}.")

    moleculetype_row = sections["moleculetype"][0]
    if len(moleculetype_row) < 2:
        raise ValueError(
            f"Unsupported [ moleculetype ] row {moleculetype_row!r} in {path!r}. "
            "Expected at least 2 tokens."
        )

    parsed_name = moleculetype_row[0]
    if moleculetype_name and parsed_name != moleculetype_name:
        raise ValueError(
            f"Expected moleculetype {moleculetype_name!r} in {path!r}, "
            f"found {parsed_name!r}."
        )

    bundle = ParsedTopologyBundle(
        source_path=str(path),
        atomtypes=_parse_atomtypes(path, sections.get("atomtypes", [])),
        moleculetype=GromacsMoleculeType(
            name=parsed_name,
            nrexcl=int(moleculetype_row[1]),
            atoms=_parse_atoms(path, sections["atoms"]),
            bonds=_parse_bonds(path, sections.get("bonds", [])),
            pairs=_parse_pairs(path, sections.get("pairs", [])),
            angles=_parse_angles(path, sections.get("angles", [])),
            dihedrals=_parse_dihedrals(path, sections.get("dihedrals", [])),
        ),
    )

    return bundle


def _render_bond_parameters(parameters):
    """Render one bond-parameter payload.

    Parameters
    ----------
    parameters : GromacsBondParameters
        Parameter payload to render.

    Returns
    -------
    text : str
        Rendered function and parameter tokens.
    """
    return " ".join((str(parameters.function), *parameters.parameters))


def _render_angle_parameters(parameters):
    """Render one angle-parameter payload.

    Parameters
    ----------
    parameters : GromacsAngleParameters
        Parameter payload to render.

    Returns
    -------
    text : str
        Rendered function and parameter tokens.
    """
    return " ".join((str(parameters.function), *parameters.parameters))


def render_itp(atomtypes, moleculetype):
    """Render one self-contained GROMACS ``.itp`` document.

    Parameters
    ----------
    atomtypes : list[GromacsAtomType]
        Atom-type definitions written before ``[ moleculetype ]``.
    moleculetype : GromacsMoleculeType
        Molecule-type payload to serialize.

    Returns
    -------
    text : str
        Serialized ``.itp`` document text.
    """
    lines = []

    if atomtypes:
        lines.extend(
            [
                "[ atomtypes ]",
                "; name at.num mass charge ptype sigma epsilon",
            ]
        )
        for atomtype in atomtypes:
            atomic_number = (
                str(atomtype.atomic_number)
                if atomtype.atomic_number is not None
                else ""
            )
            row = [
                atomtype.name,
                atomic_number,
                atomtype.mass,
                atomtype.charge,
                atomtype.particle_type,
                atomtype.sigma,
                atomtype.epsilon,
            ]
            lines.append(" ".join(token for token in row if token != ""))
        lines.append("")

    lines.extend(
        [
            "[ moleculetype ]",
            "; name nrexcl",
            f"{moleculetype.name} {moleculetype.nrexcl}",
            "",
            "[ atoms ]",
            "; nr type resnr resid atom cgnr charge mass",
        ]
    )
    for atom in moleculetype.atoms:
        row = [
            str(atom.index),
            atom.atom_type,
            str(atom.residue_number),
            atom.residue_name,
            atom.atom_name,
            str(atom.charge_group),
            atom.charge,
        ]
        if atom.mass is not None:
            row.append(atom.mass)
        lines.append(" ".join(row))

    if moleculetype.bonds:
        lines.extend(["", "[ bonds ]", "; ai aj funct params"])
        for bond in moleculetype.bonds:
            lines.append(
                f"{bond.atom_a} {bond.atom_b} {_render_bond_parameters(bond.parameters)}"
            )

    if moleculetype.pairs:
        lines.extend(["", "[ pairs ]", "; ai aj funct params"])
        for pair in moleculetype.pairs:
            row = [str(pair.atom_a), str(pair.atom_b), str(pair.function), *pair.parameters]
            lines.append(" ".join(row))

    if moleculetype.angles:
        lines.extend(["", "[ angles ]", "; ai aj ak funct params"])
        for angle in moleculetype.angles:
            lines.append(
                f"{angle.atom_a} {angle.atom_b} {angle.atom_c} "
                f"{_render_angle_parameters(angle.parameters)}"
            )

    if moleculetype.dihedrals:
        lines.extend(["", "[ dihedrals ]", "; ai aj ak al funct params"])
        for dihedral in moleculetype.dihedrals:
            row = [
                str(dihedral.atom_a),
                str(dihedral.atom_b),
                str(dihedral.atom_c),
                str(dihedral.atom_d),
                str(dihedral.function),
                *dihedral.parameters,
            ]
            lines.append(" ".join(row))

    lines.append("")
    return "\n".join(lines)


def render_top(include_filename, system_name, molecule_name):
    """Render one simple master ``.top`` document.

    Parameters
    ----------
    include_filename : str
        Included ``.itp`` filename.
    system_name : str
        Human-readable system label.
    molecule_name : str
        Molecule-type name listed in ``[ molecules ]``.

    Returns
    -------
    text : str
        Serialized ``.top`` document text.
    """
    return "\n".join(
        [
            "[ defaults ]",
            "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ",
            "1 2 yes 0.5 0.833333",
            "",
            f"#include \"{include_filename}\"",
            "",
            "[ system ]",
            system_name,
            "",
            "[ molecules ]",
            f"{molecule_name} 1",
            "",
        ]
    )
