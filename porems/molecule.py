################################################################################
# Molecule Class                                                               #
#                                                                              #
"""Tools for creating, transforming, and inspecting molecular structures."""
################################################################################


import math
import pandas as pd

import porems.utils as utils
import porems.database as db
import porems.geometry as geometry

from porems.atom import Atom


class Molecule:
    """Mutable collection of :class:`porems.atom.Atom` objects.

    The class provides geometry helpers, coordinate transforms, partial
    rotations and translations, atom-level editing, and simple file import for
    the molecular formats used throughout PoreMS.

    Parameters
    ----------
    name : str, optional
        Molecule name.
    short : str, optional
        Short residue-style identifier used in exported structure files.
    inp : None, str, list, optional
        Optional initial content. Use ``None`` for an empty molecule, a file
        path string to read a structure, a list of :class:`Molecule` objects to
        concatenate them, or a list of :class:`porems.atom.Atom` objects to use
        directly.

    Examples
    --------
    Following example generates a benzene molecule without hydrogen atoms

    .. code-block:: python

        import porems as pms

        mol = pms.Molecule("benzene", "BEN")
        mol.add("C", [0,0,0])
        mol.add("C", 0, r=0.1375, theta= 60)
        mol.add("C", 1, r=0.1375, theta=120)
        mol.add("C", 2, r=0.1375, theta=180)
        mol.add("C", 3, r=0.1375, theta=240)
        mol.add("C", 4, r=0.1375, theta=300)

    """
    def __init__(self, name="molecule", short="MOL", inp=None):
        # Initialize
        self._dim = 3

        self._name = name
        self._short = short

        self._box = []
        self._charge = 0
        self._masses = []
        self._mass = 0

        # Check data input
        if inp is None:
            self._atom_list = []
        else:
            # Read from file
            if isinstance(inp, str):
                self._atom_list = self._read(inp, inp.split(".")[-1].upper())
            # Concat multiple molecules
            elif isinstance(inp, list):
                # Atom list is provided
                if(isinstance(inp[0], Atom)):
                    self._atom_list = inp
                # List of molecules is provided
                else:
                    self._atom_list = self._concat(inp)


    ##################
    # Representation #
    ##################
    def __repr__(self):
        """Create a pandas table of the molecule data.

        Returns
        -------
        repr : DataFrame
            Pandas data frame of the molecule object
        """
        # Set colums names
        columns = ["Residue", "Name", "Type", "x", "y", "z"]

        # Get data
        data =[]
        for atom in self._atom_list:
            data.append([atom.get_residue(), atom.get_name(), atom.get_atom_type(),
                         atom.get_pos()[0], atom.get_pos()[1], atom.get_pos()[2]])

        # Create data frame
        return pd.DataFrame(data, columns=columns).to_string()


    ##############
    # Management #
    ##############
    def _read(self, file_path, file_type):
        """Read a molecule from a file. Currently only **GRO**, **PDB** and
        **MOL2** files are supported.

        Parameters
        ----------
        file_path : string
            Link to requested file
        file_type : string
            File extension name

        Returns
        -------
        atom_list : list
            Atom list

        Raises
        ------
        ValueError
            Raised when the requested file type is unsupported.
        """
        # Process input
        if not file_type in ["GRO", "PDB", "MOL2"]:
            raise ValueError("Unsupported filetype.")

        # Read molecule
        atom_list = []
        with open(file_path, "r") as file_in:
            for line_idx, line in enumerate(file_in):
                line_val = line.split()
                is_add = False

                # Gro file
                if file_type == "GRO":
                    if line_idx > 0 and len(line_val) > 3:
                        residue = int(line[0:5])-1
                        pos = [float(line_val[i]) for i in range(3, 5+1)]
                        name = line_val[1]
                        atom_type = ''.join([i for i in line_val[1] if not i.isdigit()])
                        is_add = True

                # Pdb file
                elif file_type == "PDB":
                    if line_val[0] in ["ATOM", "HETATM"]:
                        residue = int(line[22:26])-1
                        pos = [float(line_val[i])/10 for i in range(6, 8+1)]
                        name = line_val[11]
                        atom_type = line_val[11]
                        is_add = True

                # Mol2 file
                elif file_type == "MOL2":
                    if len(line_val) > 8:
                        residue = 0
                        pos = [float(line_val[i])/10 for i in range(2, 4+1)]
                        name = line_val[1]
                        atom_type = ''.join([i for i in line_val[1] if not i.isdigit()])
                        is_add = True

                if is_add:
                    atom_list.append(Atom(pos, atom_type, name, residue))

        # Transform to column
        return atom_list

    def _concat(self, mol_list):
        """Concatenate a molecule list into one molecule object.

        Parameters
        ----------
        mol_list : list
            List of molecule objects to be concatenated

        Returns
        -------
        atom_list : list
            Atom list
        """
        return sum([mol.get_atom_list() for mol in mol_list], [])

    def _temp(self, atoms):
        """Create a temporary molecule of specified atom ids.

        Parameters
        ----------
        atoms : list
            List of atoms to be included

        Returns
        -------
        mol : Molecule
            Molecule object
        """
        return Molecule(inp=[self._atom_list[x] for x in atoms])

    def append(self, mol):
        """Append all atoms from another molecule.

        Parameters
        ----------
        mol : Molecule
            Molecule whose atoms will be appended in their current order.
        """
        self._atom_list += mol.get_atom_list()

    def column_pos(self):
        """Return Cartesian coordinates grouped by dimension.

        Returns
        -------
        column : list
            Position columns ordered as ``[x_values, y_values, z_values]``.
        """
        return utils.column([atom.get_pos() for atom in self._atom_list])


    ############
    # Geometry #
    ############
    def _vector(self, pos_a, pos_b):
        """Calculate the vector between to two positions as defined in
        :class:`porems.geometry.vector` with the addition to define the inputs
        as atom indices.

        Parameters
        ----------
        pos_a : integer, list
            First position :math:`\\boldsymbol{a}`
        pos_b : integer, list
            Second position :math:`\\boldsymbol{b}`

        Returns
        -------
        vector : list
            Bond vector

        Raises
        ------
        ValueError
            Raised when the inputs are not atom ids or three-dimensional
            position vectors.
        """
        # Process input
        if isinstance(pos_a, int) and isinstance(pos_b, int):
            pos_a = self.pos(pos_a)
            pos_b = self.pos(pos_b)
        elif not (isinstance(pos_a, (list, tuple)) and isinstance(pos_b, (list, tuple))):
            raise ValueError("Vector: Wrong input...")

        # Check dimensions
        if len(pos_a) != self._dim or len(pos_b) != self._dim:
            raise ValueError("Vector: Wrong dimensions...")

        # Calculate vector
        return geometry.vector(pos_a, pos_b)

    def _box_size(self):
        """Calculate the box size of the current molecule. This is done by
        determining the maximal coordinate value of all atoms in all dimensions

        .. math::

            \\boldsymbol{b}=\\begin{pmatrix}\\max(\\boldsymbol{d}_1)&\\max(\\boldsymbol{d}_1)&\\dots&\\max(\\boldsymbol{d}_n)\\end{pmatrix}^T

        where :math:`\\boldsymbol{d}_i` is the dimension-vector of the data
        matrix.

        Returns
        -------
        box : list
            Box length of the current molecule
        """
        data = self.column_pos()
        return [max(data[i]) if max(data[i]) > 0 else 0.001 for i in range(self._dim)]


    ##############
    # Properties #
    ##############
    def pos(self, atom):
        """Return the Cartesian position of one atom.

        Parameters
        ----------
        atom : int
            Atom index.

        Returns
        -------
        pos : list
            Three-dimensional position vector of the selected atom.
        """
        return self._atom_list[atom].get_pos()

    def bond(self, inp_a, inp_b):
        """Return the vector between two atoms or positions.

        Parameters
        ----------
        inp_a : int or list
            First atom index or Cartesian position.
        inp_b : int or list
            Second atom index or Cartesian position.

        Returns
        -------
        bond : list
            Vector from ``inp_a`` to ``inp_b``.

        Examples
        --------
        .. code-block:: python

            mol.bond(0, 1)
            mol.bond(*[0, 1])
            mol.bond([1, 0, 0], [0, 0, 0])
        """
        return self._vector(inp_a, inp_b)

    def centroid(self):
        """Return the geometric centroid of all atom positions.

        .. math::

            \\text{centroid}=\\begin{pmatrix}c_1&c_2&\\dots&c_n\\end{pmatrix}^T

        with

        .. math::

            c_i=\\frac{1}{m}\\sum_j^m d_{ij}.

        Hereby :math:`i\\dots n` stands for the dimension and :math:`j\\dots m`
        for the molecule.

        Returns
        -------
        centroid : list
            Arithmetic mean of all atomic coordinates.
        """
        # Calculate the centroid
        data = self.column_pos()
        return [sum(data[i])/len(data[i]) for i in range(self._dim)]

    def com(self):
        """Return the mass-weighted center of mass.

        .. math::

            \\text{com}=\\begin{pmatrix}c_1&c_2&\\dots&c_n\\end{pmatrix}^T

        with

        .. math::

            c_i=\\frac{1}{\\sum_j^mM_j}\\sum_j^m d_{ij}\\cdot M_j

        and mass :math:`M`. Hereby :math:`i\\dots n` stands for the dimension
        and :math:`j\\dots m` for the molecule.

        Returns
        -------
        com : list
            Mass-weighted center of mass.
        """
        # Calculate the center of mass
        data = self.column_pos()
        masses = self.get_masses()
        return [sum([data[i][j]*masses[j] for j in range(self.get_num())])/sum(masses) for i in range(self._dim)]


    #################
    # Basic Editing #
    #################
    def translate(self, vec):
        """Translate every atom by the same vector.

        .. math::

            \\boldsymbol{D}_\\text{trans}=
            \\boldsymbol{D}+\\boldsymbol{a}=
            \\begin{pmatrix}
            \\boldsymbol{d}_1+a_1&\\boldsymbol{d}_2+a_2&\\dots&\\boldsymbol{d}_n+a_n&\\boldsymbol{d}_t
            \\end{pmatrix}

        Parameters
        ----------
        vec : list
            Translation vector.
        """
        for atom in self._atom_list:
            atom.set_pos([atom.get_pos()[i]+vec[i] for i in range(self._dim)])

    def rotate(self, axis, angle, is_deg=True):
        """Rotate all atom positions around a common axis.

        Parameters
        ----------
        axis : int, str, list
            Rotation axis accepted by :func:`porems.geometry.rotate`.
        angle : float
            Rotation angle.
        is_deg : bool, optional
            True if ``angle`` is given in degrees.
        """
        for atom in self._atom_list:
            atom.set_pos(geometry.rotate(atom.get_pos(), axis, angle, is_deg))

    def move(self, atom, pos):
        """Translate the molecule so one atom reaches a target position.

        Parameters
        ----------
        atom : int
            Anchor atom index.
        pos : list
            Target position of the anchor atom.
        """
        self.translate(self._vector(self.pos(atom), pos))

    def zero(self, pos=None):
        """Translate the molecule so its minimum coordinate matches ``pos``.

        Parameters
        ----------
        pos : list, optional
            New lower coordinate corner. Defaults to the origin.

        Returns
        -------
        vec : list
            Translation vector applied to the molecule.
        """
        pos = [0, 0, 0] if pos is None else pos

        # Calculate translation vector
        data = self.column_pos()
        vec = [pos[i]-min(data[i]) for i in range(self._dim)]

        # Reset box size
        self._box = []

        # Translate molecule
        self.translate(vec)

        return vec

    def put(self, atom, pos):
        """Set the Cartesian position of one atom directly.

        Parameters
        ----------
        atom : int
            Atom index to update.
        pos : list
            New position vector.
        """
        self._atom_list[atom].set_pos(pos)


    ####################
    # Advanced Editing #
    ####################
    def part_move(self, bond, atoms, length, vec=None):
        """Translate part of a molecule to adjust a bond length.

        ``atoms`` selects the atoms that are moved as a rigid fragment. When
        ``vec`` is omitted, the translation direction is derived from the bond
        between the two atom ids in ``bond`` and scaled so the bond reaches the
        requested final ``length``.

        Parameters
        ----------
        bond : list
            Two atom ids defining the bond to adjust.
        atoms : int or list
            Atom id or atom ids that are translated as one fragment.
        length : float
            Requested final bond length.
        vec : list, optional
            Explicit translation vector. When provided, ``length`` is ignored
            for the actual displacement direction.

        Examples
        --------
        .. code-block:: python

            mol.part_move([0, 1], [1, 2, 3], 0.5)
        """
        # Create temporary molecule
        if isinstance(atoms, int):
            atoms = [atoms]
        temp = self._temp(atoms)

        # Set length
        length = abs(length-geometry.length(self.bond(*bond)))

        # Set vector
        if not vec:
            vec = self._vector(bond[0], bond[1])
        vec = [v*length for v in geometry.unit(vec)]

        # Move molecule
        temp.translate(vec)

    def part_rotate(self, bond, atoms, angle, zero):
        """Rotate part of the molecule around a bond axis.

        The whole molecule is first translated so ``zero`` becomes the origin.
        The atoms listed in ``atoms`` are then rotated around the bond defined
        by ``bond``.

        Parameters
        ----------
        bond : list
            Two atom ids defining the rotation axis.
        atoms : int or list
            Atom id or atom ids to rotate.
        angle : float
            Rotation angle in degrees.
        zero : int
            Atom id that is translated to the origin before the rotation.

        Examples
        --------
        .. code-block:: python

            mol.part_rotate([0, 1], [1, 2, 3], 90, 0)
        """
        # Create temporary molecule
        self.move(zero, [0, 0, 0])
        if isinstance(atoms, int):
            atoms = [atoms]
        temp = self._temp(atoms)

        # Rotate molecule
        temp.rotate([self.pos(bond[0]), self.pos(bond[1])], angle)

    def part_angle(self, bond_a, bond_b, atoms, angle, zero):
        """Rotate a fragment to change the angle between two bonds.

        The rotation axis is the cross product of ``bond_a`` and ``bond_b``.
        As in :meth:`part_rotate`, the molecule is first translated so the atom
        identified by ``zero`` becomes the origin.

        Parameters
        ----------
        bond_a : list
            First bond definition, either two atom ids or an explicit vector.
        bond_b : list
            Second bond definition, either two atom ids or an explicit vector.
        atoms : int or list
            Atom id or atom ids to rotate.
        angle : float
            Rotation angle in degrees.
        zero : int
            Atom id translated to the origin before the rotation.

        Examples
        --------
        .. code-block:: python

            mol.part_angle([0, 1], [1, 2], [1, 2, 3], 90, 1)

        Raises
        ------
        ValueError
            Raised when the provided bond definitions do not have compatible
            dimensions.
        """
        if len(bond_a) != len(bond_b):
            raise ValueError("Part_Angle : Wrong bond dimensions...")
        if len(bond_a) not in [2, self._dim]:
            raise ValueError("Part_Angle : Wrong bond input...")

        # Create temporary molecule
        self.move(zero, [0, 0, 0])
        if isinstance(atoms, int):
            atoms = [atoms]
        temp = self._temp(atoms)

        # Rotate molecule around normal vector
        if len(bond_a) == 2:
            vec = geometry.cross_product(self._vector(*bond_a), self._vector(*bond_b))
        else:
            vec = geometry.cross_product(bond_a, bond_b)

        # Rotate molecule
        temp.rotate(vec, angle)


    #########
    # Atoms #
    #########
    def add(
        self,
        atom_type,
        pos,
        bond=None,
        r=0,
        theta=0,
        phi=0,
        is_deg=True,
        name="",
        residue=0,
        source_id=None,
    ):
        """Add a new atom using spherical coordinates relative to a reference.

        ``pos`` defines the coordinate origin, either as an atom id or as an
        explicit Cartesian vector. When ``bond`` is given, its direction is used
        as the local axis before applying ``r``, ``theta``, and ``phi``.

        Parameters
        ----------
        atom_type : str
            Atom type of the new atom.
        pos : int or list
            Reference atom id or Cartesian origin.
        bond : list, optional
            Optional two-atom bond defining the local axis orientation.
        r : float, optional
            Radial distance from ``pos``.
        theta : float, optional
            Polar rotation angle.
        phi : float, optional
            Azimuthal rotation angle.
        is_deg : bool, optional
            True if ``theta`` and ``phi`` are given in degrees.
        name : str, optional
            Optional explicit atom name.
        residue : int, optional
            Residue index stored on the new atom.
        source_id : int or None, optional
            Optional identifier of the source atom in a parent structure. This
            metadata can later be used by structure writers to reconstruct
            connectivity for objectified scaffold atoms.

        Examples
        --------
        .. code-block:: python

            mol.add("C", [0, 0, 0])
            mol.add("C", 0, r=0.153, theta=-135)
            mol.add("C", 1, [0, 1], r=0.153, theta= 135)
        """
        # Process input
        pos = self.pos(pos) if isinstance(pos, int) else pos
        vec = self._vector(*bond) if bond else geometry.main_axis("z")

        # Add coordinate transformation when given a bond
        phi += geometry.angle_polar(vec, is_deg)
        theta += geometry.angle_azi(vec, is_deg)

        # Process angles
        phi *= math.pi/180 if is_deg else 1
        theta *= math.pi/180 if is_deg else 1

        # Transform spherical to cartesian coordinates
        x = r*math.sin(theta)*math.cos(phi)
        y = r*math.sin(theta)*math.sin(phi)
        z = r*math.cos(theta)
        coord = [x, y, z]

        # Create new atom
        self._atom_list.append(
            Atom(
                [pos[i]+coord[i] for i in range(self._dim)],
                atom_type,
                name,
                residue,
                source_id=source_id,
            )
        )

    # Delete an atom
    def delete(self, atoms):
        """Delete one atom or several atoms by index.

        Parameters
        ----------
        atoms : int or list
            Atom index or indices to remove.
        """
        # Process input
        atoms = [atoms] if isinstance(atoms, int) else atoms

        # Remove atoms
        for atom in sorted(atoms, reverse=True):
            self._atom_list.pop(atom)

    def overlap(self, error=0.005):
        """Return groups of atoms whose coordinates overlap within a tolerance.

        Parameters
        ----------
        error : float, optional
            Maximum absolute coordinate difference per dimension.

        Returns
        -------
        duplicate : dict
            Mapping from reference atom id to overlapping atom ids.
        """
        # Initialize
        atom_list = {x: False for x in range(self.get_num())}
        duplicates = {}

        # Run through complete atoms list
        for atom_a in atom_list:
            # Ignore duplicate items
            if not atom_list[atom_a]:
                # Run through atom list after first loop
                for atom_b in [x for x in atom_list if x>atom_a]:
                    # Check if overlapping
                    if sum([error > abs(x) for x in geometry.vector(self.pos(atom_a), self.pos(atom_b))]) == 3:
                        if not atom_a in duplicates:
                            duplicates[atom_a] = []
                        duplicates[atom_a].append(atom_b)
                        # Set to false
                        atom_list[atom_a] = True
                        atom_list[atom_b] = True

        # Return duplicates
        return duplicates

    def switch_atom_order(self, atom_a, atom_b):
        """Swap two atoms in the internal atom list.

        Parameters
        ----------
        atom_a : int
            Index of the first atom.
        atom_b : int
            Index of the second atom.
        """
        self._atom_list[atom_a], self._atom_list[atom_b] = self._atom_list[atom_b], self._atom_list[atom_a]

    def set_atom_type(self, atom, atom_type):
        """Set the atom type of one atom.

        Parameters
        ----------
        atom : int
            Atom index.
        atom_type : str
            New atom type label.
        """
        self._atom_list[atom].set_atom_type(atom_type)

    def set_atom_name(self, atom, name):
        """Set the name of one atom.

        Parameters
        ----------
        atom : int
            Atom index.
        name : str
            New atom name.
        """
        self._atom_list[atom].set_name(name)

    def set_atom_residue(self, atom, residue):
        """Set the residue index of one atom.

        Parameters
        ----------
        atom : int
            Atom index.
        residue : int
            New residue index.
        """
        self._atom_list[atom].set_residue(residue)

    def get_atom_type(self, atom):
        """Return the atom type stored for one atom.

        Parameters
        ----------
        atom : int
            Atom index.

        Returns
        -------
        atom_type : str
            Atom type label.
        """
        return self._atom_list[atom].get_atom_type()

    def get_atom_list(self):
        """Return the underlying atom list.

        Returns
        -------
        atom_list : list
            Internal list of :class:`porems.atom.Atom` objects.
        """
        return self._atom_list


    ##################
    # Setter Methods #
    ##################
    def set_name(self, name):
        """Set the molecule name.

        Parameters
        ----------
        name : str
            Molecule name.
        """
        self._name = name

    def set_short(self, short):
        """Set the molecule short identifier.

        Parameters
        ----------
        short : str
            Short name used in exported structure records.
        """
        self._short = short

    def set_box(self, box):
        """Set the simulation box dimensions.

        Parameters
        ----------
        box : list
            Box lengths in all dimensions.
        """
        self._box = box

    def set_charge(self, charge):
        """Set the total molecular charge.

        Parameters
        ----------
        charge : float
            Total charge of the molecule.
        """
        self._charge = charge

    def set_masses(self, masses=None):
        """Set per-atom molar masses.

        Parameters
        ----------
        masses : list, optional
            Explicit molar masses in :math:`\\frac{g}{mol}`. When omitted, the
            masses are inferred from the atom types using
            :mod:`porems.database`.
        """
        self._masses = masses if masses else [db.get_mass(atom.get_atom_type()) for atom in self._atom_list]

    def set_mass(self, mass=0):
        """Set or derive the total molar mass of the molecule.

        Parameters
        ----------
        mass : float, optional
            Explicit total molar mass in :math:`\\frac{g}{mol}`. When zero, the
            value is derived from :meth:`get_masses`.
        """
        self._mass = mass if mass else sum(self.get_masses())


    ##################
    # Getter Methods #
    ##################
    def get_name(self):
        """Return the molecule name.

        Returns
        -------
        name : str
            Molecule name.
        """
        return self._name

    def get_short(self):
        """Return the molecule short identifier.

        Returns
        -------
        short : str
            Short name used in exported structure records.
        """
        return self._short

    def get_box(self):
        """Return the simulation box dimensions.

        Returns
        -------
        box : list
            Box lengths in all dimensions. When no explicit box was set, a
            bounding box derived from the current coordinates is returned.
        """
        return self._box if self._box else self._box_size()

    def get_num(self):
        """Return the number of atoms.

        Returns
        -------
        num : int
            Number of atoms currently stored in the molecule.
        """
        return len(self._atom_list)

    def get_charge(self):
        """Return the total molecular charge.

        Returns
        -------
        charge : float
            Total charge of the molecule.
        """
        return self._charge

    def get_masses(self):
        """Return per-atom molar masses.

        Returns
        -------
        masses : list
            Atom masses in :math:`\\frac{g}{mol}`.
        """
        if not self._masses:
            self.set_masses()
        return self._masses

    def get_mass(self):
        """Return the total molar mass of the molecule.

        Returns
        -------
        mass : float
            Total molar mass in :math:`\\frac{g}{mol}`.
        """
        if not self._mass:
            self.set_mass()
        return self._mass
