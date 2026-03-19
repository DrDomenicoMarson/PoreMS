################################################################################
# Atom Class                                                                   #
#                                                                              #
"""Atom container used throughout molecule and pore structures."""
################################################################################


import pandas as pd


class Atom:
    """Represent a single atom with coordinates and lightweight metadata.

    Parameters
    ----------
    pos : list
        Cartesian atom position.
    atom_type : str
        Chemical symbol used for the atom type.
    name : str, optional
        Optional atom label.
    residue : int, optional
        Residue index used by structure writers.
    """
    def __init__(self, pos, atom_type, name="", residue=0):
        # Initialize
        self._pos = pos
        self._atom_type = atom_type
        self._name = name
        self._residue = residue


    ##################
    # Representation #
    ##################
    def __repr__(self):
        """Return a tabular string representation of the atom.

        Returns
        -------
        repr : str
            Pandas-formatted string containing the stored atom data.
        """
        # Set colums names
        columns = ["Residue", "Name", "Type", "x", "y", "z"]

        # Get data
        data =[[self._residue, self._name, self._atom_type,
                self._pos[0], self._pos[1], self._pos[2]]]

        # Create data frame
        return pd.DataFrame(data, columns=columns).to_string()


    ##################
    # Setter Methods #
    ##################
    def set_pos(self, pos):
        """Update the atom position.

        Parameters
        ----------
        pos : list
            Cartesian atom position.
        """
        self._pos = pos

    def set_atom_type(self, atom_type):
        """Update the atom type.

        Parameters
        ----------
        atom_type : str
            Chemical symbol used for the atom type.
        """
        self._atom_type = atom_type

    def set_name(self, name):
        """Update the atom name.

        Parameters
        ----------
        name : str
            Atom label.
        """
        self._name = name

    def set_residue(self, residue):
        """Update the residue index.

        Parameters
        ----------
        residue : int
            Residue index used by structure writers.
        """
        self._residue = residue


    ##################
    # Getter Methods #
    ##################
    def get_pos(self):
        """Return the atom position.

        Returns
        -------
        pos : list
            Cartesian atom position.
        """
        return self._pos

    def get_atom_type(self):
        """Return the atom type.

        Returns
        -------
        atom_type : str
            Chemical symbol used for the atom type.
        """
        return self._atom_type

    def get_name(self):
        """Return the atom name.

        Returns
        -------
        name : str
            Atom label.
        """
        return self._name

    def get_residue(self):
        """Return the residue index.

        Returns
        -------
        residue : int
            Residue index used by structure writers.
        """
        return self._residue
