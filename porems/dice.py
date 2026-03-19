################################################################################
# Dice Class                                                                   #
#                                                                              #
"""Separation of a molecule object into smaller cubes for pair-search."""
################################################################################


import __main__
import math
import multiprocessing as mp
import os
import warnings

from dataclasses import dataclass
from enum import Enum

import porems.geometry as geometry


class SearchExecution(str, Enum):
    """Execution modes supported by :meth:`Dice.find`.

    Attributes
    ----------
    AUTO
        Prefer process-based search when it is safe, otherwise run serially.
    SERIAL
        Always run the search in the current process.
    PROCESSES
        Require a process pool and fail if multiprocessing cannot be started
        safely from the current entrypoint.
    """

    AUTO = "auto"
    SERIAL = "serial"
    PROCESSES = "processes"


@dataclass(frozen=True)
class SearchPolicy:
    """Execution policy for :meth:`Dice.find`.

    Parameters
    ----------
    execution : SearchExecution, optional
        Requested execution mode for the search.
    processes : int, optional
        Number of worker processes to request when process-based execution is
        used. If omitted, the CPU count detected during :class:`Dice`
        initialization is used.
    start_method : str, optional
        Explicit multiprocessing start method to use when process-based
        execution is requested. Leave ``None`` to use the interpreter default.
    warn_on_fallback : bool, optional
        True to emit a :class:`RuntimeWarning` when ``AUTO`` falls back to
        serial execution because multiprocessing is unsafe from the current
        entrypoint.
    """

    execution: SearchExecution = SearchExecution.AUTO
    processes: int | None = None
    start_method: str | None = None
    warn_on_fallback: bool = True

    def __post_init__(self):
        """Validate and normalize the configured search policy.

        Returns
        -------
        None
            The dataclass instance is normalized in place.

        Raises
        ------
        ValueError
            If ``processes`` is smaller than one or ``start_method`` is not
            supported by the current Python interpreter.
        """
        execution = SearchExecution(self.execution)
        object.__setattr__(self, "execution", execution)

        if self.processes is not None and self.processes < 1:
            raise ValueError("SearchPolicy.processes must be at least 1.")

        if self.start_method is not None:
            valid_methods = mp.get_all_start_methods()
            if self.start_method not in valid_methods:
                raise ValueError(
                    "SearchPolicy.start_method must be one of "
                    f"{valid_methods}."
                )


class Dice:
    """This class splits the molecule into smaller sub boxes and provides
    configurable functions for pair-search.

    The aim is reducing the workload on search algorithms for atom pairs.
    Normally the computational effort would be

    .. math::

        \\mathcal O(n^2)

    with the number of atoms :math:`n`, because each atom has to be compared
    with all other atoms.

    Since a bond distance is fixed, the idea is reducing the search space by
    dividing the molecule box into cubes and only performing the  pair-search
    within these smaller boxes and their 26 immediate neighbors. Assuming that
    the grid structure is ideal in a geometrical sense, that all bond length and
    angles are constant, the number of atoms in each cube are a constant
    :math:`b`. The computational effort for each atom is thus a constant

    .. math::

        \\mathcal O(27\\cdot b^2)

    Therefore, the computational effort for an entire search scales linear with
    the number of cubes. For example, doubling the cristobalite block size only
    increases the effort eightfold.

    Furthermore, the search can be parallelized across multiple processes,
    since no communication is needed between the subprocesses that each cover
    a set of cubes. Use :meth:`find` together with :class:`SearchPolicy` to
    control whether the search runs serially or via a process pool.

    Note that the cube size must be strictly greater than the intended
    bond length searches.

    Parameters
    ----------
    mol : Molecule
        Molecule to be divided
    size : float
        Cube edge size
    is_pbc : bool
        True if periodic boundary conditions are needed
    """
    def __init__(self, mol, size, is_pbc):
        # Initialize
        self._dim = 3
        self._np = mp.cpu_count()

        self._mol = mol
        self._size = size
        self._is_pbc = is_pbc

        self._atom_data = {atom_id: [atom.get_atom_type(), atom.get_pos()] for atom_id, atom in enumerate(self._mol.get_atom_list())}
        self._mol_box = self._mol.get_box()

        # Split molecule box into cubes and fill them with atom ids
        self._split()
        self._fill()


    ##############
    # Management #
    ##############
    def _split(self):
        """Here the number of cubes is calculated for each dimension for the
        defined cube size and molecule dimension. A dictionary of cubes is
        generated containing the coordinates of the origin point of each cube.
        Furthermore, an empty list for each cube is added, that will contain
        atom ids of atom objects.

        Cube ids are tuples containing three elements with the x-axis as the
        first entry, y-axis as the second entry and the z-axis as the third
        entry

        .. math::

            \\text{id}=\\begin{pmatrix}x&y&z\\end{pmatrix}.
        """
        # Calculate number of cubes in each dimension
        self._count = [max(1, math.floor(box/self._size)) for box in self._mol_box]

        # Fill cube origins
        self._origin = {}
        self._pointer = {}
        for i in range(self._count[0]):
            for j in range(self._count[1]):
                for k in range(self._count[2]):
                    self._origin[(i, j, k)] = [self._size*x for x in [i, j, k]]
                    self._pointer[(i, j, k)] = []

    def _pos_to_index(self, position):
        """Calculate the cube index for a given position.

        Parameters
        ----------
        position : list
            Three-dimensional coordinates

        Returns
        -------
        index : tuple
            Cube index
        """
        index = []
        for dim, pos in enumerate(position):
            pos_index = math.floor(pos/self._size)
            if pos_index < 0:
                pos_index = 0
            elif pos_index >= self._count[dim]:
                pos_index = self._count[dim]-1
            index.append(pos_index)

        return tuple(index)

    def _fill(self):
        """Based on their coordinates, the atom ids, as defined in the molecule
        object, are filled into the cubes.
        """
        for atom_id, atom in enumerate(self._mol.get_atom_list()):
            self._pointer[self._pos_to_index(atom.get_pos())].append(atom_id)


    ############
    # Iterator #
    ############
    def _step(self, dim, step, index):
        """Helper function for iterating through the cubes. Optionally,
        periodic boundary conditions are applied.

        Parameters
        ----------
        dim : integer
            Stepping dimension
        step : integer
            Step to move
        index : list
            Cube index

        Returns
        -------
        index : list
            New cube index
        """
        # Step in intended dimension
        index = list(index)
        index[dim] += step

        # Periodicity
        if index[dim] >= self._count[dim]:
            index[dim] = 0 if self._is_pbc else None
        elif index[dim] < 0:
            index[dim] = self._count[dim]-1 if self._is_pbc else None

        return tuple(index)

    def _right(self, index):
        """Step one cube to the right considering the x-axis.

        Parameters
        ----------
        index : list
            Cube index

        Returns
        -------
        index : list
            New cube index
        """
        return self._step(0, 1, index)

    def _left(self, index):
        """Step one cube to the left considering the x-axis.

        Parameters
        ----------
        index : list
            Cube index

        Returns
        -------
        index : list
            New cube index
        """
        return self._step(0, -1, index)

    def _top(self, index):
        """Step one cube to the top considering the y-axis.

        Parameters
        ----------
        index : list
            Cube index

        Returns
        -------
        index : list
            New cube index
        """
        return self._step(1, 1, index)

    def _bot(self, index):
        """Step one cube to the bottom considering the y-axis.

        Parameters
        ----------
        index : list
            Cube index

        Returns
        -------
        index : list
            New cube index
        """
        return self._step(1, -1, index)

    def _front(self, index):
        """Step one cube to the front considering the z-axis.

        Parameters
        ----------
        index : list
            Cube index

        Returns
        -------
        index : list
            New cube index
        """
        return self._step(2, 1, index)

    def _back(self, index):
        """Step one cube to the back considering the z-axis.

        Parameters
        ----------
        index : list
            Cube index

        Returns
        -------
        index : list
            New cube index
        """
        return self._step(2, -1, index)

    def neighbor(self, cube_id, is_self=True):
        """Get the ids of the cubes surrounding the given one.

        Parameters
        ----------
        index : list
            Main cube index
        is_self : bool, optional
            True to add the main cube to the output

        Returns
        -------
        neighbor : list
            List of surrounding cube ids, optionally including given cube id
        """
        # Initialize
        neighbor = []

        # Find neighbors
        z = [self._back(cube_id), cube_id, self._front(cube_id)]
        y = [[self._top(i), i, self._bot(i)] for i in z]

        for i in range(len(z)):
            for j in range(len(y[i])):
                neighbor.append(self._left(y[i][j]))
                neighbor.append(y[i][j])
                neighbor.append(self._right(y[i][j]))

        if not is_self:
            neighbor.pop(13)

        return [n for n in neighbor if n is not None]


    ##########
    # Search #
    ##########
    def _find_bond(self, cube_list, atom_type, distance):
        """Search for a bond in the given cubes.

        This internal helper searches for
        atom-pairs that fulfill the distance requirements within the given cube
        and all 26 surrounding ones.

        Parameters
        ----------
        cube_list : list
            List of cube indices to search in, use an empty list for all cubes
        atom_type : list
            List of two atom types
        distance : list
            Bounds of allowed distance [lower, upper]

        Returns
        -------
        bond_list : list
            Bond array containing lists of two atom ids
        """
        # Process input
        cube_list = cube_list if cube_list else list(self._pointer.keys())

        # Loop through all given cubes
        bond_list = []
        for cube_id in cube_list:
            # Get atom ids of surrounding cubes
            atoms = sum([self._pointer[x] for x in self.neighbor(cube_id) if None not in x], [])

            # Run through atoms in the main cube
            for atom_id_a in self._pointer[cube_id]:
                # Check type
                if self._atom_data[atom_id_a][0] == atom_type[0]:
                    entry = [atom_id_a, []]
                    # Search in all surrounding cubes for partners
                    for atom_id_b in atoms:
                        if self._atom_data[atom_id_b][0] == atom_type[1] and not atom_id_a == atom_id_b:
                            # Calculate bond vector
                            bond_vector = [0, 0, 0]
                            for dim in range(self._dim):
                                # Nearest image convention
                                bond_vector[dim] = self._atom_data[atom_id_a][1][dim]-self._atom_data[atom_id_b][1][dim]
                                if abs(bond_vector[dim]) > 3*self._size:
                                    bond_vector[dim] -= self._mol_box[dim]*round(bond_vector[dim]/self._mol_box[dim])
                            # Calculate bond length
                            length = geometry.length(bond_vector)
                            # Check if bond distance is within error
                            if length >= distance[0] and length <= distance[1]:
                                entry[1].append(atom_id_b)

                    # Add pairs to bond list
                    bond_list.append(entry)

        return bond_list

    def _resolve_process_count(self, processes):
        """Resolve the requested worker-process count.

        Parameters
        ----------
        processes : int, optional
            Requested number of worker processes.

        Returns
        -------
        process_count : int
            Number of worker processes to request.
        """
        return self._np if processes is None else processes

    def _chunk_cube_list(self, cube_list, process_count):
        """Split the cube list into chunks for worker processes.

        Parameters
        ----------
        cube_list : list
            List of cube indices to search in.
        process_count : int
            Number of worker processes requested.

        Returns
        -------
        cube_chunks : list
            List of non-empty cube-index chunks.
        """
        chunk_size = max(1, math.ceil(len(cube_list) / process_count))
        return [
            cube_list[index:index + chunk_size]
            for index in range(0, len(cube_list), chunk_size)
        ]

    def _unsafe_entrypoint_reason(self):
        """Describe why the current entrypoint is unsafe for multiprocessing.

        Returns
        -------
        reason : str or None
            Description of the multiprocessing entrypoint problem, or ``None``
            if the current ``__main__`` module is safely importable.
        """
        main_file = getattr(__main__, "__file__", None)
        if main_file is None:
            return "the current __main__ module does not define __file__"
        if main_file == "<stdin>":
            return "the current __main__ module originates from <stdin>"
        if not os.path.exists(main_file):
            return f"the current __main__.__file__ path does not exist: {main_file}"

        return None

    def _serial_fallback_warning(self, reason):
        """Build the warning message for automatic serial fallback.

        Parameters
        ----------
        reason : str
            Explanation of why multiprocessing is unsafe.

        Returns
        -------
        message : str
            Warning message for automatic serial fallback.
        """
        return (
            "Dice.find fell back to serial execution because multiprocessing "
            f"is unsafe from the current entrypoint: {reason}. Use a "
            "file-backed __main__ module or request "
            "SearchExecution.SERIAL explicitly to silence this warning."
        )

    def _unsafe_process_error(self, reason):
        """Build the error message for explicit process execution.

        Parameters
        ----------
        reason : str
            Explanation of why multiprocessing is unsafe.

        Returns
        -------
        message : str
            Error message for explicit process execution.
        """
        return (
            "Dice.find with SearchExecution.PROCESSES requires a file-backed "
            "__main__ module that multiprocessing workers can import. "
            f"Multiprocessing is unsafe from the current entrypoint because "
            f"{reason}."
        )

    def _find_processes(self, cube_list, atom_type, distance, policy):
        """Run the search through a multiprocessing pool.

        Parameters
        ----------
        cube_list : list
            List of cube indices to search in.
        atom_type : list
            List of two atom types.
        distance : list
            Bounds of allowed distance ``[lower, upper]``.
        policy : SearchPolicy
            Execution policy for the search.

        Returns
        -------
        bond_list : list
            Bond array containing lists of atom ids.
        """
        process_count = self._resolve_process_count(policy.processes)
        if process_count < 2:
            raise ValueError(
                "SearchExecution.PROCESSES requires at least two worker "
                "processes. Use SearchExecution.SERIAL for single-process "
                "execution."
            )

        cube_chunks = self._chunk_cube_list(cube_list, process_count)
        context = (
            mp.get_context(policy.start_method)
            if policy.start_method is not None
            else mp.get_context()
        )

        with context.Pool(processes=process_count) as pool:
            results = [
                pool.apply_async(
                    self._find_bond,
                    args=(cube_chunk, atom_type, distance),
                )
                for cube_chunk in cube_chunks
            ]
            bond_list = sum([result.get() for result in results], [])

        return bond_list

    def find(self, cube_list=None, atom_type=None, distance=None, policy=None):
        """Search for atom pairs using the requested execution policy.

        Parameters
        ----------
        cube_list : list
            List of cube indices to search in, use an empty list for all cubes
        atom_type : list
            List of two atom types
        distance : list
            Bounds of allowed distance [lower, upper]
        policy : SearchPolicy, optional
            Execution policy controlling whether the search runs serially or
            with a multiprocessing pool. ``AUTO`` uses process-based search
            only when the current ``__main__`` module is safely importable by
            worker processes. Otherwise it falls back to serial execution and,
            by default, emits a :class:`RuntimeWarning`.

        Returns
        -------
        bond_list : list
            Bond array containing lists of two atom ids

        Raises
        ------
        TypeError
            If ``atom_type`` or ``distance`` is missing, or if ``policy`` is
            not a :class:`SearchPolicy` instance.
        RuntimeError
            If ``SearchExecution.PROCESSES`` is requested from an unsafe
            multiprocessing entrypoint.
        """
        if atom_type is None or distance is None:
            raise TypeError("Dice.find requires atom_type and distance arguments.")

        cube_list = cube_list if cube_list else list(self._pointer.keys())
        policy = policy if policy is not None else SearchPolicy()
        if not isinstance(policy, SearchPolicy):
            raise TypeError("Dice.find policy must be a SearchPolicy instance.")

        if policy.execution is SearchExecution.SERIAL:
            return self._find_bond(cube_list, atom_type, distance)

        if policy.execution is SearchExecution.PROCESSES:
            unsafe_reason = self._unsafe_entrypoint_reason()
            if unsafe_reason is not None:
                raise RuntimeError(self._unsafe_process_error(unsafe_reason))

            return self._find_processes(cube_list, atom_type, distance, policy)

        process_count = self._resolve_process_count(policy.processes)
        if process_count < 2:
            return self._find_bond(cube_list, atom_type, distance)

        unsafe_reason = self._unsafe_entrypoint_reason()
        if unsafe_reason is not None:
            if policy.warn_on_fallback:
                warnings.warn(
                    self._serial_fallback_warning(unsafe_reason),
                    RuntimeWarning,
                    stacklevel=2,
                )
            return self._find_bond(cube_list, atom_type, distance)

        return self._find_processes(cube_list, atom_type, distance, policy)


    ##################
    # Setter methods #
    ##################
    def set_pbc(self, pbc):
        """Turn the periodic boundary conditions on or off.

        Parameters
        ----------
        bond : bool
            True to turn on periodic boundary conditions
        """
        self._is_pbc = pbc


    ##################
    # Getter methods #
    ##################
    def get_origin(self):
        """Return the origin positions of the cubes.

        Returns
        -------
        origin : dictionary
            Dictionary of origin positions for each cube
        """
        return self._origin

    def get_pointer(self):
        """Return the list of atoms in each cube.

        Returns
        -------
        pointer : dictionary
            Pointer dictionary
        """
        return self._pointer

    def get_count(self):
        """Return the number of cubes in each dimension.

        Returns
        -------
        count : list
            Number of cubes
        """
        return self._count

    def get_size(self):
        """Return the cubes size.

        Returns
        -------
        size : integer
            Cube size
        """
        return self._size

    def get_mol(self):
        """Return the molecule.

        Returns
        -------
        mol : Molecule
            Molecule
        """
        return self._mol
