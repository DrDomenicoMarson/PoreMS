################################################################################
# Pore Class                                                                   #
#                                                                              #
"""Surface preparation and functionalization helpers for pore blocks."""
################################################################################


import copy
import random

from collections.abc import Callable
from collections import Counter
from dataclasses import dataclass, field

import porems.geometry as geometry
import porems.generic as generic

from porems.connectivity import AttachmentRecord
from porems.dice import Dice
from porems.molecule import Molecule


_VALID_SITE_TYPES = {"in", "ex"}


@dataclass
class BindingSite:
    """Surface silicon binding site with its remaining oxygen handles.

    Parameters
    ----------
    oxygen_ids : list[int], optional
        Surface oxygen atom identifiers currently bound to the silicon site.
    site_type : str, optional
        Surface family identifier, ``"in"`` for interior or ``"ex"`` for
        exterior.
    is_available : bool, optional
        True when the site can still accept a new attachment.
    normal : callable, optional
        Surface-normal callback used to orient attached molecules at this site.
    """

    oxygen_ids: list[int] = field(default_factory=list)
    site_type: str = "in"
    is_available: bool = True
    normal: Callable[[list], list] | None = None

    @property
    def oxygen_count(self):
        """Return the number of remaining oxygen handles on the site.

        Returns
        -------
        count : int
            Number of oxygen atoms still bound to the silicon site.
        """
        return len(self.oxygen_ids)

    @property
    def is_geminal(self):
        """Return whether the site is geminal.

        Returns
        -------
        is_geminal : bool
            True when two oxygen handles remain on the silicon site.
        """
        return self.oxygen_count == 2


@dataclass(frozen=True)
class SurfaceEditRecord:
    """One tracked surface-edit event recorded during preparation.

    Parameters
    ----------
    atom_id : int
        Atom identifier affected by the edit.
    atom_type : str
        Atom type of the affected atom.
    reason : str
        Reason code describing why the atom was removed or inserted.
    neighbor_ids : tuple[int, ...], optional
        Atom identifiers bonded to the edited atom at the time of the event.
    """

    atom_id: int
    atom_type: str
    reason: str
    neighbor_ids: tuple[int, ...] = ()


@dataclass
class SurfacePreparationDiagnostics:
    """Surface-cleanup and scaffold-validation counters.

    Parameters
    ----------
    stripped_undercoordinated_si : int, optional
        Number of silicon atoms removed because they lost at least one bond
        relative to the original connectivity matrix.
    stripped_excess_surface_oxygen_si : int, optional
        Number of silicon atoms removed because three or more exposed oxygen
        handles remained on the same site during cleanup.
    removed_orphan_oxygen : int, optional
        Number of zero-bond oxygen atoms removed from the active matrix.
    removed_invalid_oxygen : int, optional
        Number of oxygen atoms removed because their active connectivity no
        longer matched a valid silica surface or scaffold role.
    removed_orphan_silicon : int, optional
        Number of zero-bond silicon atoms removed from the active matrix.
    inserted_bridge_oxygen : int, optional
        Number of bridge oxygens inserted during later siloxane editing.
    final_surface_oxygen_handles : int, optional
        Number of valid one-coordinate surface oxygen handles present on the
        final prepared slit surface.
    final_framework_oxygen : int, optional
        Number of valid two-coordinate framework oxygens present in the final
        active scaffold.
    """

    stripped_undercoordinated_si: int = 0
    stripped_excess_surface_oxygen_si: int = 0
    removed_orphan_oxygen: int = 0
    removed_invalid_oxygen: int = 0
    removed_orphan_silicon: int = 0
    inserted_bridge_oxygen: int = 0
    final_surface_oxygen_handles: int = 0
    final_framework_oxygen: int = 0

    @property
    def stripped_silicon_total(self):
        """Return the total number of stripped silicon atoms."""
        return (
            self.stripped_undercoordinated_si
            + self.stripped_excess_surface_oxygen_si
            + self.removed_orphan_silicon
        )


class Pore():
    """Prepared pore block with editable binding sites.

    The class wraps a silica block and its connectivity matrix, prepares
    chemically valid surface sites, tracks interior and exterior attachment
    states, and attaches silanol, siloxane, or user-provided molecules to the
    resulting binding sites.

    Parameters
    ----------
    block : Molecule
        Block molecule that will be interpreted as a pore scaffold.
    matrix : Matrix
        Connectivity matrix describing the bonds in ``block``.
    """
    def __init__(self, block, matrix):
        # Initialize
        self._dim = 3

        self._name = ""
        self._box = []

        self._block = block
        self._matrix = matrix

        self._num_in_ex = 0
        self._oxygen_ex = []
        self._sites = {}
        self.sites_attach_mol = {}
        self._objectified_atoms = set()
        self._surface_edit_history = []
        self._surface_preparation_diagnostics = SurfacePreparationDiagnostics()
        self._attachment_records = []

        self._mol_dict = {"block": {}, "in": {}, "ex": {}}

    def _reset_surface_preparation_tracking(self):
        """Reset internal surface-edit provenance and diagnostics."""
        self._surface_edit_history = []
        self._surface_preparation_diagnostics = SurfacePreparationDiagnostics()

    def _record_surface_edit(self, atom_id, reason):
        """Store one surface-edit event and update diagnostics counters.

        Parameters
        ----------
        atom_id : int
            Atom identifier affected by the edit.
        reason : str
            Reason code describing the edit.
        """
        atom_type = self._block.get_atom_type(atom_id)
        neighbor_ids = ()
        if atom_id in self._matrix.get_matrix():
            neighbor_ids = tuple(self._matrix.get_matrix()[atom_id]["atoms"])

        self._surface_edit_history.append(
            SurfaceEditRecord(
                atom_id=atom_id,
                atom_type=atom_type,
                reason=reason,
                neighbor_ids=neighbor_ids,
            )
        )

        diagnostics = self._surface_preparation_diagnostics
        if reason == "undercoordinated_si":
            diagnostics.stripped_undercoordinated_si += 1
        elif reason == "excess_surface_oxygen_si":
            diagnostics.stripped_excess_surface_oxygen_si += 1
        elif reason == "orphan_oxygen":
            diagnostics.removed_orphan_oxygen += 1
        elif reason == "invalid_oxygen":
            diagnostics.removed_invalid_oxygen += 1
        elif reason == "orphan_silicon":
            diagnostics.removed_orphan_silicon += 1
        elif reason == "inserted_bridge_oxygen":
            diagnostics.inserted_bridge_oxygen += 1

    def _remove_atoms(self, atoms, reason):
        """Remove active atoms from the matrix and record the change.

        Parameters
        ----------
        atoms : list[int]
            Atom identifiers that should be removed from the active matrix.
        reason : str
            Reason code recorded for each removed atom.
        """
        atoms = sorted({atom for atom in atoms if atom in self._matrix.get_matrix()})
        if not atoms:
            return
        for atom_id in atoms:
            self._record_surface_edit(atom_id, reason)
        self._matrix.remove(atoms)

    def _retained_scaffold_oxygen_ids(self, site_id, oxygen_ids):
        """Return scaffold oxygens that stay bonded to an attached mount atom.

        Parameters
        ----------
        site_id : int
            Silicon site id that is about to be consumed.
        oxygen_ids : list[int]
            Surface-handle oxygen ids removed together with the silicon site.

        Returns
        -------
        oxygen_ids : tuple[int, ...]
            Source ids of retained scaffold oxygens bonded to ``site_id``.
        """
        if site_id not in self._matrix.get_matrix():
            return ()

        return tuple(
            sorted(
                atom_id
                for atom_id in self._matrix.get_matrix()[site_id]["atoms"]
                if atom_id not in oxygen_ids and self._block.get_atom_type(atom_id) == "O"
            )
        )

    def _surface_handle_oxygen_ids(self):
        """Return oxygen atoms that currently behave as surface handles.

        Returns
        -------
        oxygen_ids : list[int]
            Degree-one oxygen atoms bonded to exactly one silicon atom.
        """
        oxygen_ids = []
        for atom_id, props in self._matrix.get_matrix().items():
            if self._block.get_atom_type(atom_id) != "O":
                continue
            if len(props["atoms"]) != 1:
                continue
            if self._block.get_atom_type(props["atoms"][0]) != "Si":
                continue
            oxygen_ids.append(atom_id)
        return oxygen_ids

    def _framework_oxygen_ids(self):
        """Return oxygen atoms that qualify as framework oxygens.

        Returns
        -------
        oxygen_ids : list[int]
            Degree-two oxygen atoms bonded to two silicon atoms.
        """
        oxygen_ids = []
        for atom_id, props in self._matrix.get_matrix().items():
            if self._block.get_atom_type(atom_id) != "O":
                continue
            if len(props["atoms"]) != 2:
                continue
            if all(self._block.get_atom_type(neighbor) == "Si" for neighbor in props["atoms"]):
                oxygen_ids.append(atom_id)
        return oxygen_ids

    def _invalid_oxygen_ids(self):
        """Return oxygen atoms whose active connectivity is chemically invalid.

        Returns
        -------
        invalid_groups : tuple[list[int], list[int]]
            Two lists containing orphan oxygens and other invalid oxygens.
        """
        orphan_oxygen = []
        invalid_oxygen = []
        for atom_id, props in self._matrix.get_matrix().items():
            if self._block.get_atom_type(atom_id) != "O":
                continue

            degree = len(props["atoms"])
            neighbor_types = [self._block.get_atom_type(neighbor) for neighbor in props["atoms"]]
            if degree == 0:
                orphan_oxygen.append(atom_id)
            elif degree == 1 and neighbor_types == ["Si"]:
                continue
            elif degree == 2 and all(atom_type == "Si" for atom_type in neighbor_types):
                continue
            else:
                invalid_oxygen.append(atom_id)

        return orphan_oxygen, invalid_oxygen

    def _orphan_silicon_ids(self):
        """Return silicon atoms that no longer have active bonded neighbors.

        Returns
        -------
        silicon_ids : list[int]
            Zero-bond silicon atoms present in the active matrix.
        """
        return [
            atom_id
            for atom_id, props in self._matrix.get_matrix().items()
            if self._block.get_atom_type(atom_id) == "Si" and len(props["atoms"]) == 0
        ]

    def _silicon_with_excess_surface_oxygen(self):
        """Return silicon atoms carrying three or more surface oxygen handles.

        Returns
        -------
        silicon_ids : list[int]
            Silicon identifiers that currently own at least three degree-one
            oxygen handles.
        """
        if not self._surface_handle_oxygen_ids():
            return []

        si_count = Counter(
            sum(
                [self._matrix.get_matrix()[atom]["atoms"] for atom in self._surface_handle_oxygen_ids()],
                [],
            )
        )
        return sorted(si for si, count in si_count.items() if count >= 3)

    def refresh_surface_preparation_diagnostics(self):
        """Refresh the final surface/scaffold counts stored in diagnostics."""
        diagnostics = self._surface_preparation_diagnostics
        diagnostics.final_surface_oxygen_handles = len(self._surface_handle_oxygen_ids())
        diagnostics.final_framework_oxygen = len(self._framework_oxygen_ids())

    def validate_scaffold_atoms(self, atoms):
        """Validate scaffold atoms before they are exported as one-atom residues.

        Parameters
        ----------
        atoms : list[int]
            Active scaffold atom identifiers that are about to be objectified.

        Raises
        ------
        ValueError
            Raised when an atom does not match a valid scaffold role.
        """
        framework_oxygen = set(self._framework_oxygen_ids())
        for atom_id in atoms:
            if atom_id not in self._matrix.get_matrix():
                raise ValueError(f"Scaffold atom {atom_id} is not present in the active matrix.")

            atom_type = self._block.get_atom_type(atom_id)
            if atom_type == "O":
                if atom_id not in framework_oxygen:
                    raise ValueError(
                        f"Scaffold oxygen {atom_id} cannot be exported as OM because it is not a valid two-coordinate framework oxygen."
                    )
            elif atom_type != "Si":
                raise ValueError(
                    f"Unsupported scaffold atom type '{atom_type}' for objectification."
                )


    ###########
    # Surface #
    ###########
    def prepare(self):
        """Prepare the carved silica surface for later functionalization.

        The preparation follows the pore-surface cleanup rules used throughout
        PoreMS: undercoordinated silicon atoms are removed, silicon atoms with
        excessive dangling oxygen neighbors are stripped, and the remaining
        unsaturated oxygen atoms become attachment handles for later surface
        chemistry.
        """
        self._reset_surface_preparation_tracking()

        while True:
            changed = False

            undercoordinated_si = [
                atom_id
                for atom_id, props in self._matrix.get_matrix().items()
                if self._block.get_atom_type(atom_id) == "Si"
                and len(props["atoms"]) < props["bonds"]
            ]
            if undercoordinated_si:
                self._remove_atoms(undercoordinated_si, "undercoordinated_si")
                changed = True

            excess_surface_oxygen_si = self._silicon_with_excess_surface_oxygen()
            if excess_surface_oxygen_si:
                self._remove_atoms(excess_surface_oxygen_si, "excess_surface_oxygen_si")
                changed = True

            orphan_oxygen, invalid_oxygen = self._invalid_oxygen_ids()
            if orphan_oxygen:
                self._remove_atoms(orphan_oxygen, "orphan_oxygen")
                changed = True
            if invalid_oxygen:
                self._remove_atoms(invalid_oxygen, "invalid_oxygen")
                changed = True

            orphan_silicon = self._orphan_silicon_ids()
            if orphan_silicon:
                self._remove_atoms(orphan_silicon, "orphan_silicon")
                changed = True

            if not changed:
                break

        self.refresh_surface_preparation_diagnostics()

    def amorph(self, dist=0.05, accept=None, trials=100):
        """Randomly displace bonded atoms to roughen the local structure.

        Parameters
        ----------
        dist : float, optional
            Maximum displacement per Cartesian component.
        accept : list, optional
            Accepted bond-length interval after displacement.
        trials : int, optional
            Maximum number of displacement attempts per atom.
        """
        accept = [0.1, 0.2] if accept is None else accept

        # Get connectivity matrix
        connect = self._matrix.get_matrix()

        # Run through atoms that have bond partners
        for atom_id in self._matrix.bound(0, "gt"):
            # Create testing atom
            atom_temp = copy.deepcopy(self._block.get_atom_list()[atom_id])
            atom_temp_pos = atom_temp.get_pos()[:]

            # Run through trials
            for i in range(trials):
                # Create random displacement vector
                disp_vec = [random.uniform(-dist, dist) for x in range(self._dim)]
                disp_pos = [atom_temp_pos[x]+disp_vec[x] for x in range(self._dim)]

                # Displace test atom
                atom_temp.set_pos(disp_pos)

                # Calculate new bond lengths
                is_disp = True
                for bond in connect[atom_id]["atoms"]:
                    bond_length = geometry.length(self._block.bond(atom_id, bond))
                    if bond_length < accept[0] or bond_length > accept[1]:
                        is_disp = False
                        break

                # Displace if new bond length is in acceptance range
                if is_disp:
                    self._block.get_atom_list()[atom_id].set_pos(disp_pos)
                    break

    def exterior(self):
        """Create exterior surface sites for reservoir-facing functionalization.

        Periodic Si-O bonds crossing the box boundary are split, replacement
        oxygen atoms are added on the exposed silicon atoms, and the block is
        shifted so the newly exposed outer surfaces can later be populated with
        molecules.
        """
        # Initialize
        box = self._block.get_box()
        atom_list = self._block.get_atom_list()
        bound_list = self._matrix.get_matrix()

        # Get gap
        gap = [-2*x for x in self._block.zero()]

        # Get list of all si atoms
        si_list = [atom_id for atom_id, atom in enumerate(self._block.get_atom_list()) if atom.get_atom_type()=="Si"]

        # Run through silicon atoms
        add_list = []
        break_list = []
        for si in si_list:
            if si not in bound_list:
                continue
            # Run through bound oxygen atoms
            for o in bound_list[si]["atoms"]:
                # Calculate bond vector
                bond_vector = [atom_list[si].get_pos()[dim]-atom_list[o].get_pos()[dim] for dim in range(3)]

                # Check if z dimension of bond - after rotation in pattern class - goes over boundary
                if abs(bond_vector[2]) > box[2]/2:
                    # Get bond direction
                    r = -0.155 if bond_vector[2] < 0 else 0.155

                    # Get bond angle
                    theta = geometry.angle_azi(bond_vector, is_deg=True)
                    phi = geometry.angle_polar(bond_vector, is_deg=True)
                    theta = theta-180 if r < 0 else theta

                    # Add new oxygen atom
                    self._block.add("O", si, r=r, theta=theta, phi=phi)

                    # Add list for new atoms
                    add_list.append([si, self._block.get_num()-1])

                    # Add to break bond list
                    break_list.append([si, o])

        # Translate initial gap
        self._block.zero()
        self._block.translate(gap)

        # Break periodic bonds
        for bond in break_list:
            self._matrix.split(bond[0], bond[1])

        # Add bonds to new oxygens
        for bond in add_list:
            self._matrix.add(bond[0], bond[1])

        # Add atoms to exterior oxygen list
        self.prepare()
        self._oxygen_ex = self._surface_handle_oxygen_ids()

    def _site_search_molecule(self, sites):
        """Create a zero-based temporary molecule for local site searches.

        Parameters
        ----------
        sites : list
            Silicon site identifiers used for a local proximity search.

        Returns
        -------
        site_molecule : Molecule
            Temporary molecule containing deep-copied silicon atoms translated
            so that all coordinates are non-negative.
        """
        site_atoms = [copy.deepcopy(self._block.get_atom_list()[atom]) for atom in sites]
        site_molecule = Molecule(inp=site_atoms)
        site_molecule.zero()

        return site_molecule

    def sites(self):
        """Build the internal binding-site registry.

        Each silicon surface site is mapped to a :class:`BindingSite`
        containing the currently exposed oxygen atoms, the site type
        (``"in"`` or ``"ex"``), and the availability flag updated during
        later attachment steps.
        """
        # Get list of surface oxygen atoms
        oxygen_list = self._surface_handle_oxygen_ids()
        connect = self._matrix.get_matrix()

        # Create binding site dictionary
        self._num_in_ex = 0
        self._sites = {}
        for o in oxygen_list:
            site_id = connect[o]["atoms"][0]
            if site_id not in self._sites:
                self._sites[site_id] = BindingSite()
            self._sites[site_id].oxygen_ids.append(o)

        # Fill other information
        for si, data in self._sites.items():
            # Site type
            is_in = False
            is_ex = False
            for o in data.oxygen_ids:
                if o in self._oxygen_ex:
                    is_ex = True
                else:
                    is_in = True

            data.site_type = "ex" if is_ex else "in"

            if is_in and is_ex:
                self._num_in_ex += 1

            # State
            data.is_available = True


    #######################
    # Molecule Attachment #
    #######################
    def _validate_site_type(self, site_type):
        """Validate a pore surface site type.

        Parameters
        ----------
        site_type : str
            Requested site type.

        Raises
        ------
        ValueError
            Raised when the site type is not supported.
        """
        if site_type not in _VALID_SITE_TYPES:
            raise ValueError(
                f"Unsupported site_type '{site_type}'. Expected one of: {sorted(_VALID_SITE_TYPES)}."
            )

    def attach(self, mol, mount, axis, sites, amount, scale=1, trials=1000, pos_list=None, site_type="in", is_proxi=True, is_random=True, is_rotate=False, is_g=True):
        """Attach molecules to available pore surface sites.

        Parameters
        ----------
        mol : Molecule
            Molecule to attach.
        mount : int
            Atom id on ``mol`` placed onto the selected silicon site.
        axis : list
            Two atom ids defining the molecule orientation axis.
        sites : list
            Silicon site ids from which attachment positions are chosen.
        amount : int
            Number of molecules to attach.
        scale : float, optional
            Effective lateral spacing multiplier for proximity searches.
        trials : int, optional
            Number of random site-selection attempts per molecule.
        pos_list : list, optional
            Explicit Cartesian target positions used to pick nearest free sites.
        site_type : str, optional
            Site family, either interior ``"in"`` or exterior ``"ex"``.
        is_proxi : bool, optional
            True to consume nearby sites with silanol fills after attachment.
        is_random : bool, optional
            True to choose sites randomly from ``sites``.
        is_rotate : bool, optional
            True to allow random rotation around the molecule axis before
            placement.
        is_g : bool, optional
            True to allow geminal surface sites as mounting positions.

        Returns
        -------
        mol_list : list
            Attached molecule copies in placement order.

        Raises
        ------
        ValueError
            Raised when the requested site type is not supported.
        """
        self._validate_site_type(site_type)
        pos_list = [] if pos_list is None else pos_list

        # Rotate molecule towards z-axis
        mol_axis = mol.bond(*axis)
        mol.rotate(geometry.cross_product(mol_axis, [0, 0, 1]), geometry.angle(mol_axis, [0, 0, 1]))
        mol.zero()

        # Search for overlapping placements - Calculate diameter and add carbon VdW-raidus (Wiki)
        if is_proxi:
            mol_diam = (max(mol.get_box()[:2])+0.17)*scale
            si_dice = Dice(self._site_search_molecule(sites), mol_diam, True)
            si_proxi = si_dice.find(None, ["Si", "Si"], [-mol_diam, mol_diam])
            si_matrix = {x[0]: x[1] for x in si_proxi}

        # Run through number of binding sites to add
        mol_list = []
        for i in range(amount):
            si = None
            # Find nearest free site if a position list is given
            if pos_list:
                pos = pos_list[i]
                min_dist = 100000000
                for site in sites:
                    if self._sites[site].is_available:
                        length = geometry.length(geometry.vector(self._block.pos(site), pos))
                        if length < min_dist:
                            si = site
                            min_dist = length
            # Randomly pick an available site
            elif is_random:
                for j in range(trials):
                    si_rand = random.choice(sites)
                    if self._sites[si_rand].is_available:
                        if is_g is False and self._sites[si_rand].is_geminal:
                            pass  
                        else: 
                            si = si_rand                  
                            break
            # Or use next binding site in given list
            else:
                si = sites[i] if i<len(sites) else None

            # Place molecule on surface
            if si is not None and self._sites[si].is_available: 
                if is_g is False and self._sites[si].is_geminal:
                    pass
                else:
                    # Disable binding site
                    self._sites[si].is_available = False
                
                    # Create a copy of the molecule
                    mol_temp = copy.deepcopy(mol)

                    # Check if geminal
                    if self._sites[si].is_geminal:
                        mol_temp.add("O", mount, r=0.164, theta=45)
                        mol_temp.add("H", mol_temp.get_num()-1, r=0.098)
                        mol_temp.set_name(mol.get_name()+"g")
                        mol_temp.set_short(mol.get_short()+"G")

                    # Rotate molecule towards surface normal vector
                    surf_axis = self._sites[si].normal(self._block.pos(si))
                    mol_temp.rotate(geometry.cross_product([0, 0, 1], surf_axis), -geometry.angle([0, 0, 1], surf_axis))

                    # Move molecule to mounting position
                    mol_temp.move(mount, self._block.pos(si))

                    # Add molecule to molecule list and global dictionary
                    mol_list.append(mol_temp)
                    if not mol_temp.get_short() in self._mol_dict[site_type]:
                        self._mol_dict[site_type][mol_temp.get_short()] = []
                    self._mol_dict[site_type][mol_temp.get_short()].append(mol_temp)

                    scaffold_oxygen_source_ids = self._retained_scaffold_oxygen_ids(
                        si,
                        self._sites[si].oxygen_ids,
                    )
                    self._attachment_records.append(
                        AttachmentRecord(
                            site_id=si,
                            site_type=site_type,
                            mount_atom_local_id=mount,
                            is_geminal=self._sites[si].is_geminal,
                            scaffold_oxygen_source_ids=scaffold_oxygen_source_ids,
                            surface_oxygen_source_ids=tuple(self._sites[si].oxygen_ids),
                            molecule=mol_temp,
                        )
                    )

                    # Remove bonds of occupied binding site
                    self._matrix.remove([si] + self._sites[si].oxygen_ids)

                    # Recursively fill sites in proximity with silanol and geminal silanol
                    if is_proxi:
                        proxi_list = [sites[x] for x in si_matrix[sites.index(si)]]
                        if len(proxi_list) > 0:
                            mol_list.extend(self.attach(generic.silanol(), 0, [0, 1], proxi_list, len(proxi_list), site_type=site_type, is_proxi=False, is_random=False))
        return mol_list

    def siloxane(self, sites, amount, slx_dist=None, trials=1000, site_type="in"):
        """Convert neighboring silanol sites into siloxane bridges.

        Candidate silicon pairs are searched within ``slx_dist`` and, when
        available, converted from two hydroxylated sites into one bridging
        siloxane oxygen placed between them.

        Parameters
        ----------
        sites : list
            Silicon site ids considered for bridge formation.
        amount : int
            Number of bridges to attempt.
        slx_dist : list
            Accepted silicon-silicon distance interval ``[lower, upper]``.
        trials : int, optional
            Number of random pair-selection attempts per bridge.
        site_type : str, optional
            Site family, either interior ``"in"`` or exterior ``"ex"``.

        Returns
        -------
        mol_list : list
            Added siloxane bridge molecules.

        Raises
        ------
        ValueError
            Raised when the requested site type is not supported.
        """
        self._validate_site_type(site_type)
        slx_dist = [0.507-1e-2, 0.507+1e-2] if slx_dist is None else slx_dist

        # Create siloxane molecule
        mol = Molecule("siloxane", "SLX")
        mol.add("O", [0, 0, 0], name="OM1")
        mol.add("O", 0, r=0.09, name="OM1")
        mount = 0
        axis = [0, 1]

        # Rotate molecule towards z-axis
        mol_axis = mol.bond(*axis)
        mol.rotate(geometry.cross_product(mol_axis, [0, 0, 1]), geometry.angle(mol_axis, [0, 0, 1]))
        mol.zero()

        # Search for silicon atoms near each other
        si_dice = Dice(self._site_search_molecule(sites), slx_dist[1], False)
        si_proxi = si_dice.find(None, ["Si", "Si"], slx_dist)
        si_matrix = {x[0]: x[1] for x in si_proxi}

        # Run through number of siloxan bridges to add
        bond_matrix = self._matrix.get_matrix()
        mol_list = []
        for i in range(amount):
            # Randomly pick an available site pair
            si = []
            for j in range(trials):
                si_rand = random.choice(sites)
                # Check if binding site in local si-si matrix and if it contains binding partners
                if sites.index(si_rand) in si_matrix and si_matrix[sites.index(si_rand)]:
                    # Choose first binding partner in list
                    si_rand_proxi = sites[si_matrix[sites.index(si_rand)][0]]
                    # Check if binding partner is in local si-si matrix
                    if sites.index(si_rand_proxi) in si_matrix:
                        # Check if unbound states
                        if self._sites[si_rand].is_available and self._sites[si_rand_proxi].is_available:
                            # Check if binding site silicon atoms are already connected with an oxygen
                            is_connected = False
                            for atom_o in bond_matrix[si_rand]["atoms"]:
                                if atom_o in bond_matrix[si_rand_proxi]["atoms"]:
                                    is_connected = True
                            # Add to siloxane list if not connected
                            if not is_connected:
                                si = [si_rand, si_rand_proxi]
                                # End trials if appended
                                break

            # Place molecule on surface
            if si and self._sites[si[0]].is_available and self._sites[si[1]].is_available:
                # Create a copy of the molecule
                mol_temp = copy.deepcopy(mol)

                # Calculate center position
                pos_vec_halve = [x/2 for x in geometry.vector(self._block.pos(si[0]), self._block.pos(si[1]))]
                center_pos = [pos_vec_halve[x]+self._block.pos(si[0])[x] for x in range(self._dim)]

                # Rotate molecule towards surface normal vector
                surf_axis = self._sites[si[0]].normal(center_pos)
                mol_temp.rotate(geometry.cross_product([0, 0, 1], surf_axis), -geometry.angle([0, 0, 1], surf_axis))

                # Move molecule to mounting position and remove temporary atom
                mol_temp.move(mount, center_pos)
                mol_temp.delete(0)

                # Add molecule to molecule list and global dictionary
                mol_list.append(mol_temp)
                if not mol_temp.get_short() in self._mol_dict[site_type]:
                    self._mol_dict[site_type][mol_temp.get_short()] = []
                self._mol_dict[site_type][mol_temp.get_short()].append(mol_temp)

                # Remove oxygen atom and if not geminal delete site
                for si_id in si:
                    self._matrix.remove(self._sites[si_id].oxygen_ids[0])
                    if self._sites[si_id].is_geminal:
                        self._sites[si_id].oxygen_ids.pop(0)
                    else:
                        del self._sites[si_id]
                    del si_matrix[sites.index(si_id)]

        return mol_list

    def fill_sites(self, sites, site_type):
        """Fill remaining free sites with silanol or geminal silanol groups.

        Parameters
        ----------
        sites : list
            Silicon site ids to fill.
        site_type : str
            Site family, either interior ``"in"`` or exterior ``"ex"``.

        Returns
        -------
        mol_list : list
            Added silanol molecules.
        """
        mol_list = self.attach(generic.silanol(), 0, [0, 1], sites, len(sites), site_type=site_type, is_proxi=False, is_random=False)
        
        return mol_list


    ###############
    # Final Edits #
    ###############
    def objectify(self, atoms):
        """Convert standalone scaffold atoms into one-atom molecule objects.

        Parameters
        ----------
        atoms : list
            Atom ids to convert.

        Returns
        -------
        mol_list : list
            One-atom molecule objects created from the selected atoms.
        """
        self.validate_scaffold_atoms(atoms)

        # Initialize
        mol_list = []

        # Run through all remaining atoms with a bond or more
        for atom_id in atoms:
            if atom_id in self._objectified_atoms:
                continue

            # Get atom object
            atom = self._block.get_atom_list()[atom_id]

            # Create molecule object
            if atom.get_atom_type() == "O":
                mol = Molecule("om", "OM")
                mol.add("O", atom.get_pos(), name="OM1", source_id=atom_id)
            elif atom.get_atom_type() == "Si":
                mol = Molecule("si", "SI")
                mol.add("Si", atom.get_pos(), name="SI1", source_id=atom_id)

            # Add to molecule list and global dictionary
            mol_list.append(mol)
            if not mol.get_short() in self._mol_dict["block"]:
                self._mol_dict["block"][mol.get_short()] = []
            self._mol_dict["block"][mol.get_short()].append(mol)
            self._objectified_atoms.add(atom_id)

        # Output
        return mol_list

    def reservoir(self, size):
        """Translate the pore content into a box with solvent reservoirs.

        Parameters
        ----------
        size : float
            Reservoir length added on each side along ``z`` in nanometers.
        """
        # Convert molecule dict into list
        mol_list = sum([x for x in self.get_mol_dict().values()], [])

        # Get zero translation
        min_z = 1000000
        max_z = 0
        for mol in mol_list:
            col_z = mol.column_pos()[2]
            min_z = min(col_z) if min(col_z) < min_z else min_z
            max_z = max(col_z) if max(col_z) > max_z else max_z

        # Translate all molecules
        for mol in mol_list:
            mol.translate([0, 0, -min_z+size])

        # Set new box size
        box = self._block.get_box()
        self.set_box([box[0], box[1], max_z-min_z+2*size])


    ##################
    # Setter Methods #
    ##################
    def set_name(self, name):
        """Set the pore name.

        Parameters
        ----------
        name : str
            Pore name.
        """
        self._name = name

    def set_box(self, box):
        """Set the pore simulation box dimensions.

        Parameters
        ----------
        box : list
            Box lengths in all dimensions.
        """
        self._box = box


    ##################
    # Getter Methods #
    ##################
    def get_name(self):
        """Return the pore name.

        Returns
        -------
        name : str
            Pore name.
        """
        return self._name

    def get_box(self):
        """Return the box size of the pore.

        Returns
        -------
        box : list
            Box lengths in all dimensions.
        """
        return self._box

    def get_block(self):
        """Return the block molecule.

        Returns
        -------
        block : Molecule
            Underlying block molecule.
        """
        return self._block

    def get_sites(self):
        """Return the binding-site registry.

        Returns
        -------
        sites : dict[int, BindingSite]
            Binding sites keyed by silicon atom id.
        """
        return self._sites

    def get_mol_dict(self):
        """Return all molecules grouped by residue short name.

        Returns
        -------
        mol_dict : dict
            Molecule dictionary merged across interior, exterior, and block
            groups.
        """
        mol_dict = {}
        for site_type in self._mol_dict.keys():
            for key, item in self._mol_dict[site_type].items():
                if not key in mol_dict.keys():
                    mol_dict[key] = []
                mol_dict[key].extend(item)

        return mol_dict

    def get_surface_preparation_diagnostics(self):
        """Return a snapshot of the surface-preparation diagnostics.

        Returns
        -------
        diagnostics : SurfacePreparationDiagnostics
            Surface-cleanup and scaffold-validation counters collected for the
            current pore.
        """
        return copy.deepcopy(self._surface_preparation_diagnostics)

    def get_surface_edit_history(self):
        """Return the recorded surface-edit provenance entries.

        Returns
        -------
        history : list[SurfaceEditRecord]
            Chronological list of surface-edit records collected during
            preparation and later bridge insertion.
        """
        return list(self._surface_edit_history)

    def get_attachment_records(self):
        """Return metadata for molecules attached to the prepared pore surface.

        Returns
        -------
        records : list[AttachmentRecord]
            Attachment records in placement order.
        """
        return list(self._attachment_records)

    def get_site_dict(self):
        """Return molecules grouped by site family.

        Returns
        -------
        site_dict : dict
            Molecule dictionary split into ``"block"``, ``"in"``, and ``"ex"``.
        """
        return self._mol_dict

    def get_num_in_ex(self):
        """Return the number of mixed interior/exterior geminal sites.

        Returns
        -------
        num_in_ex : int
            Number of geminal silicon sites spanning both interior and exterior
            oxygen assignments.
        """
        return self._num_in_ex
