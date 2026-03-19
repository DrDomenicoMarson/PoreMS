################################################################################
# Basic Pore System Classes                                                    #
#                                                                              #
"""High-level pore builders and geometry analysis helpers."""
################################################################################


import os
import math
import warnings
from dataclasses import dataclass
import yaml
import pandas as pd
import porems as pms
import numpy as np

from porems.shape import ConeConfig, CuboidConfig, CylinderConfig, ShapeSection, ShapeSpec, SphereConfig


_VALID_SHAPE_TYPES = {"CYLINDER", "SLIT", "SPHERE", "CONE"}
_VALID_ATTACH_INPUTS = {"num", "molar", "percent"}
_VALID_SITE_TYPES = {"in", "ex"}
_VALID_SPECIAL_SYMMETRIES = {"point", "mirror"}


@dataclass
class _ShapeAnalysis:
    """Cached geometry metadata for one configured pore shape."""

    shape_id: int
    shape_type: str
    centroid: list
    central: list
    extent: float
    diameter: float | None = None
    diameter_1: float | None = None
    z_min: float = 0.0
    z_max: float = 0.0
    rotated_centroid: list | None = None
    x_min: float | None = None
    x_max: float | None = None

    @classmethod
    def from_shape(cls, shape_id, shape_spec):
        """Build analysis metadata from one stored shape entry."""
        shape_type = shape_spec.shape_type
        config = shape_spec.shape.get_config()
        centroid = list(config.centroid)
        extent = config.diameter if shape_type == "SPHERE" else config.length
        z_padding = 0 if shape_type == "SPHERE" else 0.1

        rotated_centroid = None
        x_min = None
        x_max = None
        if shape_type != "SPHERE":
            rotated_centroid = [0, 0, 0]
            rotated_centroid[0] = centroid[0] * np.cos(-np.pi / 4) - centroid[1] * np.sin(-np.pi / 4)
            rotated_centroid[1] = centroid[0] * np.sin(-np.pi / 4) + centroid[1] * np.cos(-np.pi / 4)
            x_min = rotated_centroid[0] - 0.2
            x_max = rotated_centroid[0] + 0.2

        return cls(
            shape_id=shape_id,
            shape_type=shape_type,
            centroid=centroid,
            central=list(config.central),
            extent=extent,
            diameter=getattr(config, "diameter", None),
            diameter_1=getattr(config, "diameter_1", None),
            z_min=centroid[2] - extent / 2 + z_padding,
            z_max=centroid[2] + extent / 2 - z_padding,
            rotated_centroid=rotated_centroid,
            x_min=x_min,
            x_max=x_max,
        )

    def matches_site(self, pos):
        """Return True when a binding-site position belongs to the shape."""
        radi = pms.geom.length(pms.geom.vector([self.centroid[0], self.centroid[1], pos[2]], pos))
        z_min = self.centroid[2] - self.extent / 2
        z_max = self.centroid[2] + self.extent / 2

        if self.shape_type == "CYLINDER":
            return z_min < pos[2] < z_max and radi < ((self.diameter * 1.5) / 2)
        if self.shape_type == "CONE":
            return z_min < pos[2] < z_max and radi < ((self.diameter_1 * 1.5) / 2)
        if self.shape_type == "SLIT":
            return True
        if self.shape_type == "SPHERE":
            return z_min < pos[2] < z_max and radi < ((self.diameter * 1.05) / 2)
        return False

    def radius_from_position(self, pos):
        """Return the relevant analysis radius for one position."""
        if self.shape_type == "CYLINDER":
            if self.z_min < pos[2] < self.z_max and self.central == [0, 0, 1]:
                return pms.geom.length(pms.geom.vector([self.centroid[0], self.centroid[1], pos[2]], pos))

            if self.central == [1, 1, 0] and self.rotated_centroid is not None:
                pos_new = [0, 0, 0]
                pos_new[0] = pos[0] * np.cos(-np.pi / 4) - pos[1] * np.sin(-np.pi / 4)
                pos_new[1] = pos[0] * np.sin(-np.pi / 4) + pos[1] * np.cos(-np.pi / 4)
                if self.x_min < pos_new[0] < self.x_max:
                    radius = pms.geom.length(
                        pms.geom.vector(
                            [pos_new[0], self.rotated_centroid[1], self.rotated_centroid[2]],
                            pos_new,
                        )
                    )
                    diameter_inp = self.diameter + 0.5
                    if (diameter_inp / 2) * 1.1 > radius > (diameter_inp / 2) * 0.9:
                        return radius

        elif self.shape_type == "SLIT":
            return pms.geom.length(pms.geom.vector([pos[0], self.centroid[1], pos[2]], pos))

        elif self.shape_type == "SPHERE":
            if self.z_min < pos[2] < self.z_max and self.central == [0, 0, 1]:
                return pms.geom.length(pms.geom.vector(self.centroid, pos))

        elif self.shape_type == "CONE":
            if self.z_min < pos[2] < self.z_max and self.central == [0, 0, 1]:
                return pms.geom.length(pms.geom.vector([self.centroid[0], self.centroid[1], pos[2]], pos))

        return None


def _normalize_site_type(site_type):
    """Map public surface identifiers to dataclass attribute names."""
    if site_type == "in":
        return "interior"
    if site_type == "ex":
        return "exterior"
    raise ValueError(f"Unsupported site_type '{site_type}'.")


@dataclass(frozen=True)
class RoughnessProfile:
    """Surface roughness values for interior and exterior pore surfaces.

    Parameters
    ----------
    interior : list[float]
        Roughness values for each configured interior shape.
    exterior : float
        Roughness value for the exterior surface.
    """

    interior: list[float]
    exterior: float

    def for_site_type(self, site_type):
        """Return the roughness values for one surface family.

        Parameters
        ----------
        site_type : str
            Surface identifier, ``"in"`` or ``"ex"``.

        Returns
        -------
        values : list[float] or float
            Interior roughness values for ``"in"`` or the exterior roughness
            value for ``"ex"``.
        """
        return getattr(self, _normalize_site_type(site_type))

    def to_dict(self):
        """Return a serializable mapping representation.

        Returns
        -------
        payload : dict
            Mapping with legacy ``"in"`` and ``"ex"`` keys.
        """
        return {"in": list(self.interior), "ex": self.exterior}


@dataclass(frozen=True)
class SurfaceAreaSummary:
    """Surface areas for interior and exterior pore surfaces.

    Parameters
    ----------
    interior : float or list[float]
        Total interior surface area or per-shape areas.
    exterior : float or list[float]
        Total exterior surface area or per-endcap areas.
    """

    interior: float | list[float]
    exterior: float | list[float]

    def for_site_type(self, site_type):
        """Return the surface areas for one surface family.

        Parameters
        ----------
        site_type : str
            Surface identifier, ``"in"`` or ``"ex"``.

        Returns
        -------
        values : float or list[float]
            Interior or exterior surface areas.
        """
        return getattr(self, _normalize_site_type(site_type))

    def to_dict(self):
        """Return a serializable mapping representation.

        Returns
        -------
        payload : dict
            Mapping with legacy ``"in"`` and ``"ex"`` keys.
        """
        interior = list(self.interior) if isinstance(self.interior, list) else self.interior
        exterior = list(self.exterior) if isinstance(self.exterior, list) else self.exterior
        return {"in": interior, "ex": exterior}


@dataclass(frozen=True)
class SurfaceAllocationStats:
    """Allocation statistics for one surface family.

    Parameters
    ----------
    count : int
        Number of attached molecules or sites.
    density_nm2 : float
        Number density per square nanometer.
    density_mumol_m2 : float
        Surface density in :math:`\\frac{\\mu\\text{mol}}{\\text{m}^2}`.
    """

    count: int
    density_nm2: float
    density_mumol_m2: float

    def to_list(self):
        """Return the historical list representation.

        Returns
        -------
        values : list[float]
            ``[count, density_nm2, density_mumol_m2]`` representation.
        """
        return [self.count, self.density_nm2, self.density_mumol_m2]


@dataclass(frozen=True)
class AllocationSummary:
    """Allocation statistics for interior and exterior surfaces.

    Parameters
    ----------
    interior : SurfaceAllocationStats
        Statistics for the interior surface family.
    exterior : SurfaceAllocationStats
        Statistics for the exterior surface family.
    """

    interior: SurfaceAllocationStats
    exterior: SurfaceAllocationStats

    def for_site_type(self, site_type):
        """Return allocation statistics for one surface family.

        Parameters
        ----------
        site_type : str
            Surface identifier, ``"in"`` or ``"ex"``.

        Returns
        -------
        stats : SurfaceAllocationStats
            Allocation statistics for the requested surface family.
        """
        return getattr(self, _normalize_site_type(site_type))

    def to_dict(self):
        """Return a serializable mapping representation.

        Returns
        -------
        payload : dict
            Mapping with legacy ``"in"`` and ``"ex"`` keys.
        """
        return {
            "in": self.interior.to_list(),
            "ex": self.exterior.to_list(),
        }


class PoreKit():
    """Composable builder for pore systems carved from silica blocks.

    ``PoreKit`` exposes the low-level workflow used by the higher-level pore
    convenience classes: provide a structure, build its connectivity, define one
    or more pore shapes, prepare the surface, attach molecules, then finalize
    and store the result.
    """
    def __init__(self):
        # Initialize
        self._sort_list = ["OM", "SI"]
        self._res = 0
        self._hydro = {"in": [], "ex": 0}
        self._shapes = []
        self._yml = {}

    def structure(self, structure):
        """Set the silica structure used by the pore builder.

        Parameters
        ----------
        structure : Molecule
            Input silica structure as a :class:`porems.molecule.Molecule`.
        """
        # Globalize crystal structure
        self._block = structure
        self._box = self._block.get_box()
        self._centroid = self._block.centroid()

    def build(self, bonds=None):
        """Build the Si-O connectivity matrix and base pore object.

        Parameters
        ----------
        bonds : list, optional
            Accepted Si-O bond-length interval used during connectivity search.
        """
        bonds = [0.155-1e-2, 0.155+1e-2] if bonds is None else bonds

        # Dice up block
        dice = pms.Dice(self._block, 0.4, True)
        self._matrix = pms.Matrix(dice.find(None, ["Si", "O"], bonds))

        # Create pore object
        self._pore = pms.Pore(self._block, self._matrix)
        self._pore.set_name("pore")
        self._pore.set_box(self._box)

    ##########
    # Shapes #
    ##########
    def exterior(self, res, hydro=0):
        """Expose and configure the exterior surface.

        Parameters
        ----------
        res : float
            Reservoir length in nanometers added on each side during
            finalization.
        hydro : float, optional
            Target hydroxylation degree for the exterior surface in
            :math:`\\frac{\\mu\\text{mol}}{\\text{m}^2}`
            Leave zero to keep the original exterior hydroxylation.
        """
        self._pore.exterior()
        self._hydro["ex"] = hydro
        self._res = res

    def add_shape(self, shape, section=None, hydro=0):
        """Register one pore shape for drilling and later analysis.

        Parameters
        ----------
        shape : ShapeSpec
            Typed shape entry containing the shape identifier and object.
        section : ShapeSection or dict, optional
            Optional coordinate ranges used to assign interior sites to this
            shape after preparation.
        hydro : float, optional
            Target hydroxylation degree for the interior surface in
            :math:`\\frac{\\mu\\text{mol}}{\\text{m}^2}`

        Raises
        ------
        TypeError
            Raised when ``shape`` is not a :class:`porems.shape.ShapeSpec` or
            when ``section`` is not a :class:`porems.shape.ShapeSection` or
            mapping.
        ValueError
            Raised when an unsupported shape type is provided.
        """
        if not isinstance(shape, ShapeSpec):
            raise TypeError("shape must be provided as a ShapeSpec instance.")

        section_spec = self._coerce_shape_section(section) if section is not None else shape.section
        self._validate_shape_type(shape.shape_type)

        # Process user input
        shape = ShapeSpec(shape.shape_type, shape.shape, section_spec)
        self._hydro["in"].append(hydro)

        # Append to shape list
        self._shapes.append(shape)

    def shape_cylinder(self, diam, length=0, centroid=None, central=None):
        """Create a cylindrical drilling shape.

        Parameters
        ----------
        diam : float
            Cylinder diameter in nanometers.
        length : float, optional
            Cylinder length in nanometers. Use zero for the full box length.
        centroid : list, optional
            Cylinder centroid. Defaults to the system centroid.
        central : list, optional
            Central axis for cylinder. Defaults to the z-axis.

        Returns
        -------
        shape : ShapeSpec
            Shape entry ready for :meth:`add_shape`.
        """
        central = [0, 0, 1] if central is None else central

        # Process user input
        centroid = centroid if centroid else self.centroid()
        length = length if length else self._box[2]

        # Define shape
        cylinder = pms.Cylinder(
            CylinderConfig(
                centroid=tuple(centroid),
                central=tuple(central),
                length=length,
                diameter=diam-0.5,
            )
        )  # Preparation precaution

        return ShapeSpec("CYLINDER", cylinder)

    def shape_slit(self, height, length=0, centroid=None, central=None):
        """Create a slit-pore drilling shape.

        Parameters
        ----------
        height : float
            Slit height in nanometers.
        length : float, optional
            Slit length in nanometers. Use zero for the full box length.
        centroid : list, optional
            Slit centroid. Defaults to the system centroid.
        central : list, optional
            Central axis for the slit. Defaults to the z-axis.

        Returns
        -------
        shape : ShapeSpec
            Shape entry ready for :meth:`add_shape`.
        """
        central = [0, 0, 1] if central is None else central

        # Process user input
        centroid = centroid if centroid else self.centroid()
        length = length if length else self._box[2]

        # Define shape
        cuboid = pms.Cuboid(
            CuboidConfig(
                centroid=tuple(centroid),
                central=tuple(central),
                length=length,
                width=self._box[0],
                height=height-0.5,
            )
        )  # Preparation precaution

        return ShapeSpec("SLIT", cuboid)

    def shape_sphere(self, diameter, centroid=None, central=None):
        """Create a spherical drilling shape.

        Parameters
        ----------
        diameter : float
            Sphere diameter in nanometers.
        centroid : list, optional
            Sphere centroid. Defaults to the system centroid.
        central : list, optional
            Central axis for sphere orientation. Defaults to the z-axis.

        Returns
        -------
        shape : ShapeSpec
            Shape entry ready for :meth:`add_shape`.
        """
        central = [0, 0, 1] if central is None else central

        # Process user input
        centroid = centroid if centroid else self.centroid()

        # Define shape
        sphere = pms.Sphere(
            SphereConfig(
                centroid=tuple(centroid),
                central=tuple(central),
                diameter=diameter,
            )
        )

        return ShapeSpec("SPHERE", sphere)

    def shape_cone(self, diam_1, diam_2, length=0, centroid=None, central=None):
        """Create a conical drilling shape.

        Parameters
        ----------
        diam_1 : float
            First cone diameter in nanometers.
        diam_2 : float
            Second cone diameter in nanometers.
        length : float, optional
            Cone length in nanometers. Use zero for the full box length.
        centroid : list, optional
            Cone centroid. Defaults to the system centroid.
        central : list, optional
            Central axis for cone. Defaults to the z-axis.

        Returns
        -------
        shape : ShapeSpec
            Shape entry ready for :meth:`add_shape`.
        """
        central = [0, 0, 1] if central is None else central

        # Process user input
        centroid = centroid if centroid else self.centroid()
        length = length if length else self._box[2]

        # Define shape
        cone = pms.Cone(
            ConeConfig(
                centroid=tuple(centroid),
                central=tuple(central),
                length=length,
                diameter_1=diam_1-0.5,
                diameter_2=diam_2-0.5,
            )
        )  # Preparation precaution

        return ShapeSpec("CONE", cone)

    def _has_interior_hydroxylation_target(self):
        """Check whether the interior surface hydroxylation should be adjusted.

        Returns
        -------
        has_target : bool
            True if at least one interior shape requests a positive
            hydroxylation target.
        """
        return any(hydro > 0 for hydro in self._hydro["in"])

    def _init_site_tracking(self):
        """Initialize site classification containers.

        This helper keeps the interior-site bookkeeping available even when no
        siloxane adjustment is requested.
        """
        self.sites_shape = {}
        self._pore.sites_attach_mol = {}

        for i, shape in enumerate(self._shapes):
            self.sites_shape[i] = []
            self._pore.sites_attach_mol[i] = {}

    def _shape_config(self, shape_spec):
        """Return the typed configuration for one stored shape.

        Parameters
        ----------
        shape_spec : ShapeSpec
            Stored shape entry.

        Returns
        -------
        config : ShapeConfig
            Typed shape configuration.
        """
        return shape_spec.shape.get_config()

    def _shape_ranges(self, shape_spec):
        """Return normalized section ranges for one stored shape.

        Parameters
        ----------
        shape_spec : ShapeSpec
            Stored shape entry.

        Returns
        -------
        ranges : dict
            Axis-index mapping used for section checks.
        """
        return shape_spec.section.to_ranges(self._box)

    def _validate_shape_type(self, shape_type):
        """Validate a shape identifier.

        Parameters
        ----------
        shape_type : str
            Requested shape type.

        Raises
        ------
        ValueError
            Raised when the shape type is not supported.
        """
        if shape_type not in _VALID_SHAPE_TYPES:
            raise ValueError(
                f"Unsupported shape type '{shape_type}'. Expected one of: {sorted(_VALID_SHAPE_TYPES)}."
            )

    def _coerce_shape_section(self, section):
        """Normalize section input to a typed shape-section dataclass.

        Parameters
        ----------
        section : ShapeSection or dict or None
            Section description used for interior-site assignment.

        Returns
        -------
        shape_section : ShapeSection
            Normalized section dataclass.

        Raises
        ------
        TypeError
            Raised when ``section`` is neither ``None``, a
            :class:`porems.shape.ShapeSection`, nor a mapping.
        """
        return ShapeSection.from_dict(section)

    def _validate_attachment_request(self, site_type, inp):
        """Validate attachment input modes.

        Parameters
        ----------
        site_type : str
            Requested surface identifier.
        inp : str
            Attachment amount input mode.

        Raises
        ------
        ValueError
            Raised when either value is not supported.
        """
        if site_type not in _VALID_SITE_TYPES:
            raise ValueError(
                f"Unsupported site_type '{site_type}'. Expected one of: {sorted(_VALID_SITE_TYPES)}."
            )
        if inp not in _VALID_ATTACH_INPUTS:
            raise ValueError(
                f"Unsupported inp '{inp}'. Expected one of: {sorted(_VALID_ATTACH_INPUTS)}."
            )

    def _validate_position_count(self, pos_list, expected_amount):
        """Validate the number of explicit attachment positions.

        Parameters
        ----------
        pos_list : list
            Explicit surface positions.
        expected_amount : int
            Number of molecules that will be attached.

        Raises
        ------
        ValueError
            Raised when the number of positions does not match the expected
            number of attachments.
        """
        if pos_list and len(pos_list) != expected_amount:
            raise ValueError(
                "Number of given positions does not match number of groups to attach."
            )

    def _validate_special_symmetry(self, symmetry):
        """Validate the symmetry mode for special attachment.

        Parameters
        ----------
        symmetry : str
            Requested symmetry mode.

        Raises
        ------
        ValueError
            Raised when the symmetry mode is not supported.
        """
        if symmetry not in _VALID_SPECIAL_SYMMETRIES:
            raise ValueError(
                f"Unsupported symmetry '{symmetry}'. Expected one of: {sorted(_VALID_SPECIAL_SYMMETRIES)}."
            )

    def _shape_analyses(self):
        """Build cached analysis objects for the currently configured shapes.

        Returns
        -------
        analyses : list
            Shape analysis objects in shape order.
        """
        return [_ShapeAnalysis.from_shape(shape_id, shape) for shape_id, shape in enumerate(self._shapes)]

    def _resolve_site_position(self, site_or_position):
        """Resolve a site id or stored position to Cartesian coordinates.

        Parameters
        ----------
        site_or_position : int or list
            Either a pore site identifier or an explicit position.

        Returns
        -------
        pos : list
            Cartesian position.
        """
        if isinstance(site_or_position, (int, np.integer)):
            return self._pore.get_block().pos(site_or_position)
        return site_or_position

    def _classify_interior_sites(self, site_ids, allow_multiple_matches):
        """Assign interior site ids to shape buckets.

        Parameters
        ----------
        site_ids : list
            Interior silicon site identifiers.
        allow_multiple_matches : bool
            True to allow one site to belong to more than one shape bucket.

        Returns
        -------
        shape_sites : dict
            Mapping of shape ids to matching site ids, with key ``20`` used for
            unassigned sites when needed.
        """
        analyses = self._shape_analyses()
        shape_sites = {analysis.shape_id: [] for analysis in analyses}
        shape_sites[20] = []

        for site_id in site_ids:
            pos = self._pore.get_block().pos(site_id)
            matched = False
            for analysis in analyses:
                if analysis.matches_site(pos):
                    shape_sites[analysis.shape_id].append(site_id)
                    matched = True
                    if not allow_multiple_matches:
                        break
            if not matched:
                shape_sites[20].append(site_id)

        if not shape_sites[20]:
            del shape_sites[20]

        return shape_sites

    def _warn_unassigned_interior_sites(self):
        """Warn when interior sites cannot be assigned to a unique shape."""
        warnings.warn(
            "Some interior silicon binding sites could not be assigned to a specific shape. "
            "They will remain in the unassigned bucket and be filled with siloxane and silanol bridges.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _analysis_site_groups(self):
        """Return the site collection that should drive shape analysis.

        Returns
        -------
        site_groups : dict or list
            Shape-indexed site groups used for diameter and roughness.
        """
        return self.sites_shape if self.sites_shape else self._si_pos_in

    def _collect_shape_radii(self):
        """Collect per-shape radii for diameter and roughness analysis.

        Returns
        -------
        radii : list
            Shape-indexed radius lists.
        """
        radii = []
        site_groups = self._analysis_site_groups()
        analyses = self._shape_analyses()

        for analysis in analyses:
            entries = site_groups.get(analysis.shape_id, []) if isinstance(site_groups, dict) else site_groups[analysis.shape_id]
            radii_temp = []
            for entry in entries:
                radius = analysis.radius_from_position(self._resolve_site_position(entry))
                if radius is not None:
                    radii_temp.append(radius)
            radii.append(radii_temp)

        return radii

    def _prune_consumed_shape_sites(self):
        """Remove interior site ids that were consumed during attachment."""
        for shape_key in self._pore.sites_sl_shape:
            self._pore.sites_sl_shape[shape_key] = [
                site_id for site_id in self._pore.sites_sl_shape[shape_key]
                if self._pore._sites[site_id]["state"]
            ]

    def _record_attached_molecules(self, shape_key, mols):
        """Update per-shape molecule counts after an attachment step.

        Parameters
        ----------
        shape_key : int
            Shape identifier in the site-allocation dictionaries.
        mols : list
            Attached molecules returned by :meth:`porems.pore.Pore.attach`.
        """
        for attached_mol in mols:
            short_name = attached_mol.get_short()
            if short_name not in ["SL", "SLG"]:
                counts = self._pore.sites_attach_mol[shape_key]
                counts[short_name] = counts.get(short_name, 0) + 1

    def prepare(self):
        """Drill the configured shapes and prepare the resulting surface.

        This step removes atoms inside the configured shapes, prepares the
        exposed silica surface, assigns surface normals and section ownership,
        optionally creates siloxane bridges to hit hydroxylation targets, and
        initializes the per-shape tracking used by later attachment and
        analysis routines.
        """
        # Carve out shape
        del_list = []
        for shape_spec in self._shapes:
            del_list += [
                atom_id
                for atom_id, atom in enumerate(self._block.get_atom_list())
                if shape_spec.shape.is_in(atom.get_pos())
            ]
        self._matrix.strip(del_list)

        # Prepare pore surface
        self._pore.prepare()

        # Determine sites
        self._pore.sites()
        site_list = self._pore.get_sites()

        # Check if sections are given
        self._sections = [self._shape_ranges(shape_spec) for shape_spec in self._shapes]
        self._is_section = False
        for section in self._sections:
            if not section=={0: [0, self._box[0]], 1: [0, self._box[1]], 2: [0, self._box[2]]}:
                self._is_section = True

        # Define sites and save binding site si positions and allocate to interior section
        self._site_ex = [site_key for site_key, site_val in site_list.items() if site_val["type"]=="ex"]
        self._si_pos_ex = [self._block.pos(site_key) for site_key in self._site_ex]

        self._site_in = [site_key for site_key, site_val in site_list.items() if site_val["type"]=="in"]
        self._si_pos_in = [[] for x in self._sections]

        # Add normal vector to interior pore site list
        for site in self._site_in:
            # Get geometric position
            pos = self._block.pos(site)

            # Multiple shapes
            if len(self._shapes)>1:
                # Check distance to shape centroid
                lengths = []
                for k in range(len(self._shapes)):
                    shape_config = self._shape_config(self._shapes[k])
                    if shape_config.central == (0, 0, 1):
                        if isinstance(shape_config, ConeConfig):
                            dia = (shape_config.diameter_1 + shape_config.diameter_2) / 2
                            lengths.append(
                                pms.geom.length(
                                    pms.geom.vector(shape_config.centroid[:2], pos[:2])
                                ) / (0.5 * dia)
                            )
                        else:
                            lengths.append(
                                pms.geom.length(
                                    pms.geom.vector(shape_config.centroid[:2], pos[:2])
                                ) / (0.5 * shape_config.diameter)
                            )
                    if shape_config.central == (0, 1, 0):
                        centroid = shape_config.centroid
                        lengths.append(
                            pms.geom.length(
                                pms.geom.vector([centroid[0], centroid[2]], [pos[0], pos[2]])
                            ) / (0.5 * shape_config.diameter)
                        )
                    if shape_config.central == (1, 0, 0):
                        centroid = shape_config.centroid
                        lengths.append(
                            pms.geom.length(
                                pms.geom.vector([centroid[1], centroid[2]], [pos[0], pos[2]])
                            ) / (0.5 * shape_config.diameter)
                        )
                    if shape_config.central == (1, 1, 0):
                        centroid = shape_config.centroid
                        lengths.append(
                            pms.geom.length(
                                pms.geom.vector([centroid[-1]], [pos[-1]])
                            ) / (0.5 * shape_config.diameter)
                        )
                
                # Fin minimal length id                            
                min_len_id = lengths.index(min(lengths))
  
                # If sections are given
                if self._is_section:
                    for _k in range(len(self._shapes)):
                        # Check if position within section
                        for sec_idx, section in enumerate(self._sections):
                            section_config = self._shape_config(self._shapes[sec_idx])
                            if section_config.central == (0, 0, 1):
                                is_in = []
                                for dim in range(3):
                                    if dim == 2:
                                        is_in.append(self._shapes[sec_idx].shape.is_in(pos))
                                    else:
                                        if isinstance(section_config, ConeConfig):
                                            dia = (section_config.diameter_1 + section_config.diameter_2) / 2
                                            is_in.append(
                                                pos[dim] >= section_config.centroid[dim] - dia / 2
                                                and pos[dim] <= section_config.centroid[dim] + dia / 2
                                            )
                                        else:
                                            is_in.append(
                                                pos[dim] >= section_config.centroid[dim] - section_config.diameter / 2
                                                and pos[dim] <= section_config.centroid[dim] + section_config.diameter / 2
                                            )
                                if sum(is_in)==3:
                                    min_len_id = sec_idx
            else:
                min_len_id = 0

            # Add position to global list
            self._si_pos_in[min_len_id].append(pos)

            # Add normal vector to pore site list
            site_list[site]["normal"] = self._shapes[min_len_id].shape.normal

        # Add normal vector to exterior pore site list
        for site in self._site_ex:
            site_list[site]["normal"] = self._normal_ex

        # Sanity check
        num_site_err = sum([1 for site in site_list if "normal" not in site_list[site].keys()])
        if num_site_err > 0:
            warnings.warn(
                f"{num_site_err} sites were not assigned to shapes. Consider adjusting section intervals.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Siloxane bridges
        if self._has_interior_hydroxylation_target():
            self._siloxane("in")
        if self._hydro["ex"]:
            self._siloxane("ex")

        # Update site list after siloxan bridges
        self._site_ex = [site_key for site_key, site_val in site_list.items() if site_val["type"]=="ex"]
        self._site_in = [site_key for site_key, site_val in site_list.items() if site_val["type"]=="in"]

        # Calculate free binding sites of the several pores
        # Initialize dictonaries
        self._init_site_tracking()
        self.sites_sl_shape = self._classify_interior_sites(self._site_in, allow_multiple_matches=True)

        # If every site match to one shape drop dictonary "20"
        if 20 in self.sites_sl_shape:
            self._pore.sites_attach_mol[20] = {}
        else:
            self.sites_shape.pop(20, None)
            self._pore.sites_attach_mol.pop(20, None)

        # Preserve the historical analysis path for systems without interior
        # siloxane adjustment. In that case diameter/roughness should continue
        # to use the shape-assigned silicon positions in ``_si_pos_in``.
        if self._has_interior_hydroxylation_target():
            self.sites_shape = {
                shape_id: list(site_ids)
                for shape_id, site_ids in self.sites_sl_shape.items()
            }
        else:
            self.sites_shape = {}

        # Count the numbers of attached siloxane
        self._pore.sites_sl_shape = self.sites_sl_shape
        for i in self.sites_sl_shape:
            self._pore.sites_attach_mol[i]["SL"] = 0
            self._pore.sites_attach_mol[i]["SLG"] = 0
            for si in self.sites_sl_shape[i]:
                if (len(self._pore._sites[si]["o"])==1 and self._pore._sites[si]["type"]=="in"):
                    self._pore.sites_attach_mol[i]["SL"] +=1
                elif (len(self._pore._sites[si]["o"])==2 and self._pore._sites[si]["type"]=="in"):
                    self._pore.sites_attach_mol[i]["SLG"] +=1
        
        # Objectify grid
        non_grid = self._matrix.bound(1)+list(site_list.keys())
        bonded = self._matrix.bound(0, "gt")
        grid_atoms = [atom for atom in bonded if not atom in non_grid]
        self._pore.objectify(grid_atoms)

    ##############
    # Attachment #
    ##############
    def _normal_ex(self, pos):
        """Return the outward normal of the exterior reservoir surface.

        Parameters
        ----------
        pos : list
            Surface position.

        Returns
        -------
        normal : list
            Unit-like normal vector pointing away from the silica block.
        """
        return [0, 0, -1] if pos[2] < self.centroid()[2] else [0, 0, 1]

    def _attach_special(self, mol, mount, axis, amount, scale=1, symmetry="point", is_proxi=True, is_rotate=False):
        """Attach molecules to evenly spaced special positions on one pore.

        Parameters
        ----------
        mol : Molecule
            Molecule object to attach.
        mount : int
            Atom id of the molecule that is placed on the surface silicon atom.
        axis : list
            List of two atom ids defining the molecule axis.
        amount : int
            Number of molecules to attach.
        scale : float, optional
            Circumference scaling around the molecule position.
        symmetry : str, optional
            Symmetry option, either ``"point"`` or ``"mirror"``.
        is_proxi : bool, optional
            True to fill binding sites in proximity of filled binding sites.
        is_rotate : bool, optional
            True to randomly rotate the molecule around its own axis.

        Raises
        ------
        ValueError
            Raised when the requested symmetry mode is not supported.
        """
        self._validate_special_symmetry(symmetry)

        dist = self._box[2] / amount if amount > 0 else 0
        start = dist / 2
        diameters = self.diameter()
        diameter = diameters[0] if isinstance(diameters, list) else diameters

        pos_list = []
        for i in range(amount):
            coeff = -1 if symmetry == "point" and i % 2 == 0 else 1
            x = self._centroid[0] + coeff * diameter / 2
            y = self._centroid[1]
            z = start + dist * i
            pos_list.append([x, y, z])

        mols = self._pore.attach(
            mol,
            mount,
            axis,
            self._site_in,
            len(pos_list),
            scale,
            pos_list=pos_list,
            is_proxi=is_proxi,
            is_random=False,
            is_rotate=is_rotate,
        )

        for attached_mol in mols:
            if attached_mol.get_short() not in self._sort_list:
                self._sort_list.append(attached_mol.get_short())

    def _siloxane(self, site_type, slx_dist=None):
        """Adjust hydroxylation by introducing siloxane bridges.

        Parameters
        ----------
        site_type : str
            Site family, either interior ``"in"`` or exterior ``"ex"``.
        slx_dist : list, optional
            Accepted silicon-silicon distance interval for bridge formation.
        """
        slx_dist = [0.507-1e-2, 0.507+1e-2] if slx_dist is None else slx_dist

        # Initialize
        site_list = self._pore.get_sites()
        sites = self._site_in if site_type=="in" else self._site_ex
        hydro = self._hydro["in"] if site_type=="in" else self._hydro["ex"]

        # Find free binding sites of Si atoms in the structure
        if site_type=="in":
            self._init_site_tracking()
            self._pore.sites_attach_mol[20] = {}
            self.sites_shape = self._classify_interior_sites(sites, allow_multiple_matches=False)
            if 20 not in self.sites_shape:
                self._pore.sites_attach_mol.pop(20, None)
            else:
                self._warn_unassigned_interior_sites()

            # Amount - Connect two oxygen to one siloxane
            amount = []
            interior_shape_items = [
                (shape_id, shape_sites)
                for shape_id, shape_sites in self.sites_shape.items()
                if shape_id != 20
            ]
            for hydro_value, surface, (_, shape_sites) in zip(
                hydro,
                self.surface(is_sum=False).for_site_type(site_type),
                interior_shape_items,
            ):
                oh = len(sum([site_list[site]["o"] for site in shape_sites], []))
                oh_goal = pms.utils.mumol_m2_to_mols(hydro_value, surface)
                amount.append(round((oh - oh_goal) / 2))

            # Fill siloxane
            for amount_value, (shape_id, shape_sites) in zip(amount, interior_shape_items):
                if amount_value > 0:
                    # Run attachment
                    mols = self._pore.siloxane(shape_sites, amount_value, slx_dist=slx_dist, site_type=site_type)
                    for mol in mols:
                        # Count the numbers of attached siloxane
                        if mol.get_short() in self._pore.sites_attach_mol[shape_id]:
                            self._pore.sites_attach_mol[shape_id][mol.get_short()] +=1
                        else:
                            self._pore.sites_attach_mol[shape_id][mol.get_short()] = 0
                            self._pore.sites_attach_mol[shape_id][mol.get_short()] += 1

                    # Add to sorting list
                    for mol in mols:
                        if not mol.get_short() in self._sort_list:
                            self._sort_list.append(mol.get_short())
        else:
             # Amount - Connect two oxygen to one siloxane
            oh = len(sum([site_list[site]["o"] for site in sites], []))
            oh_goal = pms.utils.mumol_m2_to_mols(hydro, self.surface().for_site_type(site_type))
            amount = round((oh-oh_goal)/2)

            # Fill siloxane
            if amount > 0:
                # Run attachment
                mols = self._pore.siloxane(sites, amount, slx_dist=slx_dist, site_type=site_type)
                
                # Add to sorting list
                for mol in mols:
                    if not mol.get_short() in self._sort_list:
                        self._sort_list.append(mol.get_short())
                        

    def attach(self, mol, mount, axis, amount, site_type="in", inp="num", shape="all", pos_list=None, scale=1, trials=1000, is_proxi=True, is_rotate=False, is_g=True):
        """Attach a molecule to interior or exterior pore sites.

        Parameters
        ----------
        mol : Molecule
            Molecule to attach.
        mount : int
            Atom id on ``mol`` placed on the selected silicon site.
        axis : list
            Two atom ids defining the molecule orientation axis.
        amount : int
            Requested number or density of attachments, depending on ``inp``.
        site_type : str, optional
            Use ``"in"`` for interior sites and ``"ex"`` for exterior sites.
        inp : str, optional
            Amount interpretation: absolute number (``"num"``), hydroxylation
            density (``"molar"``), or fraction of OH groups (``"percent"``).
        shape : str, optional
            Shape selector for interior attachment. Use ``"all"`` for all
            interior shapes or ``"shape_N"`` for one specific shape.
        pos_list : list, optional
            Explicit Cartesian target positions used to pick nearest free sites.
        scale : float, optional
            Effective lateral spacing multiplier for proximity searches.
        trials : int, optional
            Number of random site-selection attempts.
        is_proxi : bool, optional
            True to consume nearby sites with silanol fills after attachment.
        is_rotate : bool, optional
            True to allow random rotation around the molecule axis.
        is_g : bool, optional
            True to allow geminal surface sites as mounting positions.

        Raises
        ------
        ValueError
            Raised when ``site_type`` or ``inp`` is unsupported, or when the
            number of explicit positions does not match the number of
            attachments.
        """
        self._validate_attachment_request(site_type, inp)
        pos_list = [] if pos_list is None else pos_list

        saved_unassigned_sites = None
        had_unassigned_sites = shape == "all" and 20 in self._pore.sites_sl_shape
        if had_unassigned_sites:
            saved_unassigned_sites = self._pore.sites_sl_shape.pop(20)

        try:
            amount_list = {}

            # Amount of SL molecules
            if inp == "molar":
                if site_type == "in" and shape != "all":
                    amount = int(
                        pms.utils.mumol_m2_to_mols(
                            amount,
                            self.surface(is_sum=False).for_site_type(site_type)[int(shape[-1])],
                        )
                    )
                elif site_type == "in" and shape == "all":
                    for sites_shape_idx in self._pore.sites_sl_shape:
                        amount_list[sites_shape_idx] = int(
                            pms.utils.mumol_m2_to_mols(
                                amount,
                                self.surface(is_sum=False).for_site_type(site_type)[sites_shape_idx],
                            )
                        )
                elif site_type == "ex":
                    amount = int(pms.utils.mumol_m2_to_mols(amount, self.surface().for_site_type(site_type)))

            elif inp == "percent":
                if site_type == "in" and shape != "all":
                    sites = self._pore.sites_sl_shape[int(shape[-1])]
                    num_oh = len(sites)
                    num_oh += sum(
                        1 for site_props in self._pore.get_sites().values()
                        if len(site_props["o"]) == 2 and site_props["type"] == site_type
                    )
                    amount = int(amount / 100 * num_oh)
                elif site_type == "in" and shape == "all":
                    for sites_shape_idx in self._pore.sites_sl_shape:
                        sites = self._pore.sites_sl_shape[sites_shape_idx]
                        num_oh = len(sites)
                        num_oh += sum(
                            1 for site_props in self._pore.get_sites().values()
                            if len(site_props["o"]) == 2 and site_props["type"] == site_type
                        )
                        amount_list[sites_shape_idx] = int(amount / 100 * num_oh)
                elif site_type == "ex":
                    sites = self._site_ex
                    num_oh = len(sites)
                    num_oh += sum(
                        1 for site_props in self._pore.get_sites().values()
                        if len(site_props["o"]) == 2 and site_props["type"] == site_type
                    )
                    amount = int(amount / 100 * num_oh)

            else:
                if site_type == "in" and shape == "all":
                    for sites_shape_idx in self._pore.sites_sl_shape:
                        amount_list[sites_shape_idx] = int(amount / len(self._pore.sites_sl_shape))

            expected_amount = sum(amount_list.values()) if site_type == "in" and shape == "all" else amount
            self._validate_position_count(pos_list, expected_amount)

            attached_mols = []

            if site_type == "ex":
                sites = self._site_ex
                attached_mols = self._pore.attach(
                    mol,
                    mount,
                    axis,
                    sites,
                    amount,
                    scale,
                    trials,
                    pos_list=pos_list,
                    site_type=site_type,
                    is_proxi=is_proxi,
                    is_random=True,
                    is_rotate=is_rotate,
                    is_g=is_g,
                )
            elif site_type == "in" and shape != "all":
                shape_idx = int(shape[-1])
                sites = self._pore.sites_sl_shape[shape_idx]
                attached_mols = self._pore.attach(
                    mol,
                    mount,
                    axis,
                    sites,
                    amount,
                    scale,
                    trials,
                    pos_list=pos_list,
                    site_type=site_type,
                    is_proxi=is_proxi,
                    is_random=True,
                    is_rotate=is_rotate,
                    is_g=is_g,
                )
                self._prune_consumed_shape_sites()
                self._record_attached_molecules(shape_idx, attached_mols)
            elif shape == "all" and site_type == "in":
                pos_offset = 0
                for sites_shape_idx in self._pore.sites_sl_shape:
                    sites = self._pore.sites_sl_shape[sites_shape_idx]
                    shape_amount = amount_list[sites_shape_idx]
                    shape_pos_list = pos_list[pos_offset:pos_offset + shape_amount] if pos_list else []
                    pos_offset += shape_amount
                    mols = self._pore.attach(
                        mol,
                        mount,
                        axis,
                        sites,
                        shape_amount,
                        scale,
                        trials,
                        pos_list=shape_pos_list,
                        site_type=site_type,
                        is_proxi=is_proxi,
                        is_random=True,
                        is_rotate=is_rotate,
                        is_g=is_g,
                    )
                    attached_mols.extend(mols)
                    self._prune_consumed_shape_sites()
                    self._record_attached_molecules(sites_shape_idx, mols)

            for attached_mol in attached_mols:
                if attached_mol.get_short() not in self._sort_list:
                    self._sort_list.append(attached_mol.get_short())
        finally:
            if had_unassigned_sites:
                self._pore.sites_sl_shape[20] = saved_unassigned_sites

    ################
    # Finalization #
    ################
    def finalize(self):
        """Finalize the pore by saturating remaining sites and boxing it.

        Remaining free interior and exterior sites are filled with silanol
        groups. If a reservoir length was requested, the pore is translated and
        re-boxed accordingly; otherwise the original block box is preserved.
        """

        # Fill silanol on the exterior surface
        mols_ex = self._pore.fill_sites(self._site_ex, "ex") if self._site_ex else []
        for mol in mols_ex:
            if not mol.get_short() in self._sort_list:
                self._sort_list.append(mol.get_short())

        # Fill silanol on the interior surface
        for i,sites in self._pore.sites_sl_shape.items():  
            mols_in = self._pore.fill_sites(sites, "in") if self._site_in else []
            # for mol in mols_in:
            #     if mol.get_short() in self._pore.sites_attach_mol[i]:
            #         self._pore.sites_attach_mol[i][mol.get_short()] +=1
            #     else:
            #         self._pore.sites_attach_mol[i][mol.get_short()] = 0
            #         self._pore.sites_attach_mol[i][mol.get_short()] += 1
            #         print(self._pore.sites_attach_mol[i][mol.get_short()])
            for mol in mols_in:
                if not mol.get_short() in self._sort_list:
                    self._sort_list.append(mol.get_short())

        # Create reservoir
        if self._res > 0:
            self._pore.reservoir(self._res)
        else:
            self._pore.set_box(self._box)

    def store(self, link="./", sort_list=None):
        """Write the finalized pore system and companion simulation files.

        Parameters
        ----------
        link : str, optional
            Output directory.
        sort_list : list, optional
            Sorting list for output structure files. Defaults to the internal
            builder ordering.
        """
        sort_list = [] if sort_list is None else sort_list

        # Process input
        link = link if link[-1] == "/" else link+"/"

        # Set sort list
        sort_list = sort_list if sort_list else self._sort_list

        # Create store object
        store = pms.Store(self._pore, link, sort_list)

        # Save files
        store.gro(use_atom_names=True)
        store.obj()
        store.top()
        store.grid()
        pms.utils.save(self, link+self._pore.get_name()+"_system.obj")
        self.yml(link)

    def yml(self, link="./"):
        """Write a YAML summary of the pore geometry and chemistry.

        Parameters
        ----------
        link : str, optional
            Output directory.
        """
        # Process input
        link = link if link[-1] == "/" else link+"/"

        # Fill system properties
        self._yml["system"] = {}
        self._yml["system"]["dimensions"] = self.box()
        self._yml["system"]["centroid"] = self.centroid()
        self._yml["system"]["reservoir"] = self.reservoir()
        self._yml["system"]["volume"] = self.volume()
        self._yml["system"]["surface"] = self.surface().to_dict()

        # Calculate properties
        diameter = self.diameter()
        roughness = self.roughness()
        volume = self.volume(is_sum=False)
        surface = self.surface(is_sum=False)

        # Fill properties for each shape
        for i, shape_spec in enumerate(self._shapes):
            shape_id = "shape_"+"%02i"%i
            self._yml[shape_id] = {}
            self._yml[shape_id]["shape"] = shape_spec.shape_type
            self._yml[shape_id]["parameter"] = shape_spec.shape.get_inp().copy()
            if shape_spec.shape_type=="CYLINDER":
                self._yml[shape_id]["parameter"]["diameter"] += 0.5
            elif shape_spec.shape_type=="SLIT":
                self._yml[shape_id]["parameter"]["height"] += 0.5
            self._yml[shape_id]["diameter"] = diameter[i]
            self._yml[shape_id]["roughness"] = roughness.interior[i]
            self._yml[shape_id]["volume"] = volume[i]
            self._yml[shape_id]["surface"] = surface.interior[i]

        # Export
        yaml.Dumper.ignore_aliases = lambda *args : True
        with open(link+self._pore.get_name()+".yml", "w") as file_out:
            file_out.write(yaml.dump(self._yml, default_flow_style=False))

    ############
    # Analysis #
    ############
    def diameter(self):
        """Return the effective pore diameter after preparation.

        The diameter is estimated from the mean radial distance of the tracked
        interior silicon positions relative to each shape axis or centroid.

        .. math::

            \\bar r=\\frac1n\\sum_{i=1}^nr_i

        with the number of silicon atoms :math:`n`. The diameter is then

        .. math::

            d=2\\bar r=\\frac2n\\sum_{i=1}^nr_i.

        Returns
        -------
        diameter : list
            Effective diameters for each configured shape.
        """
        radii = self._collect_shape_radii()
        # Calculate mean
        r_bar = [sum(r)/len(r) if len(r)>0 else 0 for r in radii]

        # Return diameter
        diam = [2*r for r in r_bar]
        return diam

    def roughness(self):
        """Return the surface roughness of the pore surfaces.

        Roughness is calculated as the root-mean-square deviation of the
        tracked silicon-site radii from the mean radius of each shape. When
        reservoirs are present, an exterior roughness value is reported as
        well.

        .. math::

            \\bar r=\\frac1n\\sum_{i=1}^nr_i

        with the number of silicon atoms :math:`n`. This mean value is used in
        the square root roughness calculation

        .. math::

            R_q = \\sqrt{\\frac1n\\sum_{i=1}^n\\|r_i-\\bar r\\|^2}.

        Returns
        -------
        roughness : RoughnessProfile
            Roughness values for the interior shapes and exterior surface.
        """
        radii_in = self._collect_shape_radii()

        # Exterior
        r_ex = []
        if self._res:
            ## Create molecules with exterior positions
            temp_mol = pms.Molecule()
            for pos in self._si_pos_ex:
                temp_mol.add("Si", pos)
            temp_mol.zero()
            size = temp_mol.get_box()[2]

            ## Calculate distance to boundary
            r_ex = [pos[2] if pos[2] < size/2 else abs(pos[2]-size) for pos in self._si_pos_ex]

        # Calculate mean
        r_bar_in = [sum(r_in)/len(r_in) if len(r_in)>0 else 0 for r_in in radii_in]
        r_bar_ex = sum(r_ex)/len(r_ex) if len(r_ex)>0 else 0

        # Calculate roughness
        r_q_in =  [math.sqrt(sum([(r_i-r_bar_in[i])**2 for r_i in r_in])/len(r_in)) if len(r_in)>0 else 0 for i, r_in in enumerate(radii_in)]
        r_q_ex =  math.sqrt(sum([(r_i-r_bar_ex)**2 for r_i in r_ex])/len(r_ex)) if len(r_ex)>0 else 0

        # Calculate square root roughness
        return RoughnessProfile(interior=r_q_in, exterior=r_q_ex)

    def volume(self, is_sum=True):
        """Return the pore volume derived from the prepared geometry.

        Notes
        -----
        The calculation rebuilds idealized analysis shapes from the prepared
        effective diameters.

        Parameters
        ----------
        is_sum : bool, optional
            True to return sum of all sections

        Returns
        -------
        volume : float, list
            Total pore volume when ``is_sum`` is True, otherwise one value per
            configured shape.
        """
        # Get diameters
        diam = self.diameter()
        diam = diam if isinstance(diam, list) else [diam]

        # Calculate volumes
        volume = []
        for i, shape_spec in enumerate(self._shapes):
            shape_config = self._shape_config(shape_spec)
            centroid = shape_config.centroid
            if shape_spec.shape_type != "SPHERE":
                length = shape_config.length
            if shape_spec.shape_type == "CYLINDER":
                volume.append(
                    pms.Cylinder(
                        CylinderConfig(
                            centroid=centroid,
                            central=(0, 0, 1),
                            length=length,
                            diameter=diam[i],
                        )
                    ).volume()
                )
            elif shape_spec.shape_type == "SLIT":
                volume.append(
                    pms.Cuboid(
                        CuboidConfig(
                            centroid=centroid,
                            central=(0, 0, 1),
                            length=length,
                            width=self._box[0],
                            height=diam[i],
                        )
                    ).volume()
                )
            elif shape_spec.shape_type == "SPHERE":
                volume.append(
                    pms.Sphere(
                        SphereConfig(
                            centroid=centroid,
                            central=(0, 0, 1),
                            diameter=diam[i],
                        )
                    ).volume()
                )
            if shape_spec.shape_type == "CONE":
                volume.append(
                    pms.Cone(
                        ConeConfig(
                            centroid=centroid,
                            central=(0, 0, 1),
                            length=length,
                            diameter_1=shape_config.diameter_1,
                            diameter_2=shape_config.diameter_2,
                        )
                    ).volume()
                )

        return sum(volume) if is_sum else volume

    def surface(self, is_sum=True):
        """Return interior and exterior surface areas of the pore system.

        Notes
        -----
        The calculation rebuilds idealized analysis shapes from the prepared
        effective diameters. Exterior surfaces are inferred from the end
        sections of the configured shapes.

        Parameters
        ----------
        is_sum : bool, optional
            True to return sum of all sections

        Returns
        -------
        surface : SurfaceAreaSummary
            Interior and exterior surface areas.
        """
        # Get diameters
        diam = self.diameter()
        diam = diam if isinstance(diam, list) else [diam]

        # Interior surface
        surf_in = []
        for i, shape_spec in enumerate(self._shapes):
            shape_config = self._shape_config(shape_spec)
            centroid = shape_config.centroid
            if shape_spec.shape_type != "SPHERE":
                length = shape_config.length
            if shape_spec.shape_type == "CYLINDER":
                surf_in.append(
                    pms.Cylinder(
                        CylinderConfig(
                            centroid=centroid,
                            central=(0, 0, 1),
                            length=length,
                            diameter=diam[i],
                        )
                    ).surface()
                )
            elif shape_spec.shape_type == "SLIT":
                surf_in.append(
                    pms.Cuboid(
                        CuboidConfig(
                            centroid=centroid,
                            central=(0, 0, 1),
                            length=length,
                            width=self._box[0],
                            height=diam[i],
                        )
                    ).surface() / 2
                )
            elif shape_spec.shape_type == "SPHERE":
                surf_in.append(
                    pms.Sphere(
                        SphereConfig(
                            centroid=centroid,
                            central=(0, 0, 1),
                            diameter=diam[i],
                        )
                    ).surface()
                )
            if shape_spec.shape_type == "CONE":
                surf_in.append(
                    pms.Cone(
                        ConeConfig(
                            centroid=centroid,
                            central=(0, 0, 1),
                            length=length,
                            diameter_1=shape_config.diameter_1,
                            diameter_2=shape_config.diameter_2,
                        )
                    ).surface()
                )

        # Exterior surface
        sections_ex = [0, 0]
        for i, section in enumerate(self._sections):
            if section[2][0]-0<=1e-2:
                sections_ex[0] = i
            if section[2][1]-self._box[2]<=1e-2:
                sections_ex[1] = i

        surf_ex = []
        for section in sections_ex:
            if self._shapes[section].shape_type=="CYLINDER":
                surf_ex.append(self._box[0]*self._box[1]-math.pi*(diam[section]/2)**2)
            elif self._shapes[section].shape_type=="SLIT":
                surf_ex.append(self._box[0]*(self._box[1]-diam[section]))
            elif self._shapes[section].shape_type=="SPHERE":
                surf_ex.append(self._box[0]*self._box[1]-math.pi*(diam[section]/2)**2)
            if self._shapes[section].shape_type=="CONE":
                surf_ex.append(0)

        return SurfaceAreaSummary(
            interior=sum(surf_in) if is_sum else surf_in,
            exterior=sum(surf_ex) if is_sum else surf_ex,
        )

    def allocation(self):
        """Return surface allocation statistics for all attached molecules.

        The reported values include absolute counts, densities per square
        nanometer, and densities converted to
        :math:`\\frac{\\mu\\text{mol}}{\\text{m}^2}`.

        Returns
        -------
        alloc : dict[str, AllocationSummary]
            Surface allocation statistics grouped by molecule short name.
        """
        # Get surfaces
        surf = self.surface()
        site_dict = self._pore.get_site_dict()

        # Calculate allocation for all molecules
        alloc = {}
        for mol in sorted(self._sort_list):
            for site_type in ["in", "ex"]:
                if mol in site_dict[site_type]:
                    if not mol in alloc:
                        alloc[mol] = AllocationSummary(
                            interior=SurfaceAllocationStats(0, 0.0, 0.0),
                            exterior=SurfaceAllocationStats(0, 0.0, 0.0),
                        )
                    # Number of molecules
                    count = len(site_dict[site_type][mol])
                    surface_value = surf.for_site_type(site_type)
                    stats = SurfaceAllocationStats(
                        count=count,
                        density_nm2=count/surface_value if surface_value > 0 else 0,
                        density_mumol_m2=pms.utils.mols_to_mumol_m2(count, surface_value) if surface_value > 0 else 0,
                    )
                    if site_type == "in":
                        alloc[mol] = AllocationSummary(interior=stats, exterior=alloc[mol].exterior)
                    else:
                        alloc[mol] = AllocationSummary(interior=alloc[mol].interior, exterior=stats)

        # OH allocation
        alloc["OH"] = AllocationSummary(
            interior=SurfaceAllocationStats(0, 0.0, 0.0),
            exterior=SurfaceAllocationStats(0, 0.0, 0.0),
        )
        for site_type in ["in", "ex"]:
            num_oh = len(sum([x["o"] for x in self._pore.get_sites().values() if x["type"]==site_type], []))
            for mol in site_dict[site_type].keys():
                num_oh -= len(site_dict[site_type][mol]) if mol not in ["SL", "SLG", "SLX"] else 0

            # num_oh = num_oh-num_in_ex if site_type=="ex" else num_oh+num_in_ex
            surface_value = surf.for_site_type(site_type)
            stats = SurfaceAllocationStats(
                count=num_oh,
                density_nm2=num_oh/surface_value if surface_value > 0 else 0,
                density_mumol_m2=pms.utils.mols_to_mumol_m2(num_oh, surface_value) if surface_value > 0 else 0,
            )
            if site_type == "in":
                alloc["OH"] = AllocationSummary(interior=stats, exterior=alloc["OH"].exterior)
            else:
                alloc["OH"] = AllocationSummary(interior=alloc["OH"].interior, exterior=stats)

        # Hydroxylation - Total number of binding sites
        alloc["Hydro"] = AllocationSummary(
            interior=SurfaceAllocationStats(0, 0.0, 0.0),
            exterior=SurfaceAllocationStats(0, 0.0, 0.0),
        )
        for site_type in ["in", "ex"]:
            num_tot = len(sum([x["o"] for x in self._pore.get_sites().values() if x["type"]==site_type], []))

            # num_tot = num_tot-num_in_ex if site_type=="ex" else num_tot+num_in_ex
            surface_value = surf.for_site_type(site_type)
            stats = SurfaceAllocationStats(
                count=num_tot,
                density_nm2=num_tot/surface_value if surface_value > 0 else 0,
                density_mumol_m2=pms.utils.mols_to_mumol_m2(num_tot, surface_value) if surface_value > 0 else 0,
            )
            if site_type == "in":
                alloc["Hydro"] = AllocationSummary(interior=stats, exterior=alloc["Hydro"].exterior)
            else:
                alloc["Hydro"] = AllocationSummary(interior=alloc["Hydro"].interior, exterior=stats)

        return alloc

    def reservoir(self):
        """Return the reservoir length.

        Returns
        -------
        res : float
            Reservoir length in nanometers.
        """
        return self._res

    def box(self):
        """Return the final simulation box dimensions.

        Returns
        -------
        box : list
            Box lengths in all dimensions.
        """
        return self._pore.get_box()

    def centroid(self):
        """Return the pore centroid.

        Returns
        -------
        centroid : list
            Geometric center used for the configured shapes.
        """
        return self._centroid

    def shape(self):
        """Return the configured shape definitions.

        Returns
        -------
        pore_shape : list
            :class:`porems.shape.ShapeSpec` entries used to drill and analyze
            the pore.
        """
        return self._shapes

    #########
    # Table #
    #########
    def table(self, decimals=3):
        """Return a tabular summary of the pore geometry and chemistry.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places used for formatted values.

        Returns
        -------
        tables : DataFrame
            Formatted table containing geometry, hydroxylation, and allocation
            summaries for interior and exterior surfaces.
        """
        # Initialize
        form = "%."+str(decimals)+"f"

        # Get allocation data
        allocation = self.allocation()

        # Get properties
        surf = self.surface()
        surface_in_by_shape = self.surface(is_sum=False).interior
        roughness = self.roughness()

        # Save data
        data = {"Interior": {}, "Exterior": {}}

        data["Interior"]["Silica block xyz-dimensions (nm)"] = " "
        data["Exterior"]["Silica block xyz-dimensions (nm)"] = "["+form%self.box()[0]+", "+form%self.box()[1]+", "+form%(self.box()[2]-2*self.reservoir())+"]"
        data["Interior"]["Simulation box xyz-dimensions (nm)"] = " "
        data["Exterior"]["Simulation box xyz-dimensions (nm)"] = "["+form%self.box()[0]+", "+form%self.box()[1]+", "+form%self.box()[2]+"]"
        data["Interior"]["Surface roughness (nm)"] = [form%val for val in roughness.interior]
        data["Exterior"]["Surface roughness (nm)"] = form%roughness.exterior
        for i, val in enumerate(self.diameter()):
            data["Interior"]["Pore "+ str(i+1) +" diameter (nm)"] = form%val
            data["Exterior"]["Pore "+ str(i+1) +" diameter (nm)"] = " "
        data["Interior"]["Solvent reservoir z-dimension (nm)"] = " "
        data["Exterior"]["Solvent reservoir z-dimension (nm)"] = form%self.reservoir()
        for i,val in enumerate(self.volume(is_sum=False)):
            data["Interior"]["Pore "+ str(i+1) +" volume (nm^3)"] = form%val
            data["Exterior"]["Pore "+ str(i+1) +" volume (nm^3)"] = " "
        data["Interior"]["Pore volume (nm^3)"] = form%self.volume()
        data["Exterior"]["Pore volume (nm^3)"] = " "
        data["Interior"]["Solvent reservoir volume (nm^3)"] = " "
        data["Exterior"]["Solvent reservoir volume (nm^3)"] = "2 * "+form%(self.box()[0]*self.box()[1]*self.reservoir())
        for i,val in enumerate(self.surface(is_sum=False).interior):
            data["Interior"]["Surface "+ str(i+1) +" area (nm^2)"] = form%val
            data["Exterior"]["Surface "+ str(i+1) +" area (nm^2)"] = " "
        data["Interior"]["Surface area (nm^2)"] = form%surf.interior
        data["Exterior"]["Surface area (nm^2)"] = "2 * "+form%(surf.exterior/2)

        data["Interior"]["Surface chemistry - Before Functionalization"] = " "
        data["Exterior"]["Surface chemistry - Before Functionalization"] = " "
        data["Interior"]["    Number of single silanol groups"] = "%i"%sum([1 for x in self._pore.get_sites().values() if len(x["o"])==1 and x["type"]=="in"])
        data["Exterior"]["    Number of single silanol groups"] = "%i"%sum([1 for x in self._pore.get_sites().values() if len(x["o"])==1 and x["type"]=="ex"])
        data["Interior"]["    Number of geminal silanol groups"] = "%i"%sum([1 for x in self._pore.get_sites().values() if len(x["o"])==2 and x["type"]=="in"])
        data["Exterior"]["    Number of geminal silanol groups"] = "%i"%sum([1 for x in self._pore.get_sites().values() if len(x["o"])==2 and x["type"]=="ex"])
        data["Interior"]["    Number of siloxane bridges"] = "%i"%allocation["SLX"].interior.count if "SLX" in allocation else "0"
        data["Exterior"]["    Number of siloxane bridges"] = "%i"%allocation["SLX"].exterior.count if "SLX" in allocation else "0"
        data["Interior"]["    Total number of OH groups"] = "%i"%allocation["Hydro"].interior.count
        data["Exterior"]["    Total number of OH groups"] = "%i"%allocation["Hydro"].exterior.count
        data["Interior"]["    Overall hydroxylation (mumol/m^2)"] = form%allocation["Hydro"].interior.density_mumol_m2
        data["Exterior"]["    Overall hydroxylation (mumol/m^2)"] = form%allocation["Hydro"].exterior.density_mumol_m2

        sites_attach_mol = getattr(self._pore, "sites_attach_mol", None)
        if sites_attach_mol is not None:
            for i in sites_attach_mol:
                if i != 20:
                    data["Interior"]["Surface chemistry - Before Functionalization (Pore " + str(i+1) +")"] = " "
                    data["Exterior"]["Surface chemistry - Before Functionalization (Pore " + str(i+1) +")"] = " "
                    data["Interior"]["    Pore " + str(i+1) + " Number of single silanol groups"] = "%i"%sites_attach_mol[i]["SL"]
                    data["Exterior"]["    Pore " + str(i+1) + " Number of single silanol groups"] = " "
                    data["Interior"]["    Pore " + str(i+1) + " Number of geminal silanol groups"] = "%i"%sites_attach_mol[i]["SLG"]
                    data["Exterior"]["    Pore " + str(i+1) + " Number of geminal silanol groups"] = " "
                    data["Interior"]["    Pore " + str(i+1) + " Number of siloxane bridges"] = "%i"%sites_attach_mol[i]["SLX"] if "SLX" in sites_attach_mol[i] else "0"
                    data["Exterior"]["    Pore " + str(i+1) + " Number of siloxane bridges"] = " "
                    data["Interior"]["    Pore " + str(i+1) + " Total number of OH groups"] = "%i"%(sites_attach_mol[i]["SL"]+2*sites_attach_mol[i]["SLG"])
                    data["Exterior"]["    Pore " + str(i+1) + " Total number of OH groups"] = " "
                    data["Interior"]["    Pore " + str(i+1) + " Overall hydroxylation (mumol/m^2)"] = form%(pms.utils.mols_to_mumol_m2(sites_attach_mol[i]["SL"]+2*sites_attach_mol[i]["SLG"],surface_in_by_shape[i]))
                    data["Exterior"]["    Pore " + str(i+1) + " Overall hydroxylation (mumol/m^2)"] = " "
                elif i == 20:
                    data["Interior"]["Surface chemistry - Before Functionalization (Unassigned binding sites)"] = " "
                    data["Exterior"]["Surface chemistry - Before Functionalization (Unassigned binding sites)"] = " "
                    data["Interior"]["    Unassigned binding sites" + " Number of single silanol groups"] = "%i"%sites_attach_mol[i]["SL"]
                    data["Exterior"]["    Unassigned binding sites" + " Number of single silanol groups"] = " "
                    data["Interior"]["    Unassigned binding sites" + " Number of geminal silanol groups"] = "%i"%sites_attach_mol[i]["SLG"]
                    data["Exterior"]["    Unassigned binding sites" + " Number of geminal silanol groups"] = " "
                    data["Interior"]["    Unassigned binding sites" + " Number of siloxane bridges"] = "%i"%sites_attach_mol[i]["SLX"] if "SLX" in sites_attach_mol[i] else "0"
                    data["Exterior"]["    Unassigned binding sites" + " Number of siloxane bridges"] = " "
                    data["Interior"]["    Unassigned binding sites" + " Total number of OH groups"] = "%i"%(sites_attach_mol[i]["SL"]+2*sites_attach_mol[i]["SLG"])
                    data["Exterior"]["    Unassigned binding sites" + " Total number of OH groups"] = " "
                    data["Interior"]["    Unassigned binding sites" + " Overall hydroxylation (mumol/m^2)"] = " "
                    data["Exterior"]["    Unassigned binding sites" + " Overall hydroxylation (mumol/m^2)"] = " "


        data["Interior"]["Surface chemistry - After Functionalization"] = " "
        data["Exterior"]["Surface chemistry - After Functionalization"] = " "
        for mol in allocation.keys():
            if mol not in ["SL", "SLG", "SLX", "Hydro", "OH"]:
                data["Interior"]["    Number of "+mol+" groups"] = "%i"%allocation[mol].interior.count
                data["Exterior"]["    Number of "+mol+" groups"] = "%i"%allocation[mol].exterior.count
                data["Interior"]["    "+mol+" density (mumol/m^2)"] = form%allocation[mol].interior.density_mumol_m2
                data["Exterior"]["    "+mol+" density (mumol/m^2)"] = form%allocation[mol].exterior.density_mumol_m2
        data["Interior"]["    Bonded-phase density (mumol/m^2)"] = form%(allocation["Hydro"].interior.density_mumol_m2-allocation["OH"].interior.density_mumol_m2)
        data["Exterior"]["    Bonded-phase density (mumol/m^2)"] = form%(allocation["Hydro"].exterior.density_mumol_m2-allocation["OH"].exterior.density_mumol_m2)
        data["Interior"]["    Number of residual OH groups"] = "%i"%allocation["OH"].interior.count
        data["Exterior"]["    Number of residual OH groups"] = "%i"%allocation["OH"].exterior.count
        data["Interior"]["    Residual hydroxylation (mumol/m^2)"] = form%allocation["OH"].interior.density_mumol_m2
        data["Exterior"]["    Residual hydroxylation (mumol/m^2)"] = form%allocation["OH"].exterior.density_mumol_m2

        if sites_attach_mol is not None:
            for i in sites_attach_mol:
                for mol in sites_attach_mol[i]:
                    if (mol not in ["SL", "SLG", "SLX", "Hydro", "OH"]): 
                        if i == 20:
                            data["Interior"]["Surface chemistry - After Functionalization (Unassigned binding sites)"] = " "
                            data["Exterior"]["Surface chemistry - After Functionalization (Unassigned binding sites)"] = " "
                            data["Interior"]["    Unassigned binding sites" + " Number of "+mol+" groups"] = "%i"%sites_attach_mol[i][mol]
                            data["Exterior"]["    Unassigned binding sites" + " Number of "+mol+" groups"] = " "
                            data["Interior"]["    Unassigned binding sites" + " "+mol+" density (mumol/m^2)"] = " "
                            data["Exterior"]["    Unassigned binding sites" + " "+mol+" density (mumol/m^2)"] = " "
                        else:
                            data["Interior"]["Surface chemistry - After Functionalization (Pore " + str(i+1) +")"] = " "
                            data["Exterior"]["Surface chemistry - After Functionalization (Pore " + str(i+1) +")"] = " "
                            data["Interior"]["    Pore " + str(i+1) + " Number of "+mol+" groups"] = "%i"%sites_attach_mol[i][mol]
                            data["Exterior"]["    Pore " + str(i+1) + " Number of "+mol+" groups"] = " "
                            data["Interior"]["    Pore " + str(i+1) + " "+mol+" density (mumol/m^2)"] = form%(pms.utils.mols_to_mumol_m2(sites_attach_mol[i][mol],surface_in_by_shape[i]))
                            data["Exterior"]["    Pore " + str(i+1) + " "+mol+" density (mumol/m^2)"] = " "
        return pd.DataFrame.from_dict(data)


class PoreCylinder(PoreKit):
    """Convenience builder for a cylindrical pore in :math:`\\beta`-cristobalite.

    Parameters
    ----------
    size : list
        Replication counts of the silica pattern.
    diam : float
        Cylinder diameter in nanometers.
    res : float, optional
        Reservoir size on each side in nanometers.
    hydro : list, optional
        Hydroxylation degree for interior and exterior of the pore in
        :math:`\\frac{\\mu\\text{mol}}{\\text{m}^2}`

    Examples
    --------
    Following example generates a cylindrical pore with a diameter of 4nm,
    reservoirs of 5nm on each side and a surface functionalized with TMS

    .. code-block:: python

        import porems as pms

        pore = pms.PoreCylinder([6, 6, 6], 4, 5)

        pore.attach(pms.gen.tms(), 0, [0, 1], 100, "in")
        pore.attach(pms.gen.tms(), 0, [0, 1], 20, "ex")

        pore.finalize()

        pore.store("output/")
    """
    def __init__(self, size, diam, res=5, hydro=None):
        # Call super class
        super(PoreCylinder, self).__init__()

        hydro = hydro if hydro is not None else [0, 0]

        # Create structure
        self.structure(pms.BetaCristobalit().generate(size, "z"))
        self.build()

        # Create reservoir
        self.exterior(res, hydro=hydro[1])

        # Add pore shape
        self.add_shape(self.shape_cylinder(diam), hydro=hydro[0])
        self.prepare()

    def attach_special(self, mol, mount, axis, amount, scale=1, symmetry="point", is_proxi=True, is_rotate=False):
        """Attach molecules at evenly spaced special positions on the cylinder.

        Parameters
        ----------
        mol : Molecule
            Molecule to attach.
        mount : int
            Atom id on ``mol`` placed on the selected silicon site.
        axis : list
            Two atom ids defining the molecule orientation axis.
        amount : int
            Number of molecules to attach.
        scale : float, optional
            Effective lateral spacing multiplier for proximity searches.
        symmetry : str, optional
            Symmetry option, either ``"point"`` or ``"mirror"``.
        is_proxi : bool, optional
            True to consume nearby sites with silanol fills after attachment.
        is_rotate : bool, optional
            True to allow random rotation around the molecule axis.
        
        Raises
        ------
        ValueError
            Raised when the requested symmetry mode is not supported.
        """
        self._attach_special(mol, mount, axis, amount, scale, symmetry, is_proxi, is_rotate)

class PoreSlit(PoreKit):
    """Convenience builder for a slit pore in :math:`\\beta`-cristobalite.

    Parameters
    ----------
    size : list
        Replication counts of the silica pattern.
    height : float
        Slit height in nanometers.
    res : float, optional
        Reservoir size on each side in nanometers.
    hydro: list, optional
        Hydroxylation degree for interior and exterior of the pore in
        :math:`\\frac{\\mu\\text{mol}}{\\text{m}^2}`

    Examples
    --------
    Following example generates a slit-pore with a height of 4nm functionalized
    with TMS

    .. code-block:: python

        import porems as pms

        pore = pms.PoreSlit([6, 6, 6], 4)

        pore.attach(pms.gen.tms(), 0, [0, 1], 100, "in")

        pore.finalize()

        pore.store("output/")
    """
    def __init__(self, size, height, res=0, hydro=None):
        # Call super class
        super(PoreSlit, self).__init__()

        hydro = hydro if hydro is not None else [0, 0]

        # Create structure
        self.structure(pms.BetaCristobalit().generate(size, "z"))
        self.build()

        # Create reservoir
        self.exterior(res, hydro=hydro[1])

        # Add pore shape
        self.add_shape(self.shape_slit(height), hydro=hydro[0])
        self.prepare()

    ##############
    # Attachment #
    ##############
    def attach_special(self, mol, mount, axis, amount, scale=1, symmetry="point", is_proxi=True, is_rotate=False):
        """Attach molecules at evenly spaced special positions on the slit.

        Parameters
        ----------
        mol : Molecule
            Molecule to attach.
        mount : int
            Atom id on ``mol`` placed on the selected silicon site.
        axis : list
            Two atom ids defining the molecule orientation axis.
        amount : int
            Number of molecules to attach.
        scale : float, optional
            Effective lateral spacing multiplier for proximity searches.
        symmetry : str, optional
            Symmetry option, either ``"point"`` or ``"mirror"``.
        is_proxi : bool, optional
            True to consume nearby sites with silanol fills after attachment.
        is_rotate : bool, optional
            True to allow random rotation around the molecule axis.
        
        Raises
        ------
        ValueError
            Raised when the requested symmetry mode is not supported.
        """
        self._attach_special(mol, mount, axis, amount, scale, symmetry, is_proxi, is_rotate)


class PoreCapsule(PoreKit):
    """Convenience builder for a capsule-shaped pore system.

    Parameters
    ----------
    size : list
        Replication counts of the silica pattern.
    diam : float
        Capsule diameter in nanometers.
    sep : float
        Separation distance between the spherical capsule sections.
    res : float, optional
        Reservoir size on each side in nanometers.
    hydro: list, optional
        Hydroxylation degree for interior and exterior of the pore in
        :math:`\\frac{\\mu\\text{mol}}{\\text{m}^2}`

    Examples
    --------
    Following example generates a capsule pore with a diameter of 4nm, a
    separation distance between the capsules of 2nm, reservoirs of 5nm on each
    side and a surface functionalized with TMS

    .. code-block:: python

        import porems as pms

        pore = pms.PoreCapsule([6, 6, 12], 4, 2)

        pore.attach(pms.gen.tms(), 0, [0, 1], 100, "in")
        pore.attach(pms.gen.tms(), 0, [0, 1], 20, "ex")

        pore.finalize()

        pore.store("output/")
    """
    def __init__(self, size, diam, sep, res=5, hydro=None):
        # Call super class
        super(PoreCapsule, self).__init__()

        hydro = hydro if hydro is not None else [0, 0]

        # Create structure
        self.structure(pms.BetaCristobalit().generate(size, "z"))
        self.build()

        # Create reservoir
        self.exterior(res, hydro=hydro[1])

        # Add pore shape
        center = [size[0]/2, size[1]/2]
        len_cyl = (size[2]-diam-sep)/2

        centroids = []
        centroids.append(center+[0+len_cyl/2])
        centroids.append(center+[len_cyl])
        centroids.append(center+[len_cyl+diam+sep])
        centroids.append(center+[size[2]-len_cyl/2])

        sections = []
        sections.append({"x": [], "y": [], "z": [0, len_cyl]})
        sections.append({"x": [], "y": [], "z": [len_cyl, len_cyl+diam/2+sep/2]})
        sections.append({"x": [], "y": [], "z": [len_cyl+diam/2+sep/2, len_cyl+diam+sep]})
        sections.append({"x": [], "y": [], "z": [size[2]-len_cyl, size[2]]})

        self.add_shape(self.shape_cylinder(diam, len_cyl, centroids[0]), section=sections[0], hydro=hydro[0])
        self.add_shape(self.shape_sphere(  diam,          centroids[1]), section=sections[1], hydro=hydro[0])
        self.add_shape(self.shape_sphere(  diam,          centroids[2]), section=sections[2], hydro=hydro[0])
        self.add_shape(self.shape_cylinder(diam, len_cyl, centroids[3]), section=sections[3], hydro=hydro[0])

        self.prepare()


class PoreAmorphCylinder(PoreKit):
    """Convenience builder for a cylindrical pore in the amorphous template from
    `Vink et al. <http://doi.org/10.1103/PhysRevB.67.245201>`_.

    The starting amorphous silica template has dimensions
    ``[9.605, 9.605, 9.605]`` in nanometers.

    Parameters
    ----------
        diam : float
        Cylinder diameter in nanometers.
    res : float, optional
        Reservoir size on each side in nanometers.
    hydro: list, optional
        Hydroxylation degree for interior and exterior of the pore in
        :math:`\\frac{\\mu\\text{mol}}{\\text{m}^2}`

    Examples
    --------
    Following example generates a cylindrical pore with a diameter of 4nm,
    reservoirs of 5nm on each side and a surface functionalized with TMS

    .. code-block:: python

        import porems as pms

        pore = pms.PoreAmorphCylinder(4, 5)

        pore.attach(pms.gen.tms(), 0, [0, 1], 100, "in")
        pore.attach(pms.gen.tms(), 0, [0, 1], 20, "ex")

        pore.finalize()

        pore.store("output/")
    """
    def __init__(self, diam, res=5, hydro=None):
        # Call super class
        super(PoreAmorphCylinder, self).__init__()

        hydro = hydro if hydro is not None else [0, 0]

        # Create structure
        self.structure(pms.Molecule(inp=os.path.split(__file__)[0]+"/templates/amorph.gro"))
        self.build(bonds=[0.160-0.02, 0.160+0.02])
        self._matrix.split(57790, 2524)

        # Create reservoir
        self.exterior(res, hydro=hydro[1])

        # Add pore shape
        self.add_shape(self.shape_cylinder(diam), section={"x": [], "y": [], "z": [-1, 10]}, hydro=hydro[0])
        self.prepare()


    ##############
    # Attachment #
    ##############
    def attach_special(self, mol, mount, axis, amount, scale=1, symmetry="point", is_proxi=True, is_rotate=False):
        """Attach molecules at evenly spaced special positions on the cylinder.

        Parameters
        ----------
        mol : Molecule
            Molecule to attach.
        mount : int
            Atom id on ``mol`` placed on the selected silicon site.
        axis : list
            Two atom ids defining the molecule orientation axis.
        amount : int
            Number of molecules to attach.
        scale : float, optional
            Effective lateral spacing multiplier for proximity searches.
        symmetry : str, optional
            Symmetry option, either ``"point"`` or ``"mirror"``.
        is_proxi : bool, optional
            True to consume nearby sites with silanol fills after attachment.
        is_rotate : bool, optional
            True to allow random rotation around the molecule axis.
        
        Raises
        ------
        ValueError
            Raised when the requested symmetry mode is not supported.
        """
        self._attach_special(mol, mount, axis, amount, scale, symmetry, is_proxi, is_rotate)
