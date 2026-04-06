"""Optimize cylindrical coils."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>

import numpy as np
import pyvista as pv

from bfieldtools.mesh_conductor import MeshConductor
from bfieldtools.coil_optimize import optimize_streamfunctions
from bfieldtools.contour import scalar_contour
from bfieldtools.line_conductor import LineConductor
from bfieldtools.viz import plot_3d_current_loops

from .metrics import homogeneity, efficiency, error
from .line_drawer import LineDrawer
from .file_io import get_loop_colors, export_to_kicad, _check_bounds
from .make_pcb import join_loops_at_cuts


def flatten_loops(loops):
    """Convert 3D cylindrical loops to 2D PCB coordinates.

    Unwraps a cylindrical surface by mapping each point (x, y, z) to
    (arc_length, z) where arc_length = atan2(y, x) * r and r = sqrt(x^2 + y^2).

    Parameters
    ----------
    loops : list of array, shape (N_points, 3)
        The 3D current loops on a cylindrical surface.

    Returns
    -------
    flat_loops : list of array, shape (N_points, 2)
        The flattened 2D loops in mm, with columns (arc_length, z).
    """
    flat_loops = []
    for loop in loops:
        x, y, z = loop[:, 0], loop[:, 1], loop[:, 2]
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        arc = theta * r  # arc length in meters
        flat = np.column_stack([arc * 1000, z * 1000])  # convert to mm
        flat_loops.append(flat)
    return flat_loops


class CylindricalCoil:
    """Cylindrical coil.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        A cylindrical mesh. The user is responsible for loading and scaling
        it to the desired physical dimensions before passing it in.
        A suitable mesh can be obtained via::

            from bfieldtools.utils import load_example_mesh
            mesh = load_example_mesh('open_cylinder')
            mesh.apply_scale(radius_scale)

    center : array, shape (3,)
        The center of the coil in meters.
    N_suh : int
        The number of surface harmonic basis functions. More basis functions
        give finer current resolution on the cylinder. Default is 400.

    Attributes
    ----------
    trace_width : float
        The trace width of the coil in mm.
    cu_oz : float
        The copper ounces per square feet.
    loops_ : list
        The discretized current loops.
    inductance : float
        The coil self-inductance in uH.
    length : float
        The total length of the coil in m.
    resistance : float
        The resistance of the coil in ohms.
    """

    def __init__(self, mesh, center, N_suh=400):
        self._center = np.asarray(center)
        self.trace_width = None  # in mm
        self.cu_oz = None        # oz per ft^2

        self.coil_ = MeshConductor(mesh_obj=mesh, basis_name='suh',
                                   N_suh=N_suh, process=False)
        self.loops_ = None
        self._shield = None

        self.FCu = list()
        self.BCu = list()

    def fit(self, target_points, target_field, abs_error=0.025):
        """Optimize the coil current distribution.

        Parameters
        ----------
        target_points : array, shape (n_points, 3)
            The target field evaluation points in meters.
        target_field : array, shape (n_points, 3)
            The vector target field at each point.
        abs_error : float
            The absolute error tolerance for the optimization.
        """
        target_spec = {
            "coupling": self.coil_.B_coupling(target_points),
            "abs_error": abs_error,
            "target": target_field,
        }

        kwargs = dict(mesh_conductor=self.coil_,
                      bfield_specification=[target_spec],
                      objective='minimum_ohmic_power')

        try:
            import mosek
            kwargs.update({'solver': 'MOSEK',
                           'solver_opts': {
                               'mosek_params': {mosek.iparam.num_threads: 8}
                           }})
        except ImportError:
            print('mosek not available. Using bfieldtools default solver')

        self.coil_.s, prob = optimize_streamfunctions(**kwargs)

    def discretize(self, N_contours=3, trace_width=5., cu_oz=2.):
        """Discretize the solution into N_contours current loops.

        Parameters
        ----------
        N_contours : int
            The number of contours to extract.
        trace_width : float
            The trace width in mm.
        cu_oz : float
            The trace thickness in oz/ft^2.
        """
        self.trace_width = trace_width
        self.cu_oz = cu_oz
        self.loops_ = scalar_contour(self.coil_.mesh, self.coil_.s.vert,
                                     N_contours=N_contours)
        self.line_conductor_ = LineConductor(loops=self.loops_)

    def predict(self, target_points):
        """Predict the magnetic field at target points.

        Parameters
        ----------
        target_points : array, shape (n_points, 3)
            The points at which to evaluate the field.

        Returns
        -------
        B_predicted : array, shape (n_points, 3)
            The predicted field in Tesla.
        """
        return self.coil_.B_coupling(target_points) @ self.coil_.s

    def add_shield(self, mesh):
        """Add a shielded room mesh."""
        self._shield = mesh

    def remove_shield(self):
        """Remove existing shield and return it."""
        if self._shield is not None:
            mesh = self._shield.copy()
            self._shield = None
            return mesh

    def evaluate(self, target_type, target_points, target_field,
                 target_points_z, metrics='all'):
        """Evaluate the coil performance.

        Parameters
        ----------
        target_type : str
            The type of target field, e.g. 'dc_x', 'dc_z', 'gradient_yz'.
        target_points : array, shape (n_points, 3)
            The target field evaluation points.
        target_field : array, shape (n_points, 3)
            The target field at each point.
        target_points_z : array, shape (n_points, 3)
            Points along the z-axis for gradient efficiency calculation.
        metrics : 'all' or list of str
            Which metrics to compute. Options: 'efficiency', 'error',
            'homog', 'inductance', 'resistance', 'length'.

        Returns
        -------
        scores : dict
            Computed metric values keyed by metric name.
        """
        if metrics == 'all':
            metrics = ['efficiency', 'error', 'homog',
                       'resistance', 'length']

        scores = dict()
        for metric in metrics:
            if metric == 'efficiency':
                ef, _ = efficiency(self, target_points, target_points_z,
                                   target_type)
                scores['efficiency (nT/mA)'] = ef
            elif metric == 'error':
                err = error(self, target_field, target_points, target_type)
                scores['error'] = err
            elif metric == 'homog':
                hmg = homogeneity(self, target_field, target_points,
                                  target_type)
                scores['homogeneity (%)'] = hmg
            elif metric == 'inductance':
                scores['inductance (uH)'] = self.inductance
            elif metric == 'resistance':
                scores['resistance (ohm)'] = self.resistance
            elif metric == 'length':
                scores['length (m)'] = self.length
        return scores

    @property
    def length(self):
        """The total length of the coil in meters."""
        return self.line_conductor_.length

    @property
    def resistance(self):
        """The coil resistance in ohms."""
        rho = 1.72e-8                    # ohm-m at 25C
        thickness = self.cu_oz * 35e-6  # 1 oz Cu == 35 um
        width = self.trace_width * 1e-3  # m
        return rho * self.length / (width * thickness)

    @property
    def inductance(self):
        """The coil self-inductance in uH."""
        return self.coil_.s.coil_inductance(Nloops=len(self.loops_)) * 1e6

    def plot_coil(self, discretized=True):
        """Plot the coil.

        Parameters
        ----------
        discretized : bool
            If True, plot the discretized current loops.
            If False, plot the continuous stream function.

        Returns
        -------
        plotter : pyvista.Plotter
        """
        plotter = pv.Plotter(window_size=(1500, 1700))
        if not discretized:
            self.coil_.s.plot(figure=plotter)
        else:
            plot_3d_current_loops(self.loops_, colors='auto', figure=plotter,
                                  tube_radius=0.0025)
        plotter.camera_position = 'xy'
        return plotter

    def plot_field(self, target_points):
        """Plot the magnetic field at target points.

        Parameters
        ----------
        target_points : array, shape (n_points, 3)

        Returns
        -------
        plotter : pyvista.Plotter
        """
        plotter = self.plot_coil()
        B_target = self.predict(target_points)
        plotter.add_arrows(target_points, B_target, mag=0.1)
        plotter.show()
        return plotter

    def make_cuts(self):
        """Make cuts to join loops interactively.

        The cylindrical loops are first flattened to 2D (arc_length vs z)
        for display. The user draws cut lines with the LineDrawer GUI
        (Ctrl+click to draw, U to undo). Results are stored in FCu and BCu.
        """
        import matplotlib.pyplot as plt

        flat_loops_mm = flatten_loops(self.loops_)
        # Close each loop
        closed = []
        for loop in flat_loops_mm:
            closed.append(np.vstack([loop, loop[0]]))

        colors = get_loop_colors([np.array(loop) for loop in closed])
        loops_2d = [loop.tolist() for loop in closed]

        fig = plt.figure()
        for color, loop in zip(colors, loops_2d):
            loop_arr = np.array(loop)
            plt.plot(loop_arr[:, 0], loop_arr[:, 1], color=color)
        plt.xlabel('Arc length (mm)')
        plt.ylabel('z (mm)')

        ld = LineDrawer(fig)
        line_cuts, line_cuts_shifted = ld.get_line_cuts()

        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8, 8))
        for line_cut, line_cut_shifted in zip(line_cuts, line_cuts_shifted):
            continuous_loop, reverse_paths, _, _, _, direction = \
                join_loops_at_cuts(loops_2d, line_cut, line_cut_shifted, colors)
            self.FCu.append(continuous_loop)
            self.BCu.append(reverse_paths)

            color = 'r' if direction == 'cc' else 'b'
            axes[0].plot(continuous_loop[:, 0], continuous_loop[:, 1],
                         f'{color}-', alpha=0.6)
            axes[1].plot(reverse_paths[:, 0], reverse_paths[:, 1], 'g',
                         zorder=0, linewidth=3, alpha=0.6)
        plt.show()

    def save(self, pcb_fname, kicad_header_fname, bounds=None,
             origin=(750, 750), bounds_wholeloop=True):
        """Save the coil to a KiCad PCB file.

        Parameters
        ----------
        pcb_fname : str or Path
            Output KiCad PCB filename.
        kicad_header_fname : str or Path
            Path to the KiCad header template file.
        bounds : tuple of (min_x, max_x, min_y, max_y) or None
            Only export loops within these bounds (in mm).
        origin : tuple of (x, y)
            The origin offset in mm.
        bounds_wholeloop : bool
            If True, check whether the entire loop is within bounds.
            If False, check only individual segments.
        """
        FCu_truncated = list()
        BCu_truncated = list()
        for FCu_loop, BCu_loop in zip(self.FCu, self.BCu):
            if _check_bounds(FCu_loop, bounds) or (bounds_wholeloop is False):
                FCu_truncated.append(FCu_loop)
                BCu_truncated.append(BCu_loop)

        export_to_kicad(pcb_fname=pcb_fname,
                        kicad_header_fname=kicad_header_fname,
                        origin=origin,
                        loops={'F.Cu': FCu_truncated, 'B.Cu': BCu_truncated},
                        net=1, scaling=1, trace_width=self.trace_width,
                        bounds=bounds, bounds_wholeloop=bounds_wholeloop)
