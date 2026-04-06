"""Optimize cylindrical coils."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>

import numpy as np
import pyvista as pv

from bfieldtools.mesh_conductor import MeshConductor
from bfieldtools.viz import plot_3d_current_loops

from .base_coil import BaseCoil
from .line_drawer import LineDrawer
from .file_io import get_loop_colors
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


class CylindricalCoil(BaseCoil):
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

    _default_metrics = ['efficiency', 'error', 'homog', 'resistance', 'length']

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
