"""Base class for coil optimization."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>

import numpy as np

from bfieldtools.coil_optimize import optimize_streamfunctions
from bfieldtools.contour import scalar_contour
from bfieldtools.line_conductor import LineConductor

from .metrics import homogeneity, efficiency, error
from .file_io import export_to_kicad, _check_bounds, get_loop_colors
from .make_pcb import join_loops_at_cuts
from .line_drawer import LineDrawer


class BaseCoil:
    """Base class for coil optimization.

    Subclasses must implement :meth:`predict`, :meth:`make_cuts`, and
    :meth:`plot_coil`. The constructor must initialize ``self.coil_``,
    ``self.trace_width``, ``self.cu_oz``, ``self.loops_``, ``self._shield``,
    ``self.FCu``, and ``self.BCu``.

    Attributes
    ----------
    trace_width : float
        The trace width of the coil in mm.
    cu_oz : float
        The copper ounces per square feet.
    loops_ : list
        The discretized current loops.
    """

    _default_metrics = ['efficiency', 'error', 'homog', 'inductance',
                        'resistance', 'length']

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

    def discretize(self, N_contours=40, trace_width=4., cu_oz=4.):
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

        Returns
        -------
        B_predicted : array, shape (n_points, 3)
        """
        raise NotImplementedError

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
            'homog', 'inductance', 'resistance', 'length', 'target_radius'.

        Returns
        -------
        scores : dict
            Computed metric values keyed by metric name.
        """
        if metrics == 'all':
            metrics = list(self._default_metrics)

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
            elif metric == 'target_radius':
                scores['target radius (cm)'] = target_points[:, 2].max() * 100
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

    def _prepare_loops_for_cuts(self):
        """Prepare loops for the make_cuts workflow.

        Returns
        -------
        loops_3d_mm : list of array, shape (N_points, 3)
            Closed loops in 3D coordinates (mm), used for color assignment.
        loops_2d : list of list
            Closed loops in 2D coordinates (mm), used for display and cutting.
        """
        raise NotImplementedError

    def make_cuts(self):
        """Make cuts to join loops interactively.

        Calls :meth:`_prepare_loops_for_cuts` to obtain 3D and 2D loop
        representations, then launches the LineDrawer GUI for the user to
        draw cut lines. Results are stored in ``FCu`` and ``BCu``.
        """
        import matplotlib.pyplot as plt

        loops_3d_mm, loops_2d = self._prepare_loops_for_cuts()
        colors = get_loop_colors(loops_3d_mm)

        fig = plt.figure()
        for color, loop in zip(colors, loops_2d):
            loop_arr = np.array(loop)
            plt.plot(loop_arr[:, 0], loop_arr[:, 1], color=color)

        ld = LineDrawer(fig)
        line_cuts, line_cuts_shifted = ld.get_line_cuts()

        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8, 8))
        for line_cut, line_cut_shifted in zip(line_cuts, line_cuts_shifted):
            continuous_loop, reverse_paths, _, _, _, direction = join_loops_at_cuts(
                loops_2d, line_cut, line_cut_shifted, colors)
            self.FCu.append(continuous_loop)
            self.BCu.append(reverse_paths)

            color = 'r' if direction == 'cc' else 'b'
            axes[0].plot(continuous_loop[:, 0], continuous_loop[:, 1],
                         f'{color}-', alpha=0.6)
            axes[1].plot(reverse_paths[:, 0], reverse_paths[:, 1], 'g',
                         zorder=0, linewidth=3, alpha=0.6)
        plt.show()

    def plot_coil(self, discretized=True):
        """Plot the coil."""
        raise NotImplementedError

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
