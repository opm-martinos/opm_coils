"""Optimize biplanar coils."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Padma Sundaram

import numpy as np
import scipy

from scipy.sparse import csr_matrix
from scipy.linalg import block_diag

import trimesh
import pyvista as pv

import matplotlib.pyplot as plt

from bfieldtools import sphtools
from bfieldtools.mesh_conductor import MeshConductor
from bfieldtools.utils import combine_meshes, load_example_mesh
from bfieldtools.coil_optimize import optimize_streamfunctions
from bfieldtools.contour import scalar_contour
from bfieldtools.line_conductor import LineConductor

from bfieldtools.viz import plot_3d_current_loops

from .metrics import homogeneity, efficiency, error
from .line_drawer import LineDrawer
from .file_io import get_loop_colors, export_to_kicad, _check_bounds
from .make_pcb import join_loops_at_cuts


def trimesh_to_pv(mesh):
    return pv.PolyData(mesh.vertices,
                       np.c_[np.full(len(mesh.faces), 3), mesh.faces])


def mesh_to_coil(planemesh, N_suh, standoff, center_offset):
    """Create biplanar coil for optimization."""

    # Create coil plane pairs
    coil_plus = trimesh.Trimesh(
        planemesh.vertices + center_offset + standoff, planemesh.faces,
        process=False
    )

    coil_minus = trimesh.Trimesh(
        planemesh.vertices + center_offset - standoff, planemesh.faces,
        process=False
    )

    joined_planes = combine_meshes((coil_plus, coil_minus))

    # Create separate surface harmonic bases for the planes
    coil_plus_C = MeshConductor(mesh_obj=coil_plus, basis_name='suh',
                                N_suh=N_suh, process=False) #Change the number 300 as desired
    coil_minus_C = MeshConductor(mesh_obj=coil_minus, basis_name='suh',
                                 N_suh=N_suh, process=False)

    # Combine the separate bases stacked "on top of each other"
    stacked_basis = block_diag(coil_plus_C.basis, coil_minus_C.basis)

    # Combine the basis transformation matrices
    stacked_inner2vert = csr_matrix(block_diag(coil_plus_C.inner2vert.toarray(),
                                               coil_minus_C.inner2vert.toarray()))
    stacked_vert2inner = csr_matrix(block_diag(coil_plus_C.vert2inner.toarray(),
                                               coil_minus_C.vert2inner.toarray()))

    # Create a MeshConductor with both planes but a "dummy" basis choice
    coil = MeshConductor(mesh_obj=joined_planes,
                         basis_name='inner', N_suh=10,
                         process=False) #Number of harmonics not important here, will not be used

    # Overwrite dummy basis with stacked basis
    coil.basis = stacked_basis
    coil.inner2vert = stacked_inner2vert
    coil.vert2inner = stacked_vert2inner

    return coil

def get_sphere_points(center, n=8, sidelength=0.5):

    #%%

    # Set up target and stray field points
    # Here, the target points are on a volumetric grid within a sphere

    xx = np.linspace(-sidelength / 2, sidelength / 2, n)
    yy = np.linspace(-sidelength / 2, sidelength / 2, n)
    zz = np.linspace(-sidelength / 2, sidelength / 2, n)
    X, Y, Z = np.meshgrid(xx, yy, zz, indexing="ij")

    x = X.ravel()
    y = Y.ravel()
    z = Z.ravel()

    target_points = np.array([x, y, z]).T
    axis_points = np.zeros_like(yy)[:, None]

    target_points_z = np.c_[axis_points, axis_points, zz[:, None]]

    # Turn cube into sphere by rejecting points "in the corners"
    mask = np.linalg.norm(target_points, axis=1) < sidelength / 2
    target_points = (
        target_points[mask] + center
    )

    return target_points, target_points_z

def get_2D_point_grid(center, n=8, sidelength=0.5):

    xx = np.linspace(-sidelength / 2, sidelength / 2, n)
    yy = np.linspace(-sidelength / 2, sidelength / 2, n)
    zz = np.linspace(-sidelength / 2, sidelength / 2, n)
    X, Y, Z = np.meshgrid([0], yy, zz, indexing="ij")

    x = X.ravel()
    y = Y.ravel()
    z = Z.ravel()

    target_points = np.array([x, y, z]).T
    # axis_points = np.zeros_like(yy)[:, None]
    #
    # target_points_z = np.c_[axis_points, axis_points, zz[:, None]]
    #
    # Turn cube into sphere by rejecting points "in the corners"
    target_points += center

    return target_points, xx

def get_target_field(target_type, target_points, lmax=3):
    """Set field in target region using spherical harmonics.

    Parameters
    ----------
    target_type : str
        'gradient_x', 'gradient_y' etc.
    """
    alm = np.zeros((lmax * (lmax + 2),))
    blm = np.zeros((lmax * (lmax + 2),))

    # Define target field
    if 'gradient' in target_type:

        # see Brookes (2018)
        if target_type == 'gradient_x':
            blm[4] += 1  # dBx/dz = dBz/dx (l=2, m=-1)
        elif target_type == 'gradient_y':
            blm[6] += 1  # dBy/dz = dBz/dy (l=2, m=1)
        elif target_type == 'gradient_z':
            blm[5] += 1  # dBz/dz = -dBx/dx = -2dBy/dy (l=2, m=0)

        sphfield = sphtools.field(target_points, alm, blm, lmax, R=1.)
        target_field = sphfield / np.max(sphfield)

    elif 'dc' in target_type:
        target_field = np.zeros(target_points.shape)
        if target_type == 'dc_x':
            target_field[:, 0] += 1
        elif target_type == 'dc_y':
            target_field[:, 1] += 1
        elif target_type == 'dc_z':
            target_field[:, 2] += 1

    return target_field


class BiplanarCoil:
    """Biplanar coil.

    Parameters
    ----------
    planemesh : mesh
        One of the meshes in the biplanar mesh pair. The loaded
        mesh is duplicated and positioned in space
        using the standoff and center.
    center : array, shape (3, )
        The center of the biplanar mesh pair.
    N_suh : int
        The number of harmonics to use in 
    standoff : float
        The distance between the mesh pairs.

    Attributes
    ----------
    trace_width : float
        The trace width of the coil in mm.
    cu_oz : float
        The copper ounces per square feet.
    loops_ : loop
        The discretized current loop.
    inductance : float
        The coil self-inductance in uH.
    length : float
        The total length of the coil in m.
    resistance : float
        The resistance of the coil in ohms.
    """

    def __init__(self, planemesh, center, N_suh=50, standoff=1.6):

        self._standoff = np.array([0, 0, standoff / 2])
        self.trace_width = None     # in mm
        self.cu_oz = None           # oz per ft^2

        # XXX: don't modify planemesh directly
        temp = planemesh.vertices[:, 2].copy()
        planemesh.vertices[:, 2] = planemesh.vertices[:, 1]
        planemesh.vertices[:, 1] = temp

        # XXX: ideally there should be self.coil
        # and self.coil_ but for now we'll fuse the two
        # to not deal with deepcopy
        self.coil_ = mesh_to_coil(planemesh, N_suh,
                                  self._standoff, center)
        self.loops_ = None

        self._shield = None

        self.FCu = list()
        self.BCu = list()

    def fit(self, target_points, target_field, abs_error=0.025):
        """Create bfield specifications used when optimizing the coil geometry

        The absolute target field amplitude is not of importance,
        and it is scaled to match the C matrix in the optimization function.

        Parameters
        ----------
        target_field : array, shape (n_points, 3)
            The vector target field.
        """

        target_spec = {
            "coupling": self.coil_.B_coupling(target_points),
            "abs_error": abs_error,
            "target": target_field,
        }

        bfield_specification = [target_spec]

        kwargs = dict(mesh_conductor=self.coil_,
                      bfield_specification=bfield_specification,
                      objective='minimum_ohmic_power')

        try:
            import mosek

            kwargs.update({'solver': "MOSEK",
                           'solver_opts': {
                               "mosek_params": {mosek.iparam.num_threads: 8}
                               }
                           })
        except ImportError:
            print('mosek not available. Using bfieldtools default solver')

        self.coil_.s, prob = optimize_streamfunctions(**kwargs)

    def discretize(self, N_contours=40, trace_width=4., cu_oz=4.):
        """Discretize the solution into N_contours

        Parameters
        ----------
        N_contours : int
            The number of contours.
        trace_width : float
            The tracewidth in mm.
        cu_oz : float
            The trace thickness in oz/ft^2.
        """
        self.trace_width = trace_width
        self.cu_oz = cu_oz
        self.loops_ = scalar_contour(self.coil_.mesh, self.coil_.s.vert,
                                     N_contours=N_contours)
        self.line_conductor_ = LineConductor(loops=self.loops_)

    def predict(self, target_points):
        """Predict the field.

        Parameters
        ----------
        target_points : array, (n_points, 3)
            Plot the field at the target points.

        Returns
        -------
        B_predicted : array, (n_points, 3)
            The predicted field at the target points.
        """
        B_coupling = self.coil_.B_coupling(target_points)
        if self._shield is not None:
            print('Computing effect of shielded room')
            shielded_room = MeshConductor(
                mesh_obj=self._shield, process=True, fix_normals=True, basis_name="vertex"
            )
            d = 1e-3
            shield_points = self._shield.vertices - d * self._shield.vertex_normals
            B_coupling += shielded_room.B_coupling(target_points) @ np.linalg.solve(
                shielded_room.U_coupling(shield_points), -self.coil_.U_coupling(shield_points)
            )
            print('Done')

        B_expected = B_coupling @ self.coil_.s
        return B_expected

    def add_shield(self, mesh):
        """Add a shielded room."""
        self._shield = mesh

    def remove_shield(self):
        """Remove existing shield"""
        if self._shield is not None:
            mesh = self._shield.copy()
            self._shield = None
            return mesh

    def evaluate(self, target_type, target_points, target_field,
                 target_points_z, metrics='all'):
        """Evaluate the coil.
        
        Parameters
        ----------
        target_points : array, (n_points, 3)
            Plot the field at the target points.
        """
        if metrics == 'all':
            metrics = ['efficiency', 'error', 'homog', 'inductance',
                       'resistance', 'length', 'target_radius']

        scores = dict()
        for metric in metrics:
            if metric == 'efficiency':
                ef, _ = efficiency(self, target_points,
                                   target_points_z, target_type)
                scores['efficiency (nT/mA)'] = ef
            elif metric == 'error':
                err = error(self, target_field, target_points, target_type)
                scores['error'] = ef
            elif metric == 'homog':
                hmg = homogeneity(self, target_field, target_points, target_type)
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
        """The total length of the coil."""
        return self.line_conductor_.length

    @property
    def shape(self):
        """The length and width of the coil."""
        loops = [loop for loop in self.loops_ if loop[0, 2] > 0]
        min_x = np.min([loop[:, 0].min() for loop in loops])
        max_x = np.max([loop[:, 0].max() for loop in loops])

        min_y = np.min([loop[:, 1].min() for loop in loops])
        max_y = np.max([loop[:, 1].max() for loop in loops])

        return (self.trace_width * 1e-3 + (max_x - min_x),
                self.trace_width * 1e-3 + (max_y - min_y))

    @property
    def resistance(self):
        """The coil resistance."""
        # this formulation for PCB resistance matches the internet calculators
        rho = 1.72e-8                       # ohm-m at 25C
        thickness = self.cu_oz * 35e-6      # (1 oz cu == 35 um thick)
        width = self.trace_width * 1e-3     # m

        resistance = rho * self.length / (width * thickness)
        return resistance

    @property
    def inductance(self):
        """The coil self-inductance"""
        return self.coil_.s.coil_inductance(Nloops=len(self.loops_)) * 1e6

    def plot_field_2D(self):
        fig, ax = plt.subplots(2, layout='constrained')
        center = np.array([0, 0, 0])
        target_points_2D, grid = get_2D_point_grid(center, n=32,
                                                    sidelength=.7)
        field_2D = self.predict(target_points_2D)

        points = np.arange(-.35,.35,.01)
        n_points = np.shape(points)[0]
        profile_ax = 2
        target_points = np.zeros((n_points, 3))
        target_points[:, profile_ax] = points
        field_1D = self.predict(target_points)

        ax[0].pcolormesh(grid, grid, field_2D[:, 2].reshape(len(grid), len(grid)))
        ax[1].plot(points, field_1D[:, profile_ax] * 1e9, 'bo-')

    def plot(self, check_normals=False):
        """Plot the coil.

        Parameters
        ----------
        target_points : array, (n_points, 3)
            Plot the field at the target points.
        """
        if self._shield is not None:
            shield = trimesh_to_pv(self._shield)

        plotter = self.plot_coil(single=False)
        plotter.show_axes()

        if self._shield is not None:
            plotter.add_mesh(shield, opacity=0.1)
            if check_normals:
                plotter.add_arrows(self._shield.vertices[::100],
                                   self._shield.vertex_normals[::100], mag=0.3)
                d = 1e-3
                shield_points = self._shield.vertices - d * self._shield.vertex_normals
                plotter.add_points(shield_points[::40], point_size=10)

        plotter.view_isometric()
        plotter.camera.roll = 0.
        # plotter.camera.elevation = -45.
        # plotter.camera.azimuth = 30.
        # plotter.camera.zoom(0.7)

        return plotter

    def plot_coil(self, discretized=True, single=True):
        """Plot the coil

        Parameters
        ----------
        discretized : bool
            Plot the discretized coil.
        single : bool
            Plot only one coil loop in the pair.
            Applies only if discretized is True.

        Returns
        -------
        plotter : pyvista.Plotter
            The plotter object.
        """
        plotter = pv.Plotter(window_size=(1500, 1700))
        if not discretized:
            self.coil_.s.plot(figure=plotter)
        else:
            loops = self.loops_
            if single:
                loops = [loop for loop in self.loops_ if loop[0, 2] > 0]
            plot_3d_current_loops(loops, colors='auto',
                                  figure=plotter, tube_radius=0.0025,
                                  origin=self._standoff * 8)

        plotter.camera_position = 'xy'
        return plotter

    def plot_field(self, target_points):
        """Plot the field.

        Parameters
        ----------
        target_points : array, (n_points, 3)
            Plot the field at the target points.
        """
        # XXX: add option to plot non-discretized stream functions

        # plotter = pvqt.BackgroundPlotter(window_size=(1500, 1700))
        plotter = self.plot()
        B_target = self.predict(target_points)
        plotter.add_arrows(target_points, B_target, mag=0.1)
                           # clim=(0.075, 0.125), cmap='jet')

        plotter.show()

        return plotter
        # mlab.quiver3d(*target_points.T, *B_target.T, mode="arrow",
        #               scale_factor=0.1)


    def make_cuts(self):
        """Make cuts to join loops."""
        import matplotlib.pyplot as plt

        # Discard one panel of the pair
        loops = list()
        for loop in self.loops_:
            if np.allclose(loop[:, 2], self._standoff[2]):
                loop = [pt for pt in loop] + [loop[0]]  # make closed loop
                loops.append((np.array(loop) * 1000))

        colors = get_loop_colors([np.array(loop) for loop in loops])
        # Discard z-coordinate
        loops = [np.array(loop)[:, [0, 1]].tolist() for loop in loops]

        fig = plt.figure()
        for color, loop in zip(colors, loops):
            loop_arr = np.array(loop)
            plt.plot(loop_arr[:, 0], loop_arr[:, 1], color=color)

        ld = LineDrawer(fig)
        line_cuts, line_cuts_shifted = ld.get_line_cuts()

        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8, 8))
        for line_cut, line_cut_shifted in zip(line_cuts, line_cuts_shifted):

            continuous_loop, reverse_paths, _, _, _, direction = join_loops_at_cuts(loops, line_cut, line_cut_shifted, colors)
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
        """Save the files to be loaded in KICAD.

        Parameters
        ----------
        pdb_fname : str
            The file name of the KICAD pcb file.
        kicad_header_fname : str
            The file name of the KICAD header.
        bounds : tuple of (min_x, max_x, min_y, max_y)
            Save only loops within the bounds expressed in mm.
        origin : tuple of (x, y)
            The origin in mm.
        """
        FCu_truncated = list()
        BCu_truncated = list()
        for FCu_loop, BCu_loop in zip(self.FCu, self.BCu):
            if _check_bounds(FCu_loop, bounds) or (bounds_wholeloop is False):
                FCu_truncated.append(FCu_loop)
                BCu_truncated.append(BCu_loop)
        
        #print("in save\n")
        #print(len(FCu_truncated))
        #print(len(BCu_truncated))
        export_to_kicad(pcb_fname=pcb_fname,
                        kicad_header_fname=kicad_header_fname,
                        origin=origin,
                        loops={'F.Cu': FCu_truncated, 'B.Cu': BCu_truncated},
                        net=1, scaling=1, trace_width=self.trace_width,
                        bounds=bounds, bounds_wholeloop=bounds_wholeloop)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    center = np.array([0, 0, 0])
    scaling_factor = 0.16
    standoff = 1.6
    target_type = 'dc_z'

    planemesh = load_example_mesh("10x10_plane_hires")
    planemesh.apply_scale(scaling_factor)

    coil = BiplanarCoil(planemesh, center, N_suh=50, standoff=1.6)

    target_points, points_z = get_sphere_points(center, n=8, sidelength=0.5)
    target_field = get_target_field(target_type, target_points)

    coil.fit(target_points, target_field)
    coil.discretize(N_contours=40, trace_width=4., cu_oz=3.)
    coil.plot_field(target_points)

    effs = list()
    Rs = list()
    n_contours = [30, 35, 40]
    for n_contour in n_contours:
        coil.discretize(N_contours=n_contour, trace_width=4.,
                        cu_oz=3.)
        ef = coil.evaluate(target_type, target_points, points_z,
                           metric='efficiency')
        effs.append(ef)
        Rs.append(coil.resistance)

    plt.plot(n_contours, effs, 'b-', linewidth=3)
    plt.xlabel('n_contours')
    plt.ylabel('Efficiency (nT/mA)')
    ax = plt.twinx(plt.gca())
    ax.plot(n_contours, Rs, 'ro-')
    ax.set_ylabel('Resistance (ohm)')
    plt.show()
