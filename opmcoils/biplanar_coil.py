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
from bfieldtools.line_conductor import LineConductor
from bfieldtools.viz import plot_3d_current_loops

from .base_coil import BaseCoil
from .line_drawer import LineDrawer
from .file_io import get_loop_colors


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
                                N_suh=N_suh, process=False)
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
                         process=False)

    # Overwrite dummy basis with stacked basis
    coil.basis = stacked_basis
    coil.inner2vert = stacked_inner2vert
    coil.vert2inner = stacked_vert2inner

    return coil

def get_sphere_points(center, n=8, sidelength=0.5):

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
        if target_type == 'gradient_xz':
            blm[4] += 1  # dBx/dz = dBz/dx (l=2, m=-1)
        elif target_type == 'gradient_yz':
            blm[6] += 1  # dBy/dz = dBz/dy (l=2, m=1)
        elif target_type == 'gradient_zz':
            blm[5] += 1  # 2dBz/dz = -dBx/dx - dBy/dy (l=2, m=0)
        elif target_type == 'gradient_xx':
            blm[3] += 1  # dBx/dx = -dBy/dy (l=2, m=-2)
        elif target_type == 'gradient_xy':
            blm[7] += 1  # dBx/dy = dBy/dx (l=2, m=2)

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


class BiplanarCoil(BaseCoil):
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
        The number of harmonics to use.
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

    _default_metrics = ['efficiency', 'error', 'homog', 'inductance',
                        'resistance', 'length', 'target_radius']

    def __init__(self, planemesh, center, N_suh=50, standoff=1.6):

        self._standoff = np.array([0, 0, standoff / 2])
        self.trace_width = None     # in mm
        self.cu_oz = None           # oz per ft^2

        # XXX: don't modify planemesh directly
        temp = planemesh.vertices[:, 2].copy()
        planemesh.vertices[:, 2] = planemesh.vertices[:, 1]
        planemesh.vertices[:, 1] = temp

        self.coil_ = mesh_to_coil(planemesh, N_suh,
                                  self._standoff, center)
        self.loops_ = None

        self._shield = None

        self.FCu = list()
        self.BCu = list()

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
                mesh_obj=self._shield, process=True, fix_normals=True,
                basis_name="vertex"
            )
            d = 1e-3
            shield_points = self._shield.vertices - d * self._shield.vertex_normals
            B_coupling += shielded_room.B_coupling(target_points) @ np.linalg.solve(
                shielded_room.U_coupling(shield_points),
                -self.coil_.U_coupling(shield_points)
            )
            print('Done')

        return B_coupling @ self.coil_.s

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

    def plot_field_2D(self):
        fig, ax = plt.subplots(2, layout='constrained')
        center = np.array([0, 0, 0])
        target_points_2D, grid = get_2D_point_grid(center, n=32,
                                                    sidelength=.7)
        field_2D = self.predict(target_points_2D)

        points = np.arange(-.35, .35, .01)
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
        check_normals : bool
            If True, visualize shield normals and near-surface points.
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

        return plotter

    def plot_coil(self, discretized=True, single=True):
        """Plot the coil.

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

    def make_cuts(self):
        """Make cuts to join loops."""
        import matplotlib.pyplot as plt
        from .make_pcb import join_loops_at_cuts

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

            continuous_loop, reverse_paths, _, _, _, direction = join_loops_at_cuts(
                loops, line_cut, line_cut_shifted, colors)
            self.FCu.append(continuous_loop)
            self.BCu.append(reverse_paths)

            color = 'r' if direction == 'cc' else 'b'
            axes[0].plot(continuous_loop[:, 0], continuous_loop[:, 1],
                         f'{color}-', alpha=0.6)
            axes[1].plot(reverse_paths[:, 0], reverse_paths[:, 1], 'g',
                         zorder=0, linewidth=3, alpha=0.6)
        plt.show()


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
