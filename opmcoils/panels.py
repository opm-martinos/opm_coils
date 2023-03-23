# Authors: Mainak Jas <mainakjas@gmail.com>
#          Gabriel Motta <gabrielbenmotta@gmail.com>

from pathlib import Path

import numpy as np

import os

import copy

import pyvista as pv

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from bfieldtools.line_conductor import LineConductor
from shapely.geometry import LineString
from shapely.geometry import Point

from .biplanar_coil import get_2D_point_grid


def segment_len(seg):
    return (((seg[0][1]-seg[1][1])**2) + ((seg[0][2] - seg[1][2])**2))**.5


def kicad_to_loops(fname, offset, min_len=.1, mult=1000):
    """Load in front, back and via information from kicad file.
    Applies scaling, offset, and rejection based on input parameters.

    Parameters
    ----------
    fname : string
        file path to kicad file.
    offset : list of 3 coordinates
        initial offset for loaded points.
    min_len : float
        discard any segment with length smaller than this value.
    mult : float
        scale points according to this multiplier.
    """
    with open(fname, 'r') as fp:
        lines = fp.readlines()

    if not offset:
        offset = [0, 0, 0]
    lines = [line.strip(' ').strip('\n') for line in lines]
    loops = {'F.Cu': list(), 'B.Cu': list(), 'via': list()}
    for layer in ['F.Cu', 'B.Cu']:
        lines_layer = [line for line in lines if line.startswith('(segment')
                       and layer in line]
        for line in lines_layer:
            parts = line.split('(')
            parts = [part.strip('start').strip('end').strip(' ').strip(')')
                     for part in parts if part.startswith(('start', 'end'))]
            xstart, ystart = [float(coord) for coord in parts[0].split(' ')]
            xend, yend = [float(coord) for coord in parts[1].split(' ')]

            seg = [[offset[0],
                    xstart / mult + offset[1],
                    ystart / mult + offset[2]],
                   [offset[0],
                    xend / mult + offset[1],
                    yend / mult + offset[2]]]
            if segment_len(seg) > min_len:
                loops[layer].append(seg)

    via_lines = [line for line in lines if line.startswith('(via')
                 and layer[0] in line and layer[1] in line]

    for line in via_lines:
        parts = line.split('(')
        parts = [part.strip('at').strip(' ').strip(')') for part in parts
                 if part.startswith('at')]
        x, y = [float(coord) for coord in parts[0].split(' ')]

        via = [0, x, y]
        loops['via'].append(via)

    return loops


def _move_loops(loops, offset):
    new_loops = dict()
    for layer in ['F.Cu', 'B.Cu']:
        new_loops[layer] = list()
        for loop in loops[layer]:
            for seg in loop:
                seg = (np.array(seg) + offset).tolist()
                new_loops[layer].append(seg)
    return new_loops


def do_intersect(seg1, seg2, tolerance=.02):
    line1 = LineString(seg1)
    line2 = LineString(seg2)

    return not line1.intersection(line2, grid_size=tolerance).is_empty


def dist(pt1, pt2):
    return np.linalg.norm(np.array(pt2) - np.array(pt1))

"""
def do_intersect(seg1, seg2, tol=1e-3):
    if dist(seg1[0], seg2[0]) < tol:
        return True
    elif dist(seg1[1], seg2[0]) < tol:
        return True
    elif dist(seg1[0], seg2[1]) < tol:
        return True
    elif dist(seg1[1], seg2[1]) < tol:
        return True
    return False

# a bit less robust to points like:
(start 114.22 274.22) (end 114.22 274.22)
"""


def get_chain(segments, tolerance=.02, verbose=True):
    """ Build continous chains from a list of segments

    Parameters
    ----------
    segments : list of line segments (2 3D points)
        segments to be connected into chains
    tolerance : float
        how close points need to be to be considered overlapping.
    verbose : bool
        whether to print progress information.
    """
    z_offset = segments[0][0][0]
    segments = np.array(segments)[:, :, 1:].tolist()  # remove z-coordinate
    chains = list()
    print(end='')
    init_len = len(segments)
    while len(segments) > 0:
        chain = [segments[0]]
        segments.remove(segments[0])
        keep_looping = True
        while keep_looping:
            keep_looping = False
            for seg in segments:
                if seg in chain:
                    segments.remove(seg)
                    keep_looping = True
                    break
                elif do_intersect(seg, chain[-1], tolerance):
                    chain.append(seg)
                    segments.remove(seg)
                    keep_looping = True
                    break
                elif do_intersect(seg, chain[0], tolerance):
                    chain.insert(0, seg)
                    segments.remove(seg)
                    keep_looping = True
                    break
            if verbose:
                print(f'\rChains: {len(chains) + 1},' +
                      f' remaining segments: {len(segments)}/{init_len}      ')
                loading_bar(1 - len(segments) / init_len)
                print("\033[F", end='')
        chain = np.array(chain)
        chain_3d = np.zeros((chain.shape[0], chain.shape[1], 3))
        chain_3d[:, :, 1:] = chain
        chain_3d[:, :, 0] = z_offset
        chains.append(chain_3d)

    if verbose:
        print()
        print(70 * ' ' + '\r', end='')
    return chains


def loading_bar(ratio, length=30, full_char='#', empty_char='-'):
    """Prints a loading bar"""
    comp = min(int(ratio * length), length)
    loaded = "".join([full_char for _ in range(comp)])
    unloaded = "".join([empty_char for _ in range(length - comp)])
    print(f'\r[{loaded}{unloaded}] {int(100 * ratio)}%', end='', flush=True)


def link_chains(chain1, chain2, precision):
    """Given two chains, tries to return a linked configuration.

    Parameters
    ----------
    chain1 : list of line segments (2 3d points)
        one of the chains to be linked.
    chain2 : list of line segments (2 3d points)
        one of the chains to be linked.
    precision : float
        how close points need to be to be considered overlapping.
    """
    c2_pts = [Point(chain2[-1][0][1:]), Point(chain2[-1][1][1:]),
              Point(chain2[0][0][1:]), Point(chain2[0][1][1:])]

    def linkable(point, ind):
        nonlocal c2_pts
        nonlocal precision
        return point.dwithin(c2_pts[ind], precision)

    for c1_point in chain1[0]:
        point = Point(c1_point[1:])
        if linkable(point, 0) or linkable(point, 1):
            return np.concatenate((chain2, chain1))
        elif linkable(point, 2) or linkable(point, 3):
            return np.concatenate((np.flip(chain2, 0), chain1))

    for c1_point in chain1[-1]:
        point = Point(c1_point[1:])
        if linkable(point, 0) or linkable(point, 1):
            return np.concatenate((chain1, np.flip(chain2, 0)))
        elif linkable(point, 2) or linkable(point, 3):
            return np.concatenate((chain1, chain2))

    return None


def combine_chains(chains, precision=2):
    """Attempts to link together chains.

    Parameters
    ----------
    chains : list of line segments (2 3D points)
        the chains to be combined.
    precision : float
        how close segments have to be to count as 'overlapping'
    """
    for i in range(len(chains)):
        candidate = chains[0]
        chains.pop(0)

        for _ in range(len(chains)):
            for i, chain in enumerate(chains):
                new_c = link_chains(candidate, chain, precision)
                if new_c is not None:
                    candidate = new_c
                    chains.pop(i)
                    break
        chains.append(candidate)
    return chains


def combine_while_bypassing(chains, precision):
    """ Tries to combine chains while considering beyond the endpoints.

    Parameters
    ----------
    chains : list of list of line segements (2 3D points)
        chains to be combined.
    precision : float
        how close points need to be to be considered overlapping.
    """
    for i in range(len(chains)):
        if len(chains) == 1:
            break
        candidate = chains[0]
        chains.pop(0)
        if len(candidate) == 1:
            chains.append(candidate)
            continue

        found = False
        for cand in [candidate[1:], candidate[:-1]]:
            if found:
                break
            for i, chain in enumerate(chains):
                new_c = link_chains(cand, chain, precision)
                if new_c is not None:
                    print('Bypassing chain endpoint to join chains.')
                    candidate = new_c
                    chains.pop(i)
                    found = True
                    break
        chains.append(candidate)
    return chains


def allign_chain(chain):
    """ Make all links in a chain point in a single direction, such that the
    first point of a link is linked to the second point of the preceeding link.
    The given chain is modified in place.

    Parameters
    ----------
    chain : array of pairs of points
        Array to be alligned.
    """
    if len(chain) < 2:
        return

    i = 1
    dists = [0, 0, 0, 0]
    dists[0] = Point(chain[i-1][1][1:]).distance(Point(chain[i][0][1:]))
    dists[1] = Point(chain[i-1][1][1:]).distance(Point(chain[i][1][1:]))
    dists[2] = Point(chain[i-1][0][1:]).distance(Point(chain[i][0][1:]))
    dists[3] = Point(chain[i-1][0][1:]).distance(Point(chain[i][1][1:]))
    min_ind = dists.index(min(dists))
    if min_ind in [2, 3]:
        chain[0][0], chain[0][1] = chain[0][1], chain[0][0]

    for i in range(1, len(chain)):
        dist_norm = Point(chain[i-1][1][1:]).distance(Point(chain[i][0][1:]))
        dist_flipped = Point(chain[i-1][1][1:]).distance(Point(chain[i][1][1:]))
        if dist_flipped < dist_norm:
            chain[i] = np.flip(chain[i], axis=0)


def flip_chain(chain):
    """ Flips the orientation of each link in the chain, such that the start
    and end points are flipped. This is done in place.

    Parameters
    ----------
    chain : array of pairs of points
        Array to be flipped.
    """
    if len(chain) < 1:
        return

    for i in range(len(chain)):
        chain[i] = np.flip(chain[i], axis=0)


def flip_chains(chain_list):
    """ Flips the orientation of each link in the chain, such that the start
    and end points are flipped. This is done in place.

    Parameters
    ----------
    chain : array of pairs of points
        Array to be flipped.
    """
    for chain in chain_list:
        flip_chain(chain)


def plot_chains_2D(chains):
    ax = plt.figure().add_subplot()

    loops = dict()
    layers = list()
    for idx, chain in enumerate(chains):
        loops[f'chain{idx}'] = chains[idx]
        layers.append(f'chain{idx}')

    colors = list(mcolors.TABLEAU_COLORS.keys())

    while len(colors) < len(chains):
        colors = colors + colors
    num = 0
    for color, layer in zip(colors, layers):
        if layer == 'B.Cu':
            continue
        loop_array = np.array(loops[layer])
        cen = loop_array[:, 0, :]
        arrow_dir = loop_array[:, 1, :] - loop_array[:, 0, :]
        if layer == 'B.Cu':
            arrow_dir *= -1
        print(f'Layer {layer}, color {color}')
        arrows = ax.quiver(cen[:, 1], cen[:, 2], arrow_dir[:, 1],
                           arrow_dir[:, 2], color=color,
                           scale_units='xy', scale=1, angles='xy')

        if layer not in ['F.Cu', 'B.Cu']:
            ax.quiver(cen[0][1], cen[0][2], arrow_dir[0][1],
                      arrow_dir[0][2], color='y', scale_units='xy',
                      scale=1, angles='xy')
            ax.quiver(cen[-1][1], cen[-1][2], arrow_dir[-1][1],
                      arrow_dir[-1][2],
                      color='r', scale_units='xy', scale=1,
                      angles='xy')
            ax.annotate(f'{num}',(cen[0][1], cen[0][2]))
        num += 1


def plot_chains(chains, ax=None, pl=None):
    """ Plots chains with arrows as the links.

    Parameters
    ----------
    chains : list of arrays of pairs of points
        Chains to be plotted.
    ax : matplotlib element
        matplotlib subplot to use. If none, a new one is created.
    """

    # if not ax:
    #     ax = plt.figure().add_subplot()

    if pl is None:
        pl = pv.Plotter()

    colors = list(mcolors.TABLEAU_COLORS.keys())

    while len(colors) < len(chains):
        colors = colors + colors

    for color, chain in zip(colors, chains):
        loop_array = np.array(chain)
        cen = loop_array[:, 0, :]
        arrow_dir = loop_array[:, 1, :] - loop_array[:, 0, :]

        pl.add_arrows(cen, arrow_dir, color=color)
        pl.add_arrows(cen[0], arrow_dir[0], color='y', mag=5)
        pl.add_arrows(cen[-1], arrow_dir[-1], color='r', mag=5)

        # XXX: sphinx_gallery does not like below statement
        # pl.add_axes(xlabel='Z', ylabel='X', zlabel='Y')

    return pl


class PCB:

    def __init__(self, fname=None, offset=None, loops=None, pcb_dict=None):
        """Class to represent a PCB loaded from KiCAD.

        Attributes
        ----------
        loops : dict of list
            The keys may be 'via', 'F.Cu', 'B.Cu' for via,
            front, and back copper layer.
        """
        self.chains = None

        if pcb_dict is not None:
            self.chains = pcb_dict['chains']
            return

        if loops is not None:
            self.loops = loops
        else:
            self.loops = kicad_to_loops(fname, offset, mult=1)
        self.check = None

    def get_chains(self):
        """Returns continuous loops in pcb, builds them if needed."""
        if self.chains is not None:
            return self.chains

        def build_chain(segments, tolerances):
            chains = get_chain(segments, tolerance=.02)
            initial_size = len(chains)
            for tol in tolerances:
                chains = combine_chains(chains, tol)
            chains = [ch for ch in chains if len(ch) > 1 or segment_len(ch[0]) > 1]
            return chains, initial_size

        tol_iters = [.1, .5, 1, 2]

        print('Parsing front...')
        if 'F.Cu' in self.loops.keys() and len(self.loops['F.Cu']) > 0:
            front_chains, front_size = build_chain(self.loops['F.Cu'], tol_iters)
        else:
            front_chains = list()
            front_size = 0

        print('Parsing back...')
        if 'B.Cu' in self.loops.keys() and len(self.loops['B.Cu']) > 0:
            back_chains, back_size = build_chain(self.loops['B.Cu'], tol_iters)
        else:
            back_chains = list()
            back_size = 0

        print('Combining...', end='\r')
        chains = front_chains + back_chains
        for tol in tol_iters:
            chains = combine_chains(chains, tol)

        if len(chains) > 1:
            chains = combine_while_bypassing(chains, 1)

        print(f'Combining: {front_size + back_size} --> {len(chains)}')

        for chain in chains:
            allign_chain(chain)

        self.chains = chains
        return self.chains

    def plot(self, ax=None, pl=None, show=False):
        """Plots PCB

        Parameters
        ----------
        ax : matplotlib Axes
            The matplotlib axis
        pl : pyvista.Plotter object
            The plotter object.
        """
        chains = self.get_chains()
        plot_chains(chains, ax, pl)
        if show:
            pl.show()

        return pl

    def plot2D(self, ax=None, pl=None):
        """Plots PCB"""
        chains = self.get_chains()
        plot_chains_2D(chains)
        if pl:
            pl.show()

        return pl

    @property
    def length(self):
        # XXX: todo add repr which prints number of segments
        total_len = 0
        for chain in self.chains:
            for segment in chain:
                total_len += dist(segment[0], segment[1]) 
        return total_len

    def magnetic_field(self, target_points):
        """Calculate total magnetic field from both layers for unit current.

        Parameters
        ----------
        target_points : array, shape (n_points, 3)
            The target points at which to evaluate the magnetic fields.

        Returns
        -------
        field : array, shape (n_points, 3)
            The magnetic field at the target points
        """
        field = list()

        for chain in self.chains:
            line_conductor = LineConductor(chain)
            field.append(line_conductor.magnetic_field(target_points))

        return sum(field)

    def adjust_offset(self, offsets):
        """Adjust the offset for a panel.

        Parameters
        ----------
        offsets : list of length 3
            The x, y, and z offset
        """
        for segment in self.chains:
            segment[:, :, 0] = offsets[0]
            segment[:, :, 1] += offsets[1]
            segment[:, :, 2] += offsets[2]

    def adjust_scale(self, mult):
        for segment in self.chains:
            segment[:, :, 1:] /= mult

    def to_dict(self):
        """Convert to dictionary"""
        pcb_dict = dict()
        pcb_dict['chains'] = self.chains
        return pcb_dict


class PCBPanel:

    def __init__(self, pcb_folder=None, half_names=None, standoff=None,
                 rearrange=False, panel_dict=None):
        """A Panel is a collection of PCBs used to null one field component.

        Parameters
        ----------
        pcb_folder : str
            The path to the kicad folder with the PCB design.
        half_names : list of str
            The names of the halves used for folder naming. E.g.,
            ['top', 'bott']
        standoff : float
            The standoff between the two biplanar PCBs. Typically,
            this is equal to the length of the PCBs, i.e., 1.4 m.

        Attributes
        ----------
        pcbs : dict of PCB
            keys of dictionary can be 'left_top', 'left_bott',
            'right_top', 'right_bott' etc.
        """
        self.pcbs = dict()
        self.chains = dict()

        if not pcb_folder and not half_names and not standoff:
            if panel_dict:
                for key, pcb in panel_dict.items():
                    self.pcbs[key] = PCB(pcb_dict=pcb)
                    self.chains[key] = self.pcbs[key].chains
            return

        pcb_folder = Path(pcb_folder)
        shift_xs = {'left': -standoff / 2.,  # in m
                    'right': standoff / 2.}

        for half_name in half_names:
            if half_name == 'top' or (rearrange and half_name == 'first'):
                shift_y = -0.75
                shift_z = 0
            elif half_name == 'bott' or (rearrange and half_name == 'second'):
                shift_y = -0.75
                shift_z = -0.75
            elif half_name == 'first':
                shift_y = 0
                shift_z = -0.75
            elif half_name == 'second':
                shift_y = -0.75
                shift_z = -0.75

            fname = pcb_folder / half_name / f'coil_template_{half_name}.kicad_pcb'
            pcb = PCB(fname)  # offsets in mm
            pcb.get_chains()

            for dir, shift_x in shift_xs.items():
                offset = np.array([shift_x, shift_y, shift_z])
                offset_pcb = copy.deepcopy(pcb)
                offset_pcb.adjust_scale(1000)
                offset_pcb.adjust_offset(offset)
                self.pcbs[f'{dir}_{half_name}'] = offset_pcb
                self.chains[f'{dir}_{half_name}'] = offset_pcb.get_chains()

    @property
    def length(self):
        """The total length of the panel."""
        total_len = 0
        for pcb_name, pcb in self.pcbs.items():
            total_len += pcb.length
        return total_len

    def resistance(self, cu_oz=2, trace_width=5):
        """The coil resistance."""
        # this formulation for PCB resistance matches the internet calculators
        rho = 1.72e-8                       # ohm-m at 25C
        thickness = cu_oz * 35e-6      # (1 oz cu == 35 um thick)
        width = trace_width * 1e-3     # m

        resistance = rho * self.length / (width * thickness)
        return resistance

    def adjust_standoff(self, standoff):
        """Adjust the standoff.

        Parameters
        ----------
        standoff : float
            The new standoff distance (in m).
        """
        for pcb_name, pcb in self.pcbs.items():
            if 'left' in pcb_name:
                shift = -standoff / 2.
            else:
                shift = standoff / 2.
            pcb.adjust_offset(np.array([shift, 0, 0]))

    def magnetic_field(self, target_points, current):
        """Calculate total magnetic field from panel.

        Parameters
        ----------
        current : dict
            The current.
        """
        field = 0.
        for pcb_name, pcb in self.pcbs.items():
            dir, half_name = pcb_name.split('_')
            this_current = current[dir]
            field += pcb.magnetic_field(target_points) * this_current

        return field

    def build_chains(self):
        """Link PCBs into continuous loops."""
        self.chains.clear()
        for pcb_name, pcb in self.pcbs.items():
            print(f'[[ Parsing {pcb_name} ]]')
            self.chains[pcb_name] = pcb.get_chains()
            print()

    def plot(self, target_points=None, current=None, pl=None, show=True):
        """Plot the PCBs."""
        if pl is None:
            pl = pv.Plotter()
        for pcb_name, pcb in self.pcbs.items():
            pcb.plot(pl=pl, show=False)
        if target_points is not None and current is not None:
            field = self.magnetic_field(target_points, current)
            pl.add_arrows(target_points, field, mag=1e7)

        if show:
            # pl.remove_scalar_bar()
            pl.show()

        return pl

    def plot_profile(self, current, profile_dir='x', field_component='x',
                     spacing=0.01, min_pos=-0.2, max_pos=0.2, ax=None):
        """Plot line profile for panel.

        Parameters
        ----------
        current : dict of floats
            current to be applied to the panel.
        profile_dir : string ('x', 'y', or 'z')
            axis along which the field will plotted.
        field_component : string ('x', 'y', or 'z') 
            axis of field to be plotted.
        spacing : float
            distance between points
        min_pos : float 
            negative disctance from (0,0) in meters
        max_pos : float 
            positive disctance from (0,0) in meters
        ax : Axes
            matplotlib axes on which to create the plot
        """
        plot_panel_profile(self, current, profile_dir, field_component,
                           spacing, min_pos, max_pos, ax)

    def to_dict(self):
        panel_dict = dict()
        for key, pcb in self.pcbs.items():
            panel_dict[key] = pcb.to_dict()
        return panel_dict


def plot_panel_profile(panel, current, profile_dir='x', field_component='x',
                       spacing=0.01, min_pos=-0.2, max_pos=0.2, ax=None):
    """Plot line profile for panel.

    Parameters
    ----------
    panel : PCBPanel
        panel to plot the field for.
    current : dict of floats
        current to be applied to the panel.
    profile_dir : string ('x', 'y', or 'z')
        axis along which the field will plotted.
    field_component : string ('x', 'y', or 'z') 
        axis of field to be plotted.
    spacing : float
        distance between points
    min_pos : float 
        negative disctance from (0,0) in meters
    max_pos : float 
        positive disctance from (0,0) in meters
    ax : Axes
        matplotlib axes on which to create the plot
    """
    if not ax:
        _, ax = plt.subplots()
    points = np.arange(min_pos, max_pos, spacing)
    n_points = np.shape(points)[0]
    profile_ax = dict(x=1, y=2, z=0)[profile_dir]
    target_points = np.zeros((n_points, 3))
    target_points[:, profile_ax] = points

    field = panel.magnetic_field(target_points, current=current)

    field_ax = dict(x=1, y=2, z=0)[field_component]
    ax.plot(points, field[:, field_ax] * 1e9, 'bo-')  # plot in nT
    ax.set_ylabel(f'B{field_component} (nT)')
    ax.set_xlabel(f'{profile_dir} (m)')


def plot_field_colormap(field, grid, axis, ax=None, colorbar=True,
                        vmin=None, vmax=None):
    """Plot colormap of one field axis along an x-z plane between panels.

    Parameters
    ----------
    field : array of 3D points
        x,y,z field values to be plotted
    grid : array of floats
        values of distances to used to get target point grid.
    axis : string ('x', 'y', or 'z')
        what field component to plot
    ax : matplotlib element
        matplotlib subplot to use. If none, a new one is created.
    colorbar : bool
        whether to show a colorbar.
    """
    if not ax:
        ax = plt.figure().add_subplot()

    opt = {'y': 2, 'x': 1, 'z': 0}

    vals = dict()
    if vmin:
        vals['vmin'] = vmin
    if vmax:
        vals['vmax'] = vmax

    cm = ax.pcolormesh(grid, grid,
                       field[:, opt[axis]].reshape(len(grid), len(grid)), **vals)
    ax.set_xlabel('z (m)')
    ax.set_ylabel('x (m)')
    if colorbar:
        fig = ax.get_figure()
        fig.colorbar(cm, label=f'B{axis} (nT)')


def plot_field_arrows(target_points, field, ax=None, field_mult=1e7):
    """Plot x, z field components as arrows on a 2D grid

    Parameters
    ----------
    target_points : array of 3D points
        target points for which fields will be plotted.
    field : array of 3D points
        x, y, z field values at each target point
    ax : matplotlib element
        matplotlib subplot to use. If none, a new one is created.
    field_mult : float
        value by which to multiply field values so they are visible in plot.
    """
    if not ax:
        ax = plt.figure().add_subplot()

    ax.quiver(target_points[:, 0], target_points[:, 1],
              field[:, 0] * field_mult, field[:, 1] * field_mult,
              scale_units='xy', scale=1)


def check_half_names(pcb_folder):
    halves = ['top', 'bott', 'first', 'second']
    use_names = list()

    for half in halves:
        if os.path.isdir(os.path.join(pcb_folder, half)):
            use_names.append(half)
    return use_names


def load_panel(path, standoff=1.4, flip=None, rearrange=False):
    """Loads and returns PCBPanel

    Parameters
    ----------
    path : string
        path to pcb folder.
    standoff : float
        distance between the two instances of the board.
    flip : list of str | dict
        names of which halves of the pcbs to flip. If it is
        a dict, the keys are the names of the halves and values
        are the chain index.
    rearrange : bool
        whether to treat 'first/second' halves as 'top/bott'

    Returns
    -------
    panel : PCBPanel
        panel object for the given pcb panel.
    """
    if os.path.exists(path):
        half_names = check_half_names(path)
    else:
        raise FileExistsError(f'{path} does not exist')

    panel = PCBPanel(path, half_names=half_names,
                     standoff=standoff, rearrange=rearrange)

    if flip is None:
        flip = dict()

    if isinstance(flip, list):
        flip = {k: None for k in flip}

    for item in flip:
        if item not in panel.chains:
            raise ValueError(f'Cannot flip {item}, not a valid panel.')

        if flip[item] is None:
            flip_chains(panel.chains[item])
        elif isinstance(flip[item], list):
            flip_chains([panel.chains[item][idx] for idx in flip[item]])

    return panel


def plot_panel(panel, target_size, n_points, current,
               axis, title='', show=True):
    """Creates 2d colormap, line profile, and arrow field map for coil.

    Parameters
    ----------
    panel : PCBPanel
        panel to be plotted.
    target_size : float
        length of the side of the square/line to be considered when plotting.
    n_points : int
        number of points per side for the square grid.
    current : dict of floats
        current to be applied to the panels when computing field.
    axis : string ('x', 'y', 'z')
        which field axis to plot values for.
    title : string
        title of the plot
    show : bool
        If True, show the plot
    """
    center = np.array([0, 0, 0])
    target_points, grid = get_2D_point_grid(center, n=n_points,
                                            sidelength=target_size)
    target_points[:, [0, 2]] = target_points[:, [2, 0]]

    field = panel.magnetic_field(target_points, current=current)

    fig, ax = plt.subplots(3, layout='constrained')

    plot_field_colormap(field, grid, axis, ax[0])

    plot_field_arrows(target_points, field, ax[2])

    panel.plot_profile(current, profile_dir='z',
                       field_component=axis, ax=ax[1],
                       min_pos=-(target_size/2),
                       max_pos=target_size/2)

    if title:
        fig.suptitle(title)

    if show:
        plt.show()
    # panel.plot(target_points, current) # 3D


def combined_panel_field(panels, currents, target_points):
    """Compute field from multiple sets of panels

    Parameters
    ----------
    panels : list of PCBPanels
        each panel to compute field for.
    currents : list of currents (dict of floats)
        currents to be applied to each panel.
    target_points : list of points (3d points ndarray)
        points where to compute fields.
    """
    field = 0.
    for panel, current in zip(panels, currents):
        field += panel.magnetic_field(target_points, current=current)
    return field


def plot_combined_panels(panels, currents, target_points, show=True):
    """Plots multiple panels and their combined fields in 3D

    Parameters
    ----------
    panels : list of PCBPanels
        each panel to be plotted and field to be computed.
    currents : list of currents (dict of floats)
        currents to be applied to each panel
    target_points : list of points (3d points ndarray)
        points where to compute fields
    """
    pl = pv.Plotter()
    for panel in panels:
        panel.plot(pl=pl, show=False)

    field = combined_panel_field(panels, currents, target_points)
    pl.add_arrows(target_points, field, mag=1e7)

    pl.remove_scalar_bar()
    if show:
        pl.show()
