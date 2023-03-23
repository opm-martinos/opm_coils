import trimesh
import numpy as np
from shapely.geometry import Polygon

from bfieldtools.line_conductor import LineConductor

def _check_bounds(loop, bounds):
    if min(loop[:, 0]) < bounds[0]:
        return False
    if max(loop[:, 0]) > bounds[1]:
        return False
    if min(loop[:, 1]) < bounds[2]:
        return False
    if max(loop[:, 1]) > bounds[3]:
        return False
    return True


def python_to_kicad(
    loops, filename, plane_axes, origin, layer, net, scaling=1, trace_width=0.2
):
    """
    Appends polylines to KiCAD PCB files as traces.
    Paramaters
    ----------
        loops: N_loops long list of (Nv_loop, 3) arrays
            Each corresponding to the vertex locations for widning loops
        filename: str
            filename/path where the ifno is written (append mode is used)
        plane_axes: tuple with length 2
            specifies the X- and Y-dimensions used for the PCB
        origin: (2, ) array-like
            Origin shift applid in original units
        layer: str
            Which layer to write to
        scaling: float
            Scaling factor applied to the loops
        trace_width: float
            The trace width in mm
        net: int
            Specifies which net number the loops are assigned to
    Returns
    -------
        None
    """

    with open(filename, "w") as file:

        file.write("(kicad_pcb (version 4) (host pcbnew 4.0.6)\n")
        file.write("(page User 1200 1200)\n")
        file.write("(0 Bx signal)\n")
        for loop in loops:
            for seg_idx in range(1, len(loop)):
                x_start = loop[seg_idx - 1, plane_axes[0]] + origin[0]
                y_start = loop[seg_idx - 1, plane_axes[1]] + origin[1]

                x_end = loop[seg_idx, plane_axes[0]] + origin[0]
                y_end = loop[seg_idx, plane_axes[1]] + origin[1]

                file.write(
                    "    (segment (start %.2f %.4f) (end %.2f %.2f) (width %.2f) (layer %s) (net %d))\n"
                    % (
                        x_start * scaling,
                        y_start * scaling,
                        x_end * scaling,
                        y_end * scaling,
                        trace_width,
                        layer,
                        net,
                    )
                )

    return

def export_to_kicad(pcb_fname, kicad_header_fname, loops, origin=(600, 600),
                    net=1, scaling=1, trace_width=2., bounds=None,
                    bounds_wholeloop=True):

    print('export to kicad %s: \n' % (pcb_fname))

    kicad_header = open(kicad_header_fname, 'r')
    header = kicad_header.readlines()

    with open(pcb_fname, "w") as file:
        for line in header:
            file.write(line)

        for layer, layer_loops in loops.items():

            for loop in layer_loops:
                # print('loop\n')
                for seg_idx in range(1, len(loop)-1):
                    # print(seg_idx)
                    seg = loop[seg_idx]
                    x_start = seg[0] + origin[0]
                    y_start = seg[1] + origin[1]

                    seg = loop[seg_idx+1]
                    x_end = seg[0] + origin[0]
                    y_end = seg[1] + origin[1]

                    #print("[%5.2f %5.2f][%5.2f %5.2f] in [%5.2f %5.2f %5.2f %5.2f]?? \n" % (x_start, y_start,
                    #x_end, y_end, bounds[0],bounds[1], bounds[2], bounds[3]))

                    if bounds_wholeloop or _check_bounds(np.array([[x_start, y_start],[x_end, y_end], ]), bounds):
                        file.write(
                            "    (segment (start %.2f %.4f) (end %.2f %.2f) (width %.2f) (layer %s) (net %d))\n"
                            % (
                            x_start * scaling,
                            y_start * scaling,
                            x_end * scaling,
                            y_end * scaling,
                            trace_width,
                            layer,
                            net,
                            )
                        )
        file.write('\n)\n\n')

    print('done\n')



def loops_to_obj(fname, loops):
    """Generate circular polygon
    
    XXX: needs networkx and mapbox-earcut installed
    """
    tube_radius = 0.625 # radius for the resulting tubes
    n_components = 10 # number of the circle segments
    vec = np.array([0.0, 1.0]) * tube_radius
    angle = 2 * np.pi / n_components
    rotmat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    perim = []
    for _ in range(n_components):
        perim.append(vec)
        vec = np.dot(rotmat, vec)
    poly = Polygon(perim)

    scene = trimesh.Scene()
    for contour_num in range(len(loops.paths)):
        scene.add_geometry(trimesh.creation.sweep_polygon(polygon=poly, path=loops.discrete[contour_num]))
    trimesh.exchange.export.export_scene(scene, fname, file_type='obj', resolver=None)


def get_loop_colors(loops):
        origin=np.array([0, 0, 0])
        
        colors = []

        palette = [(1, 0, 0), (0, 0, 1)]
        for loop_idx, loop in enumerate(loops):

            # Compute each loop segment
            segments = np.vstack(
                (loop[1:, :] - loop[0:-1, :], loop[0, :] - loop[-1, :])
            )

            # Find mean normal vector following right-hand rule, in loop centre
            centre_normal = np.mean(np.cross(segments, loop), axis=0)
            centre_normal /= np.linalg.norm(centre_normal, axis=-1)

            # Check if normal "points in" or "out" (towards or away from origin)
            origin_vector = np.mean(loop, axis=0) - origin

            colors.append(
                palette[int((np.sign(centre_normal @ origin_vector) + 1) / 2)]
            )
            
        return colors
    
    
def python_to_rhino(fname, loops):
    fname_colors = "colors" + fname
    loopcolors = get_loop_colors(loops)
    with open(fname_colors, 'w') as fp_colors:
        with open(fname, 'w') as fp:
            for rhino_loopidx, rhino_loop in enumerate(loops):
                c = loopcolors[rhino_loopidx]
                c = np.multiply(c,255)
                for pt in rhino_loop:
                    fp.write(f'{pt[0]:.02f}, {pt[1]:.02f}, {pt[2]:.02f}\n') 
                    fp_colors.write(f'{c[0]:.02f}, {c[1]:.02f}, {c[2]:.02f}\n')
            
                fp.write('#\n')
                fp_colors.write('#\n')


def read_rhino_loops(fname):
    points = list()
    loops = list()
    with open(fname) as fp:
        pt_strs = fp.readlines()
        for pt_str in pt_strs:

            if pt_str[0] == '#':
                points.append(points[0]) # so that we get a closed curve
                loops.append(points)
                points = list()
                continue

            pt_str = pt_str.strip('\n').split(',')
            pt = tuple([float(coord) for coord in pt_str])
            points.append(pt)
    return loops


if __name__ == '__main__':

    line_conductor = kicad_to_loops('coil_template/coil_template.kicad_pcb')
    # line_conductor = kicad_to_loops('coil_template/coil_templateloopA.kicad_pcb')
    line_conductor.plot_loops()
    field = line_conductor.magnetic_field(points=np.array([0., 0., 0.,])[None, :])
