"""
=========================
01. Design biplanar coils
=========================

Example demonstrating how to create biplanar coils for production.
"""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Padma Sundaram <padma@nmr.mgh.harvard.edu>

# First, we will import the necessary libraries
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bfieldtools.utils import load_example_mesh

import opmcoils
from opmcoils import BiplanarCoil, get_sphere_points, get_target_field
from opmcoils.shielding import shielded_room

N_suh = 50
N_contours = 30  # Use N_contours = 30 for gradient_z
save = False

center = np.array([0, 0, 0])
target_type = 'dc_y'  # 'gradient_x' | 'gradient_y' | 'dc_x' | 'dc_y'

# %%
# Next we define the output directory containing the kicad files for our
# PCB design.

pcb_dir = Path(opmcoils.__path__[0]).parents[0]

output_dir = {'dc_x': 'Bx_coil',
              'dc_y': 'By_coil_dev',
              'dc_z': 'Bz_coil',
              'gradient_x': 'Gx_coil',
              'gradient_y': 'Gy_coil',
              'gradient_z': 'Gz_coil'}
header_type = {'dc_x': 'vert',
               'dc_y': 'horz',
               'dc_z': 'vert',
               'gradient_x': 'vert',
               'gradient_y': 'horz',
               'gradient_z': 'vert'}
bounds_wholeloop = {'dc_x': False,
                    'dc_y': True,
                    'dc_z': False, 
                    'gradient_x': False,
                    'gradient_y': True,
                    'gradient_z': False}

# %%
# Next we will define the parameters of our coils

standoffs = {"dc_y": 0.1400, "gradient_y": 0.1408,
             "dc_x": 0.1416, "gradient_x": 0.1424, 
             "dc_z": 0.1432, "gradient_z": 0.1440}

scaling = {"dc_y": 0.1400, "gradient_y": 0.1420,
           "dc_x": 0.1420, "gradient_x": 0.1436, 
           "dc_z": 0.1441, "gradient_z": 0.14565}

trace_width = 5.  # mm
cu_oz = 2.  # oz per ft^2

# %%
# A 10 m x 10 m biplanar coil mesh is loaded from bfieldtools.
# We will scale the mesh so as to achieve the dimensions of
# 1.4 m x 1.4 m that we will use in our work.

scaling_factor = scaling[target_type]
standoff = scaling_factor * 10

planemesh = load_example_mesh("10x10_plane_hires")
planemesh.apply_scale(scaling_factor)

# %%
# The BiplanarCoil class is instantiated
coil = BiplanarCoil(planemesh, center, N_suh=N_suh, standoff=standoff)

# %%
# Then the target points and the fields are used to fit the coil design
target_points, points_z = get_sphere_points(center, n=8, sidelength=0.5)
target_field = get_target_field(target_type, target_points)

coil.fit(target_points, target_field)

# %%
# Then, we can discretize the coil into current loops. At this point,
# we can also specify the trace width and the copper thickness used
# in the PCB design.
coil.discretize(N_contours=N_contours, trace_width=trace_width, cu_oz=cu_oz)

# %%
# To evaluate the effect of the shielded room, we can add it to the coil
# specification and it will be taken into account for estimating the
# magnetic field at any point
room_dims = (4, 2.3, 3.)
coil_pos = (1.89, 1.05, 1.6)
shield_mesh = shielded_room(room_dims=room_dims,
                            coil_pos=coil_pos)
coil.add_shield(shield_mesh)

# %%
# The field at some target points can be computed by doing
B_target = coil.predict(target_points)

# %%
# The field can be computed and plotted by doing
plotter = coil.plot_field(target_points)

# %%
# We can evaluate the coil for metrics such as efficiency
# and also compute its dimensions by doing
metrics = coil.evaluate(target_type, target_points, target_field,
                        points_z, 'all')
print(metrics)
print(f'The coil has dimensions {coil.shape} m')

# %%
# We can now interactively create paths to join the loops in the discretized coils
# by making "cuts". Uncomment below to use it.
# coil.make_cuts()

# %%
# Finally, we can export the files to KiCad
kicad_dir = Path.cwd().parent / 'hardware' / 'template' / 'headers'
pcb_dir = Path.cwd().parent / 'hardware'
if header_type[target_type] == 'vert':
    coil.save(
        pcb_fname=pcb_dir / f'{output_dir[target_type]}/first/coil_template_first.kicad_pcb',
        kicad_header_fname=kicad_dir / f'/kicad_header_{header_type[target_type]}_first_half.txt',
        bounds=(0, 750, 0, 1500), origin=(0, 750),
        bounds_wholeloop=bounds_wholeloop[target_type])

    coil.save(
        pcb_fname=pcb_dir / f'{output_dir[target_type]}/second/coil_template_second.kicad_pcb',
        kicad_header_fname=kicad_dir / f'kicad_header_{header_type[target_type]}_second_half.txt',
        bounds=(-750, 750, 0, 1500), origin=(750, 750),
        bounds_wholeloop=bounds_wholeloop[target_type])
else:
    coil.save(
        pcb_fname=pcb_dir / f'{output_dir[target_type]}/first/coil_template_first.kicad_pcb',
        kicad_header_fname=kicad_dir / f'kicad_header_{header_type[target_type]}_first_half.txt',
        bounds=(-750, 750, 0, 750), origin=(750, 0))

    coil.save(
        pcb_fname=pcb_dir / f'{output_dir[target_type]}/second/coil_template_second.kicad_pcb',
        kicad_header_fname=kicad_dir / f'kicad_header_{header_type[target_type]}_second_half.txt',
        bounds=(-750, 750, -750, 0), origin=(750, 750))
