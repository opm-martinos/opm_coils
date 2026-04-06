"""
==============================
02. Design cylindrical coils
==============================

Example demonstrating how to create a cylindrical nulling coil for production.
The coil wraps around a cylindrical surface and can be unrolled into a flat
flex PCB.
"""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>

import numpy as np

from bfieldtools.utils import load_example_mesh

from opmcoils import CylindricalCoil, get_sphere_points, get_target_field

# %%
# Define coil parameters. The cylindrical mesh from bfieldtools has a default
# radius of ~1 m and height of ~1 m. We scale it to the desired physical size.

N_suh = 400       # number of surface harmonic basis functions
N_contours = 3    # number of discrete current loops to extract
trace_width = 5.  # mm
cu_oz = 2.        # oz per ft^2
target_type = 'dc_z'

center = np.array([0, 0, 0])

# Radius and height scales (in meters)
radius_scale = 0.15   # 15 cm radius
height_scale = 0.25   # 25 cm height

# %%
# Load the cylindrical mesh from bfieldtools and scale it.

mesh = load_example_mesh('open_cylinder')
# The default mesh has radius ~1 and height ~1; apply independent scaling
mesh.vertices[:, 0] *= radius_scale   # x
mesh.vertices[:, 1] *= radius_scale   # y
mesh.vertices[:, 2] *= height_scale   # z

# %%
# Instantiate the CylindricalCoil.
coil = CylindricalCoil(mesh, center, N_suh=N_suh)

# %%
# Define target points inside the cylinder and the desired field.
# We use a spherical grid of points centred at the origin.
target_points, points_z = get_sphere_points(center, n=8, sidelength=0.1)
target_field = get_target_field(target_type, target_points)

# %%
# Fit the coil by optimizing the current distribution on the cylindrical
# surface to produce the target field.
coil.fit(target_points, target_field)

# %%
# Discretize the continuous stream function into N_contours current loops.
coil.discretize(N_contours=N_contours, trace_width=trace_width, cu_oz=cu_oz)
coil.plot_coil()

# %%
# Predict the field at the target points and plot it.
B_target = coil.predict(target_points)
coil.plot_field(target_points)

# %%
# Evaluate coil performance metrics.
metrics = coil.evaluate(target_type, target_points, target_field,
                        points_z, metrics='all')
print(metrics)

# %%
# Interactively draw cut lines to join the discrete loops into a continuous
# PCB trace. Uncomment to use.
# coil.make_cuts()

# %%
# Export to KiCad. Uncomment and set paths as needed.
# from pathlib import Path
# kicad_dir = Path('hardware') / 'template' / 'headers'
# coil.save(
#     pcb_fname='cylindrical_coil.kicad_pcb',
#     kicad_header_fname=kicad_dir / 'kicad_header_vert_first_half.txt',
# )
