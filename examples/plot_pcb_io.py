"""
=====================
04. Loading PCB files
=====================

Here, we will demonstrate how to load PCB files and
save them to HDF5 format after inferring the connections
between the wire segments.
"""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Gabriel Motta <gabrielbenmotta@gmail.com>

# %%
# First, we will import the necessary libraries
from pathlib import Path
import h5io

from opmcoils.panels import load_panel, plot_panel

# %%
# Then, we define the necessary folders
hardware_folder = Path.cwd().parent / 'hardware'
pcb_folder = hardware_folder / 'By_coil'
panels = dict()
current = dict(left=1e-3, right=-1e-3)

# %%
# To load a file, we simply do
panel = load_panel(pcb_folder, flip=['left_bott', 'right_bott'])

# %%
# This may take a while ... it is because KiCAD does not store
# the order of individual segments of copper wire. We have to connect
# the segments together based on their positions.

panels['By'] = panel

# %%
# We can repeat this process for Bx coil
# In this case, we have to flip the direction of the connected segments
# in two PCBs in order to obtain the correct orientation.

pcb_folder = hardware_folder / 'Bx_coil'
panel = load_panel(pcb_folder, standoff=1.416,
                   flip=['left_second', 'right_second'])

#
# We can verify that the panel was loaded correctly by plotting the field
# due to it. This function plots a 2d colormap of the field along the
# x-z plane, a line profile along the z axis, and an arrow field map.
plot_panel(panel, .7, 32, current=current, axis='x', title='Bx Panels')
panels['Bx'] = panel

# %%
# For Bz coil, we have to manually correct the direction of
# the connected segment at each solder joint.
pcb_folder = hardware_folder / 'Bz_coil'
flip = dict(left_first=[3, 1, 4, 6, 0, 5, 25, 12, 17, 20],
            left_second=[20, 5, 13, 30, 29, 2, 14, 6, 15, 9, 4, 23, 21, 18])
flip['right_first'] = flip['left_first']
flip['right_second'] = flip['left_second']
panel = load_panel(pcb_folder, standoff=1.432, flip=flip)
panels['Bz'] = panel

# %%
# We repeat this process for the Gy coil
pcb_folder = hardware_folder / 'Gy_coil'
current = dict(left=1e-3, right=1e-3)

panel = load_panel(pcb_folder, standoff=1.408, rearrange=True,
                   flip=['left_second', 'right_second'])
panels['Gy'] = panel

# %%
# The Gx coil
pcb_folder = hardware_folder / 'Gx_coil'
panel = load_panel(pcb_folder, standoff=1.424)
panels['Gx'] = panel

# %%
# and the Gz coil.
pcb_folder = hardware_folder / 'Gz_coil'
current = dict(left=1e-3, right=-1e-3)
flip = dict(left_first=[9, 11, 13, 12, 15, 2, 16, 5],
            left_second=[5, 3, 10, 15, 7, 6, 0, 11, 17])
flip['right_first'] = flip['left_first']
flip['right_second'] = flip['left_second']
panel = load_panel(pcb_folder, standoff=1.440,
                   flip=flip)
panels['Gz'] = panel

# %%
# Finally, we can save these files to HDF5 to avoid
# recomputing the inference of segment order.
panels_dict = dict()
for key, panel in panels.items():
    panels_dict[key] = panel.to_dict()

h5io.write_hdf5('panels.hdf5', panels_dict, overwrite=True)
