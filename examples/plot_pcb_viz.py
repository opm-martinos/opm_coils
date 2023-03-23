"""
=================================================
05. Compute and visualize magnetic field from PCB
=================================================

Example demonstrating how to load the PCB from HDF5 files
which includes the connection information between wire segments
and how to visualize the field due to them.
"""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Gabriel Motta <gabrielbenmotta@gmail.com>

# sphinx_gallery_thumbnail_number = 7

from pathlib import Path
import h5io

from opmcoils.panels import PCBPanel, plot_panel

# %%
# Let us first load the nulling coils
hardware_folder = Path.cwd().parent / 'hardware' / 'hdf5'

panels = dict()
panels_data = h5io.read_hdf5(hardware_folder / 'panels.hdf5')
for key, pan in panels_data.items():
    panels[key] = PCBPanel(panel_dict=pan)

# %%
# Now we can plot each of the coils
current = dict(left=1e-3, right=1e-3)
for panel_name in panels:
    if 'G' in panel_name:
        current['right'] = 1e-3
    else:
        current['right'] = -1e-3

    axis = panel_name.strip('B').strip('G')
    if axis == 'z':
        current['right'] *= -1.

    print(f'Length and resistance of the panel {panel_name} is'
          f' {panels[panel_name].length:.2f} m and '
          f'{panels[panel_name].resistance():.2f} Ohm')
    plot_panel(panels[panel_name], .7, 32, current=current, axis=axis,
               title=f'{panel_name} Panels')

# %%
# We can also compute the combined field due to several nulling coils
from opmcoils import get_sphere_points
from opmcoils.panels import combined_panel_field

target_points, _ = get_sphere_points([0, 0, 0], n=8, sidelength=0.5)
panels = [panels['By'], panels['Bx'], panels['Bz']]
currents = [dict(left=1e-3, right=-1e-3),
            dict(left=-1e-3, right=1e-3),
            dict(left=.2e-3, right=.2e-3)]
field = combined_panel_field(panels, currents, target_points)
print(field)

# %%
# Finally, we can plot the combined panels and their resulting field
from opmcoils.panels import plot_combined_panels

plot_combined_panels(panels, currents, target_points)
