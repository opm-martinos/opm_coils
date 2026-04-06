from .base_coil import BaseCoil
from .biplanar_coil import (BiplanarCoil, get_sphere_points, get_target_field,
                            get_2D_point_grid)
from .cylindrical_coil import CylindricalCoil, flatten_loops
from .panels import (PCBPanel, plot_field_colormap, plot_field_arrows,
                     load_panel, plot_panel, check_half_names)

__version__ = '0.1.dev0'
