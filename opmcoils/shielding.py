import numpy as np

from trimesh.util import concatenate
from bfieldtools.utils import load_example_mesh


def _swap(mesh, ax1, ax2):
    
    temp = mesh.vertices[:, ax1].copy()
    mesh.vertices[:, ax1] = mesh.vertices[:, ax2]
    mesh.vertices[:, ax2] = temp
    return mesh

def _shield(location='front', shift=None, invert_normals=False,
            scale=(1, 1, 1)):

    shield = load_example_mesh("10x10_plane") # hires? too slow ...
    shield.apply_scale(0.1)  # 1 m
    shield.apply_scale(scale)

    if location in ('front', 'back'):
        shield = _swap(shield, 0, 1)
    elif location in ('left', 'right'):
        shield = _swap(shield, 1, 2)

    if shift is not None:
        shield.vertices += np.array(shift) 

    if invert_normals:
        shield.invert()

    return shield

def shielded_room(room_dims=(4, 2.3, 2.9), coil_pos=(1.89, 0.86, 1.73)):
    """Construct a shielded room and place coil inside it.

    Parameters
    ----------
    room_dims : tuple (x, y, z)
        The room dimensions along x, y, and z coordinates.
    coil_pos : tuple (x, y, z)
        The coil position along x, y, and z coordinates.

    Notes
    -----
    Coordinate system is below
           +--------------+
          /|             /|
         / |    Top     / |  
        +--------------+  |  
        |  |    Left   |  | Front 
        |  y           |  |  
        |  |           |  |
        |  O----x---------+
        | /            | /   
        |z             |/    
        +--------------+   
    """

    scales = {
        'front': (room_dims[1], 1, room_dims[2]),
        'back': (room_dims[1], 1, room_dims[2]),
        'left': (room_dims[0], 1, room_dims[1]),
        'right': (room_dims[0], 1, room_dims[1]),
        'top': (room_dims[0], 1, room_dims[2]),
        'bottom': (room_dims[0], 1, room_dims[2])
    }
    shield_pos = {
        'front': (room_dims[0], room_dims[1] / 2., room_dims[2] / 2.),
        'back': (0, room_dims[1] / 2., room_dims[2] / 2.),
        'left': (room_dims[0] / 2., room_dims[1] / 2., 0.),
        'right': (room_dims[0] / 2., room_dims[1] / 2., room_dims[2]),
        'top': (room_dims[0] / 2., room_dims[1], room_dims[2] / 2.),
        'bottom': (room_dims[0] / 2., 0., room_dims[2] / 2.)
    }
    invert_normals = {'front': True, 'back': False, 'top': False,
                      'bottom': True, 'left': False, 'right': True}

    shields = list()
    for name, pos in shield_pos.items():
        shield_pos[name] = np.array(pos) - np.array(coil_pos)
        shields.append(
            _shield(name, shield_pos[name], invert_normals[name], scale=scales[name])
            )

    total_shield = shields[0].copy()
    for this_shield in shields[1:]:
        total_shield = concatenate(total_shield, this_shield)

    return total_shield
