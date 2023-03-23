"""Metrics for evaluating coil designs."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Padma Sundaram <padma@nmr.mgh.harvard.edu>

import numpy as np

def _percent_error(biplanar_coil, target_field, target_points, target_type):
    field = biplanar_coil.predict(target_points)
    ax = {'x': 0, 'y': 1, 'z': 2}[target_type.split('_')[1]]
    return np.abs((field[:, ax] - target_field[:, ax]) / target_field[:, ax])

def homogeneity(biplanar_coil, target_field, target_points, target_type,
                allowed_error=0.05):
    """Compute homogeneity percentage with discretized current loops."""
    err = _percent_error(biplanar_coil, target_field, target_points, target_type)
    count = np.sum(err <= allowed_error)
    return count / target_field.shape[0] * 100

def efficiency(biplanar_coil, target_points, target_points_z, target_type):
    """Compute efficiency with discretized current loops."""
    ax = {'x': 0, 'y': 1, 'z': 2}[target_type.split('_')[1]]
    current_loops = biplanar_coil.line_conductor_
    mesh = biplanar_coil.remove_shield()

    if mesh is not None:
        field_nounits = biplanar_coil.predict(target_points)
        field_z_nounits = biplanar_coil.predict(target_points_z)

        biplanar_coil.add_shield(mesh)
        field_nounits_shield = biplanar_coil.predict(target_points)
        field_z_nounits_shield = biplanar_coil.predict(target_points_z)

        shield_factor = field_nounits_shield / field_nounits
        shield_factor_z = field_z_nounits_shield / field_z_nounits

    if 'dc' in target_type:
        field = current_loops.magnetic_field(points=target_points)
        if mesh is not None:
            field *= shield_factor
        # print(field * 1e-3 * 1e9)
        efficiency = np.mean(field * 1e-3 * 1e9, axis=0)[ax]
        unit = 'nT / mA'
    elif 'grad' in target_type:
        field = current_loops.magnetic_field(points=target_points_z)
        if mesh is not None:
            field *= shield_factor_z
        dB = (field[-1, ax] - field[0, ax]) * 1e-3 * 1e9 # nT
        dz = target_points_z[-1, 2] - target_points_z[0, 2]
        # add statement to assert dz remains constant
        efficiency = dB / dz  
        unit = 'nT / m / mA'
    
    return efficiency, unit

def error(biplanar_coil, target_field, target_points, target_type):
    """Compute error percentage with idealized current loops.

    coil :
        The coil object.
    target_field : array, shape (n_points, 3)
        The target field
    target_type : str
        'gradient_x' | 'gradient_y' | 'dc_x' | 'dc_y' etc.
    """
    return np.mean(
        _percent_error(biplanar_coil, target_field, target_points, target_type)) * 100
