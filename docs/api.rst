:orphan:

API
===

Design and manufacture nulling coils
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimize stream functions, discretize into current
loops, join interactively, and export to KiCAD.

.. currentmodule:: opmcoils

.. autosummary::
   :toctree: generated/

    BiplanarCoil
    get_sphere_points
    get_target_field

Evaluation metrics
^^^^^^^^^^^^^^^^^^

Evaluate performance using efficiency and homogeneity.

.. currentmodule:: opmcoils.metrics

.. autosummary::
    :toctree: generated/

    efficiency
    homogeneity

Simulating PCBs
^^^^^^^^^^^^^^^

Load the completed coil path from KiCAD and compute the
total magnetic field from a combination of PCBs.

.. currentmodule:: opmcoils.panels

.. autosummary::
    :toctree: generated/

    PCB
    PCBPanel
    load_panel
    combined_panel_field
    plot_field_colormap
    plot_panel_profile
    plot_panel
    plot_combined_panels

Data Analysis
^^^^^^^^^^^^^

Analyze the performance of a coil design.

.. currentmodule:: opmcoils.analysis

.. autosummary::
    :toctree: generated/

    get_good_chs
    load_remnant_fields
    add_ch_loc
    read_opm_info
    find_events
