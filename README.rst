OPM Nulling coil design
=======================

This repository hosts Python code for design of biplanar coils
for nulling uniform fields for operation of Optically Pumped Magnetometers (OPMs)

.. image:: https://lh3.googleusercontent.com/pw/AP1GczNo5RmF9yM8JIiJWSJs2JLgEsq52t4XrnajXLujju7_w4nYH4li78IKHDQ7WdW07ZL-YmwFBjf1iQoRs7qZybG934ZlZJEHqGk7tWnNaGSsNNbqE5jP8Ep9joX6Y5HqyYR4W8_rat4B2AX7jIVr01Hksg=w1914-h1576-s-no
   :width: 500

Installation
^^^^^^^^^^^^

We recommend creating a virtual environment. For example, using Anaconda,
we can create a virtual environment using the following command::

    $ conda create -n opmcoils -c conda-forge python=3.9

Then, inside the environment::

    $ conda activate opmcoils

Install ``opmcoils`` by doing::

    $ pip install .

Advanced users or contributors may add the ``-e`` flag to create an editable install.

Usage
^^^^^

Please check the examples/ directory

License
^^^^^^^

The software is distributed under BSD 3-Clause and the hardware is distributed
under CERN-OHL-S (strongly reciprocal) licenses. Please include the appropriate
license when re-distributing any parts of the hardware/software.

Citation
^^^^^^^^

If you use the data, scripts, or hardware designs from this repository, please
cite::

    Jas M., Kamataris J., Matsubara T., Dong C., Motta G.,
    Sohrabpour A., Ahlfors S.P., Hamalainen M., Okada Y., and Sundaram P..
    Biplanar nulling coil system for OPM-MEG using printed circuit boards.
    2024.

Funding
^^^^^^^

This work was supported by NIH grants P41EB030006, 1R21NS140619-01, 2R01NS104585-05,
1R01NS112183-01A1, and S10OD030469.


.. toctree::
   :hidden:

   Example gallery<auto_examples/index>
   API<api>
   PCB design files<https://github.com/opm-martinos/opm_coils/tree/main/hardware>
   Coil assembly<assembly>
