"""
Example demonstrating how to load Kicad PCBs and evaluate their performance.
"""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Padma Sundaram <padma@nmr.mgh.harvard.edu>
import os
import sys
import numpy as np
import argparse

from opmcoils import get_sphere_points, PCBPanel
from opmcoils.panels import flip_chain


def check_half_names(pcb_folder):
    halves = ['top', 'bott', 'first', 'second']
    use_names = list()

    for half in halves:
        if os.path.isdir(os.path.join(pcb_folder, half)):
            use_names.append(half)
    return use_names

def get_boards():
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    prod_path = os.path.join(script_dir, "../production/")
    develop_path = os.path.join(script_dir, "../development/")

    subfolders = [f.path for f in os.scandir(prod_path) if f.is_dir()]
    subfolders += [f.path for f in os.scandir(develop_path) if f.is_dir()]

    for i, s in enumerate(subfolders):
        print(f'{i:2} - {s.split("../")[1]}')

    selection = input("Select a PCB: ")
    selection = int(selection)

    if selection < 0 or selection >= len(subfolders):
        return "", list()

    pcb_folder = subfolders[selection]

    use_names = check_half_names(pcb_folder)

    print(f'Loading from {pcb_folder}')
    print(f'Found {use_names}')

    return pcb_folder, use_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pcb_path', nargs='*',
                        help='Path to pcb project files.')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Toggle interactive mode. On by default when no'
                        'pcb path provided.')
    parser.add_argument('-r', '--rearrange', action='store_true',
                        help='Treat first/second as top/bottom')

    args = parser.parse_args()

    pcb_folder = ''
    interactive = False
    if args.interactive or len(args.pcb_path) == 0:
        interactive = True

    standoff = 1.4
    current = dict(left=1e-3, right=1e-3)

    if interactive:
        pcb_folder, half_names = get_boards()
    elif os.path.exists(args.pcb_path[0]):
        half_names = check_half_names(args.pcb_path[0])
        if half_names:
            pcb_folder = args.pcb_path[0]

    rearrange = False
    if interactive and 'first' in half_names:
        print("Treat 'first' and 'second' as 'top' and 'bott'? (y/n)", end='')
        rearrange = input()
        rearrange = True if rearrange in ['y', 'Y'] else False
    if not interactive and args.rearrange:
        rearrange = True

    if not pcb_folder:
        sys.exit(1)

    print()

    center = np.array([0, 0, 0])
    # target_points, points_z = get_sphere_points(center, n=8, sidelength=0.5)
    target_points, points_z = get_between_points(center, n=16, sidelength=1)

    # x and z coordinates must be swapped from
    # Brookes to "world" coordinate frame
    target_points[:, [0, 2]] = target_points[:, [2, 0]]
    panel = PCBPanel(pcb_folder, half_names=half_names,
                     standoff=standoff, rearrange=rearrange)
    # field = panel.magnetic_field(target_points, current=current)

    pcb_plot = panel.plot(target_points, current)

    if interactive:
        print('Flip coils?')
        options = [key for key in panel.chains.keys()]
        for i, key in enumerate(options):
            print(f'{i:2} - {key}')
        flip = input('csv of which chains to flip, leave empty to flip none: ')
        flip = [int(c.strip()) for c in flip.split(',') if c]

        for num in flip:
            if num > -1 and num < len(options):
                flip_chain(panel.chains[options[num]])
            else:
                print(f'Invalid index {num}. Skipping.')
        if len(flip) > 0:
            pcb_plot = panel.plot(target_points, current)

    # Let's do a line profile
    panel.plot_profile(current, profile_dir='z', field_component='y')
