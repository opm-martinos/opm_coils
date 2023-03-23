"""Code to make the PCB design from discretized current loops."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Padma Sundaram <padma@nmr.mgh.harvard.edu>

import numpy as np
import matplotlib.pyplot as plt


def winding_number(point, polygon):
    """Courtsey ChatGPT."""
    wn = 0
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        if y1 <= point[1]:
            if y2 > point[1] and ccw(point, (x1, y1), (x2, y2)):
                wn += 1
        else:
            if y2 <= point[1] and not ccw(point, (x1, y1), (x2, y2)):
                wn -= 1

    return wn

def ccw(A, B, C):
    """Are A, B, and C counterclockwise?"""
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def intersect(A, B, C, D):
    """Does AB and CD intersect?"""
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def get_intersection_segment(loops, line_cut):
    """https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/"""
    loop_idxs, segment_idxs = list(), list()
    cut_segments, cut_pts = list(), list()
    for loop_idx, loop in enumerate(loops):
        line_segs = zip(loop, loop[1:] + [loop[0]])
        for segment_idx, line_seg in enumerate(line_segs):
            if intersect(line_seg[0], line_seg[1], line_cut[0], line_cut[1]):
                cut_pts.append(get_intersection_pt(line_cut, line_seg))
                cut_segments.append(line_seg)
                loop_idxs.append(loop_idx)
                segment_idxs.append(segment_idx)
    return loop_idxs, segment_idxs, cut_segments, np.array(cut_pts)


def get_intersection_pt(line1, line2):
    """https://stackoverflow.com/a/20677983"""
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def _do_reverse(loops, loop_idxs, line_cut):
    """If first point is inside any of the loops, reverse it so it is outside."""
    reverse = False
    for loop_idx in loop_idxs:
        if winding_number(line_cut[0], loops[loop_idx]) != 0:
            return True
    return reverse

def join_loops_at_cuts(loops, line_cut, line_cut_shifted, colors):
    """Join loops into one continuous path"""

    # join loops at cut
    loop_idxs, segment_idxs, segments, cut_pts = get_intersection_segment(
        loops, line_cut)

    # force line cuts to be "out to in".
    if _do_reverse(loops, loop_idxs, line_cut):
        line_cut_shifted, line_cut = line_cut[::-1], line_cut_shifted[::-1]
        loop_idxs, segment_idxs, segments, cut_pts = get_intersection_segment(
            loops, line_cut)
    loop_idxs2, segment_idxs2, segments2, cut_pts2 = get_intersection_segment(
        loops, line_cut_shifted)

    assert len(loop_idxs) == len(loop_idxs2)

    if colors[loop_idxs[0]] == (0, 0, 1):
        direction = 'c'
    else:
        direction = 'cc'

    continuous_loop = list()
    reverse_paths = list()
    for idx, (loop_idx, segment_idx1, segment_idx2) in enumerate(zip(loop_idxs, segment_idxs, segment_idxs2)):
        this_loop = loops[loop_idx]
        if direction == 'cc':
            loop_pts = this_loop[segment_idx2 + 1:] + this_loop[:segment_idx1]
        elif direction == 'c':
            loop_pts = this_loop[segment_idx2::-1] + this_loop[:segment_idx1:-1]
        for pt in loop_pts:
            continuous_loop.append(pt)

        # diagonal
        # XXX: could be simplified but works for now
        continuous_loop.append(cut_pts[idx])
        reverse_paths.append(cut_pts[idx])
        if idx + 1 < len(cut_pts2):
            continuous_loop.append(cut_pts2[idx + 1])
            reverse_paths.append(cut_pts2[idx + 1])

    # XXX: don't ask why
    if direction == 'c':
        reverse_paths.insert(0, cut_pts2[0])
        reverse_paths.insert(0, loops[loop_idxs[0]][segment_idxs2[0]])
    else:
        reverse_paths.insert(0, loops[loop_idxs[0]][segment_idxs2[0]])
        reverse_paths.insert(0, loops[loop_idxs[0]][segment_idxs2[0] + 1])

    return np.array(continuous_loop), np.array(reverse_paths), segments, cut_pts, cut_pts2, direction
