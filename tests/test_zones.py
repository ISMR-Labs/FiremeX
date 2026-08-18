"""Exclusion zone geometry."""

from __future__ import annotations

from firemex.detect.base import BBox
from firemex.incident.zones import ZoneMask, box_overlap_fraction, point_in_polygon

SQUARE = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]
TRIANGLE = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]


def test_point_inside_and_outside_square():
    assert point_in_polygon((0.25, 0.25), SQUARE)
    assert not point_in_polygon((0.75, 0.25), SQUARE)
    assert not point_in_polygon((0.25, 0.75), SQUARE)


def test_point_in_triangle():
    assert point_in_polygon((0.1, 0.1), TRIANGLE)
    # Beyond the hypotenuse.
    assert not point_in_polygon((0.9, 0.9), TRIANGLE)


def test_concave_polygon_handles_notch():
    # An L shape: the notch in the top-right must read as outside.
    l_shape = [
        (0.0, 0.0), (0.6, 0.0), (0.6, 0.2), (0.2, 0.2), (0.2, 0.6), (0.0, 0.6),
    ]
    assert point_in_polygon((0.1, 0.1), l_shape)
    assert point_in_polygon((0.4, 0.1), l_shape)
    assert not point_in_polygon((0.4, 0.4), l_shape)


def test_overlap_fraction_full_and_none():
    assert box_overlap_fraction(BBox(0.1, 0.1, 0.3, 0.3), SQUARE) == 1.0
    assert box_overlap_fraction(BBox(0.7, 0.7, 0.9, 0.9), SQUARE) == 0.0


def test_overlap_fraction_partial_is_between():
    fraction = box_overlap_fraction(BBox(0.4, 0.4, 0.6, 0.6), SQUARE)
    assert 0.1 < fraction < 0.6


def test_mask_excludes_only_when_mostly_inside():
    mask = ZoneMask([SQUARE], overlap_threshold=0.6)
    # A fire deep inside the excluded window.
    assert mask.excludes(BBox(0.1, 0.1, 0.3, 0.3))
    # A real fire that merely clips the window's corner must still be reported --
    # this is the failure mode that gets people killed by an over-eager mask.
    assert not mask.excludes(BBox(0.45, 0.45, 0.95, 0.95))


def test_empty_mask_excludes_nothing():
    mask = ZoneMask([])
    assert not mask
    assert not mask.excludes(BBox(0.1, 0.1, 0.2, 0.2))


def test_degenerate_zones_are_dropped():
    mask = ZoneMask([[(0.0, 0.0), (1.0, 1.0)]])
    assert not mask.zones
