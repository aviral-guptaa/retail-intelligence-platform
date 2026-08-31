"""Unit tests for geometry helpers."""
import numpy as np

from ml.geometry import direction_of_traversal, iou, point_in_polygon, segments_cross


def test_point_in_polygon_basic():
    poly = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert point_in_polygon([5, 5], poly)
    assert not point_in_polygon([15, 5], poly)
    assert not point_in_polygon([-1, 5], poly)


def test_point_in_polygon_concave():
    # L-shaped polygon: the notch (5,5) is OUTSIDE, right strip (17,8) is IN.
    poly = [[0, 0], [20, 0], [20, 12], [15, 12], [15, 4], [0, 4]]
    assert point_in_polygon([8, 2], poly)     # main rectangle
    assert point_in_polygon([17, 8], poly)    # right strip
    assert not point_in_polygon([5, 5], poly)  # the notch
    assert not point_in_polygon([17, 13], poly)  # above the shape


def test_segments_cross():
    assert segments_cross([0, 0], [10, 10], [0, 10], [10, 0])
    assert not segments_cross([0, 0], [10, 0], [0, 10], [10, 10])


def test_direction_of_traversal():
    line = ([0, 5], [20, 5])
    assert direction_of_traversal([5, 0], [5, 10], *line) == 1
    assert direction_of_traversal([5, 10], [5, 0], *line) == -1
    assert direction_of_traversal([5, 6], [5, 9], *line) == 0  # no crossing


def test_iou():
    a = [0, 0, 10, 10]
    b = [5, 0, 15, 10]
    assert 0.0 < iou(a, b) < 1.0
    assert iou(a, a) == 1.0
    assert iou(a, [100, 100, 110, 110]) == 0.0