from typing import Any, Callable, Dict, List, Optional

import numpy as np


def point_in_polygon(point, polygon) -> bool:
    """Ray-casting point-in-polygon test (works for convex and concave shapes).

    Args:
        point: (x, y) coordinates.
        polygon: sequence of (x, y) vertices (closed implicitly).
    """
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def cross_2d(a, b) -> float:
    return a[0] * b[1] - a[1] * b[0]


def side_of_line(point, line_start, line_end) -> float:
    """Signed distance of a point from the infinite line through the two endpoints."""
    return cross_2d(
        np.asarray(line_end, dtype=float) - np.asarray(line_start, dtype=float),
        np.asarray(point, dtype=float) - np.asarray(line_start, dtype=float),
    )


def _straddles(a: float, b: float) -> bool:
    """True when a and b lie on opposite sides of a line (touch counts)."""
    return (a < 0 < b) or (b < 0 < a) or a == 0 or b == 0


def segments_cross(p1, p2, q1, q2) -> bool:
    """True if segment p1-p2 intersects segment q1-q2 (touching counts)."""
    return _straddles(side_of_line(q1, p1, p2), side_of_line(q2, p1, p2)) and \
           _straddles(side_of_line(p1, q1, q2), side_of_line(p2, q1, q2))


def direction_of_traversal(p_prev, p_curr, line_start, line_end) -> int:
    """Signed traversal across a line.

    Returns +1 when the point crossed from the *negative* half-plane to the
    *positive* half-plane, -1 for the reverse, and 0 when no clean crossing
    happened. The caller maps this +/-1 onto its own convention (for the store
    entrance, entering the floor is defined as crossing toward the interior).
    """
    s1 = side_of_line(p_prev, line_start, line_end)
    s2 = side_of_line(p_curr, line_start, line_end)
    if s1 == 0 or s2 == 0:
        return 0            # ambiguous / resting on the line: skip
    if (s1 > 0) != (s2 > 0):
        return 1 if s1 < 0 and s2 > 0 else -1
    return 0


def iou(box_a, box_b) -> float:
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def lerp(polygon) -> List[np.ndarray]:
    return [np.asarray(v, dtype=float) for v in polygon]