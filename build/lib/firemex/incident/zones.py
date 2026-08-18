"""Exclusion zones.

Per-camera polygons whose interiors are ignored. This is the cheapest and most
effective false-positive control available: the sunset through a west window, the
stove top, the welding bay, the monitor showing a fire, the designated smoking
area. All are static, and all are better excluded geometrically than argued with
by the model.
"""

from __future__ import annotations

from ..detect.base import BBox

Point = tuple[float, float]
Polygon = list[Point]


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Ray-casting point-in-polygon test on normalised coordinates."""
    x, y = point
    inside = False
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]
        # Does the horizontal ray at y cross this edge?
        if (y1 > y) != (y2 > y):
            t = (y - y1) / (y2 - y1)
            if x < x1 + t * (x2 - x1):
                inside = not inside
    return inside


def polygon_bounds(polygon: Polygon) -> BBox:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return BBox(min(xs), min(ys), max(xs), max(ys))


def box_overlap_fraction(box: BBox, polygon: Polygon, samples: int = 7) -> float:
    """Approximate the fraction of ``box`` lying inside ``polygon``.

    Sampled on a grid rather than computed exactly: an exact polygon clip would be
    more code and more edge cases for a number only ever compared against a
    coarse threshold.
    """
    if box.area <= 0:
        return 0.0
    if not _bounds_intersect(box, polygon_bounds(polygon)):
        return 0.0
    inside = 0
    total = samples * samples
    for i in range(samples):
        # Sample cell centres so edges are not double-counted.
        x = box.x1 + box.width * (i + 0.5) / samples
        for j in range(samples):
            y = box.y1 + box.height * (j + 0.5) / samples
            if point_in_polygon((x, y), polygon):
                inside += 1
    return inside / total


def _bounds_intersect(a: BBox, b: BBox) -> bool:
    return not (a.x2 < b.x1 or b.x2 < a.x1 or a.y2 < b.y1 or b.y2 < a.y1)


class ZoneMask:
    """The set of exclusion zones for one camera."""

    def __init__(self, zones: list[Polygon] | None = None, overlap_threshold: float = 0.6) -> None:
        self.zones = [list(zone) for zone in (zones or []) if len(zone) >= 3]
        #: A detection is excluded when this fraction of its box falls inside a zone.
        #: Fraction rather than centroid-only, so a fire that merely clips the corner
        #: of an excluded window is still reported.
        self.overlap_threshold = overlap_threshold

    def __bool__(self) -> bool:
        return bool(self.zones)

    def excludes(self, box: BBox) -> bool:
        if not self.zones:
            return False
        return any(
            box_overlap_fraction(box, zone) >= self.overlap_threshold for zone in self.zones
        )
