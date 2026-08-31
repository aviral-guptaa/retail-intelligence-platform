"""Zone analytics: occupancy and dwell time per configurable polygon zone."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.schemas.models import Track, ZoneEvent
from ml.geometry import point_in_polygon


class ZoneDwellTracker:
    """Tracks which polygons shoppers are inside and how long they stay.

    For each (track, zone) pair we record the entry timestamp once the person's
    center enters the polygon and clear it when the center leaves. Zone dwell
    histograms let us report average visit length and current occupancy - the
    input to footfall-by-zone and heatmap aggregation.

    A visit that is *still in progress* contributes to ``current_dwell()``
    (how long the current occupants have already been in the zone) but not to
    the completed-visit average.
    """

    def __init__(self, zones: Dict[str, List[np.ndarray]], camera_id: str):
        self.zones = zones
        self.camera_id = camera_id
        self._tracks_in_zone: Dict[Tuple[int, str], float] = {}
        self.dwell_total: Dict[str, float] = defaultdict(float)
        self.visits: Dict[str, int] = defaultdict(int)

    def update(self, tracks: List[Track], now: Optional[float] = None) -> List[ZoneEvent]:
        events: List[ZoneEvent] = []
        now = time.time() if now is None else now
        active_keys = set()

        for tr in tracks:
            center = tr.center
            for zname, poly in self.zones.items():
                inside = point_in_polygon(center, poly)
                key = (tr.id, zname)
                if inside and key not in self._tracks_in_zone:
                    self._tracks_in_zone[key] = now
                    events.append(ZoneEvent(now, self.camera_id, "zone_enter", 1.0,
                                            {"zone": zname, "track_id": tr.id}))
                if inside:
                    active_keys.add(key)

        # Close out dwell intervals for tracks that left a zone or vanished.
        for key, start in list(self._tracks_in_zone.items()):
            if key not in active_keys:
                dwell = now - start
                _, zname = key
                self.dwell_total[zname] += dwell
                self.visits[zname] += 1
                events.append(ZoneEvent(now, self.camera_id, "zone_exit",
                                        round(dwell, 2),
                                        {"zone": zname, "track_id": key[0]}))
                del self._tracks_in_zone[key]

        return events

    def occupancy(self) -> Dict[str, int]:
        occ: Dict[str, int] = defaultdict(int)
        for (_, zname) in self._tracks_in_zone:
            occ[zname] += 1
        return dict(occ)

    def current_dwell(self, now: Optional[float] = None) -> Dict[str, float]:
        """Seconds occupied shoppers have already spent in each zone (in progress)."""
        now = time.time() if now is None else now
        current: Dict[str, float] = defaultdict(float)
        latest: Dict[str, float] = {}
        for (_, zname), start in self._tracks_in_zone.items():
            dwell = now - start
            latest[zname] = max(latest.get(zname, 0.0), dwell)
            current[zname] += dwell
        # Report the dwell of the newest arrival as "current wait at the back"
        # for queue zones; aggregate is the sum across occupants.
        return {z: round(d, 2) for z, d in current.items()}

    def avg_dwell(self) -> Dict[str, float]:
        out = {}
        for zname in self.zones:
            v = self.visits[zname]
            out[zname] = round(self.dwell_total[zname] / v, 2) if v else 0.0
        return out

    def reset(self) -> None:
        self._tracks_in_zone.clear()
        self.dwell_total.clear()
        self.visits.clear()