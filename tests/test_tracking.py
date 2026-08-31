"""Unit tests for the tracking + line counter (entry/exit)."""
from app.schemas.models import Detection
from ml.shopper.line_counter import LineCounter
from ml.tracking.tracker import Tracker


def _det(x, y, w=30, h=60, cls=0, name="person"):
    return Detection(x - w / 2, y - h, x + w / 2, y, 0.9, cls, name)


def test_tracker_assigns_stable_ids_and_recycles():
    tr = Tracker({"max_age_frames": 5, "match_threshold": 0.2}, "store_01")
    ids = [t.id for t in tr.update([_det(10, 10), _det(200, 200)])]
    same = tr.update([_det(11, 11), _det(201, 201)])  # same people, slightly moved
    assert [t.id for t in same] == ids
    # Occupy one slot for a while, the other track disappears -> no id collision
    tr.update([_det(11, 11)])
    tr.update([_det(11, 11)])
    tr.update([_det(11, 11)])
    tr.update([_det(11, 11)])
    new = tr.update([_det(11, 11), _det(300, 300)])
    assert new[0].id == ids[0]


def test_tracker_ignores_non_person_classes():
    tr = Tracker({}, "store_01")
    out = tr.update([_det(10, 10, cls=999, name="product")])
    assert out == [] or all(t.class_name == "person" for t in out)


def test_line_counter_counts_entries_and_exits():
    lc = LineCounter(line_start=(0, 50), line_end=(100, 50), camera_id="store_01")
    # Person enters (downward) then exits (upward).
    t1 = _det(50, 20); t1.track_id = 1
    events = lc.update([t1])
    assert lc.entries == 0
    t2 = _det(50, 80); t2.track_id = 1
    events += lc.update([t2])
    assert lc.entries == 1 and lc.exits == 0
    assert any(e.event_type == "entry" for e in events)
    t3 = _det(50, 10); t3.track_id = 1
    events += lc.update([t3])
    assert lc.entries == 1 and lc.exits == 1
    assert any(e.event_type == "exit" for e in events)


def test_line_counter_no_duplicate_on_parallel_motion():
    lc = LineCounter((0, 50), (100, 50), "store_01")
    prev = _det(10, 60); prev.track_id = 7
    cur = _det(90, 60); cur.track_id = 7   # never crosses the line
    lc.update([prev])
    lc.update([cur])
    assert (lc.entries, lc.exits) == (0, 0)