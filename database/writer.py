"""Background DB writer: decouples the analytics loop from disk.

Synchronous per-frame commits (the old design) were both slow and a crash risk.
This writer drains a bounded queue on a daemon thread and commits in batches on
a cadence. If the database is unavailable it logs and *drops* - never blocks
or crashes the analytics loop.

The pipeline submits plain dicts via :meth:`submit`; the worker maps them onto
the ORM models in :mod:`database.models`.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional

from database.models import AlertRecord, AnalyticsSnapshot, TrajectorySample

logger = logging.getLogger(__name__)


class BackgroundWriter:
    """Threaded ORM writer with bounded queue + batched commits."""

    _KINDS = {"snapshot": AnalyticsSnapshot, "alert": AlertRecord,
              "position": TrajectorySample}

    def __init__(self, session_factory: Optional[Callable] = None,
                 flush_interval: float = 2.0, max_queue: int = 10000):
        self._session_factory = session_factory
        self.flush_interval = float(flush_interval)
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=max_queue)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.dropped = 0
        self.flushed_rows = 0
        if session_factory is not None:
            self.start()

    # ---------------------------------------------------------------- public
    def start(self) -> None:
        if self._session_factory is None or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._drain_loop, name="db-writer", daemon=True)
        self._thread.start()

    def submit(self, kind: str, **data: Any) -> bool:
        """Enqueue one row. Returns True if accepted, False if dropped."""
        model = self._KINDS.get(kind)
        if model is None or not self._running:
            return False
        try:
            self._queue.put_nowait((model, data))
            return True
        except queue.Full:
            self.dropped += 1
            logger.warning("db writer queue full - dropping %s row", kind)
            return False

    def flush_now(self, timeout: float = 2.0) -> int:
        """Signal a flush and wait briefly for the batch to be committed."""
        if not self._running:
            return 0
        try:
            self._queue.put_nowait(("__flush__", None))
        except queue.Full:
            return 0
        # Wait for the worker to communicate completion is hard; give the
        # thread a moment to drain then best-effort return.
        deadline = time.time() + timeout
        while time.time() < deadline and not self._queue.empty():
            time.sleep(0.01)
        return self.flushed_rows

    def shutdown(self, flush_timeout: float = 3.0) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=flush_timeout)
        self._queue = queue.Queue()

    # ---------------------------------------------------------------- worker
    def _drain_loop(self) -> None:
        session = self._session_factory()
        flush_at = time.time() + self.flush_interval
        # Drain everything that was enqueued even after _running flips False,
        # so shutdown never loses rows that were already accepted.
        while self._running or not self._queue.empty():
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                try:
                    item = self._queue.get(timeout=min(self.flush_interval, 0.2))
                except queue.Empty:
                    item = None
            if item is None:
                # idle tick or shutdown flush point
                self._commit(session)
                flush_at = time.time() + self.flush_interval
                continue
            model, data = item
            if model == "__flush__":
                self._commit(session)
                flush_at = time.time() + self.flush_interval
                continue
            try:
                session.add(model(**data))
                self.flushed_rows += 1
                if time.time() >= flush_at or self.flushed_rows % 200 == 0:
                    self._commit(session)
                    flush_at = time.time() + self.flush_interval
            except Exception as exc:
                logger.warning("db writer row failed (%s): %s",
                               getattr(model, "__name__", str(model)), exc)
                try:
                    session.rollback()
                except Exception:
                    pass
        try:
            self._commit(session)
        finally:
            try:
                session.close()
            except Exception:
                pass

    @staticmethod
    def _commit(session) -> None:
        # Only touch the DB when there is something to write.
        if not list(session.new):
            return
        try:
            session.commit()
        except Exception as exc:
            logger.warning("db batch commit failed: %s", exc)
            session.rollback()