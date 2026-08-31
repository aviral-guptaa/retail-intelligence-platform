"""Movement / occupancy heatmap generation."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from app.schemas.models import Track


class HeatmapAccumulator:
    """Accumulates tracked person positions into a coarse occupancy grid.

    A box around each person's center is added to the grid each frame, giving a
    smooth "time spent here" intensity map. :meth:`to_image` normalises the
    grid and applies the OpenCV JET colormap so the dashboard can display it
    with a single PNG decode, regardless of who rendered it.

    An optional per-frame ``decay`` factor (0 < decay < 1) bounds long-running
    captures: each update scales the grid down (e.g. 0.995) before adding the
    new positions, so old dwells fade instead of growing forever. With decay=0
    the grid is cumulative (fine for demos and short runs).
    """

    def __init__(self, width: int, height: int, scale: int = 4,
                 camera_id: str = "store_01", decay: float = 0.0):
        self.scale = max(1, int(scale))
        self.camera_id = camera_id
        self.decay = float(decay)
        self.width = width
        self.height = height
        self.grid = np.zeros((height // self.scale + 1, width // self.scale + 1),
                             dtype=np.float32)

    def update(self, tracks: List[Track]) -> None:
        if self.decay > 0:
            self.grid *= self.decay
        for tr in tracks:
            cx, cy = tr.center
            gx = int(cx // self.scale)
            gy = int(cy // self.scale)
            # Soft 5x5 gaussian-ish kernel around the person's cell.
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    xx, yy = gx + dx, gy + dy
                    if 0 <= xx < self.grid.shape[1] and 0 <= yy < self.grid.shape[0]:
                        weight = 1.0 if dx == 0 and dy == 0 else 0.45
                        self.grid[yy, xx] += weight

    def to_image(self, normalize: bool = True) -> np.ndarray:
        """Return an upscaled 3-channel colormap image (H x W x 3, uint8)."""
        grid = self.grid
        if grid.max() > 0 and normalize:
            grid = grid / grid.max()
        heat = (grid * 255).astype(np.uint8)
        try:
            import cv2

            colored = cv2.applyColorMap(cv2.resize(heat, (self.width, self.height),
                                                   interpolation=cv2.INTER_LINEAR),
                                        cv2.COLORMAP_JET)
            return colored
        except ImportError:
            # No-op fallback so analytics still run without opencv installed.
            return np.stack([heat] * 3, axis=-1)

    def save(self, path: str | Path) -> bool:
        """Write the PNG heatmap to ``path`` (parent dirs created). Returns success."""
        try:
            import cv2

            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".png", self.to_image())
            if ok:
                path.write_bytes(buf.tobytes())
                return True
        except Exception:
            return False
        return False

    def reset(self) -> None:
        self.grid.fill(0.0)