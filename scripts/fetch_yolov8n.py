"""Best-effort download of YOLOv8n weights (used only during the ML Docker build)."""
import pathlib
import urllib.request

URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
DEST = pathlib.Path("models/yolo/yolov8n.pt")


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size > 0:
        print("yolov8n already present:", DEST.stat().st_size)
        return
    try:
        urllib.request.urlretrieve(URL, DEST)
        print("yolov8n downloaded:", DEST.stat().st_size)
    except Exception as exc:  # keep the build alive even if download fails
        print("yolov8n skipped:", exc)
        DEST.write_bytes(b"")


if __name__ == "__main__":
    main()