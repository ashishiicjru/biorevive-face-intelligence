"""Shared helpers for the standalone command-line programs."""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app import model_registry as mr  # noqa: E402


def ensure_models(*names):
    """Make sure the given registry models are installed (downloads them if not)."""
    for n in names:
        if not mr.is_installed(n):
            print(f"[models] '{n}' not installed - downloading ...")
            try:
                mr.install(n, lambda d, t: print(f"\r  {d/1e6:6.1f} MB", end="", flush=True))
                print()
            except Exception as e:      # noqa: BLE001
                sys.exit(f"Could not install model '{n}': {e}\nRun: python -m app.download_models {n}")


def open_source(args):
    """-> (kind, capture-or-image). Exactly one of --image / --video / --camera."""
    if getattr(args, "image", None):
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"Cannot read image: {args.image}")
        return "image", img
    if getattr(args, "video", None):
        cap = cv2.VideoCapture(args.video)
    else:
        idx = int(getattr(args, "camera", 0))
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
    if not cap.isOpened():
        sys.exit("Cannot open the video source (check the camera index / permissions).")
    return "stream", cap


class Display:
    """cv2.imshow when a GUI build of OpenCV is installed, otherwise writes annotated frames to outputs/."""

    def __init__(self, title, gui=True, every=30):
        self.title, self.every, self.n, self.gui = title, every, 0, gui
        if gui:
            try:
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            except cv2.error:
                print("[display] This OpenCV build has no GUI (opencv-python-headless).\n"
                      "          Install the GUI build:  pip uninstall -y opencv-python-headless && pip install opencv-python\n"
                      "          Falling back to saving frames in the outputs/ folder.")
                self.gui = False

    def show(self, frame, tag="frame"):
        self.n += 1
        if self.gui:
            cv2.imshow(self.title, frame)
            return (cv2.waitKey(1) & 0xFF) in (27, ord("q"))          # Esc or q quits
        if self.n % self.every == 1:
            OUT.mkdir(exist_ok=True)
            cv2.imwrite(str(OUT / f"{tag}.jpg"), frame)
        return False

    def close(self):
        if self.gui:
            cv2.destroyAllWindows()


def put_label(img, text, org, color=(255, 255, 255), bg=(20, 24, 36), scale=0.6):
    (w, h), b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = org
    cv2.rectangle(img, (x, y - h - b - 4), (x + w + 8, y + 4), bg, -1)
    cv2.putText(img, text, (x + 4, y - b), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def beep():
    try:
        if os.name == "nt":
            import winsound
            winsound.Beep(900, 180)
        else:
            print("\a", end="", flush=True)
    except Exception:       # noqa: BLE001
        pass


def common_args(ap):
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--image", help="analyse one image file")
    g.add_argument("--video", help="analyse a video file")
    g.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--no-gui", action="store_true", help="never open a window; save annotated frames to outputs/")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = until you press q / Esc)")
    return ap
