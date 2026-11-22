#!/usr/bin/env python3
"""Facial-expression detector (happy / sad / angry / surprised / ...), live camera, video or image.

    python programs/expression_detector.py                    # webcam
    python programs/expression_detector.py --image photo.jpg --tiles   # group photo
    python programs/expression_detector.py --video clip.mp4 --no-gui

Model: EfficientNet-B0 trained on AffectNet (8 classes). It reads the *visible expression*, not feelings -
a serious neutral face is often scored "anger". Press q / Esc to quit.
"""
import argparse
import json
import time

from _common import *      # noqa: F401,F403  (cv2, np, OUT, helpers)
from app.vision import expression, faces

COLORS = {"Happiness": (113, 211, 52), "Neutral": (196, 163, 143), "Sadness": (255, 140, 79), "Anger": (133, 113, 251),
          "Surprise": (36, 191, 251), "Fear": (246, 92, 139), "Disgust": (21, 230, 163), "Contempt": (182, 114, 244)}


def annotate(frame, results):
    for r in results:
        x, y, w, h = r["box"]
        e = r["expression"]
        col = COLORS.get(e["label"], (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), col, 3)
        put_label(frame, f'{e["label"]} {e["confidence"]*100:.0f}%', (x, max(24, y - 6)), col)
        for i, (k, v) in enumerate(sorted(e["probs"].items(), key=lambda kv: -kv[1])[:4]):      # mini bar chart
            by = y + h + 16 + i * 16
            if by < frame.shape[0] - 4:
                cv2.rectangle(frame, (x, by - 9), (x + int(v * 120), by - 2), COLORS.get(k, (200, 200, 200)), -1)
                cv2.putText(frame, f"{k[:5]} {v*100:.0f}%", (x + 126, by), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    return frame


def analyse(frame, tiles, state):
    out = []
    for i, f in enumerate(faces.detect_faces(frame, tiles=tiles)[:12]):
        e = expression.predict(faces.crop(frame, f["box"]))
        if i == 0:                                   # smooth the main face over time (live video)
            sm = expression.smooth(state.get("p"), e["probs"])
            state["p"] = sm
            top = max(sm, key=sm.get)
            e = dict(e, label=top, confidence=sm[top], probs=sm)
        out.append({"box": f["box"], "expression": e})
    return out


def main():
    ap = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--tiles", action="store_true", help="tiled face search for group photos (finds small faces)")
    ap.add_argument("--json", help="write results to this JSON file (image mode)")
    a = ap.parse_args()
    ensure_models("expression_effnet_b0", "face_ssd_res10")
    kind, src = open_source(a)
    if kind == "image":
        res = analyse(src, True if a.tiles else False, {})
        OUT.mkdir(exist_ok=True)
        cv2.imwrite(str(OUT / "expression_result.jpg"), annotate(src, res))
        for i, r in enumerate(res, 1):
            e = r["expression"]; print(f"face {i}: {e['label']:<10} {e['confidence']*100:5.1f}%   " + "  ".join(f"{k}={v*100:.0f}%" for k, v in sorted(e['probs'].items(), key=lambda kv: -kv[1])[:3]))
        if not res:
            print("No face found.")
        if a.json:
            Path(a.json).write_text(json.dumps([{"box": r["box"], **r["expression"]} for r in res], indent=2, default=float))
        print("Saved:", OUT / "expression_result.jpg")
        return
    disp, state, n, t0 = Display("Expression detector", not a.no_gui), {}, 0, time.time()
    while True:
        ok, frame = src.read()
        if not ok:
            break
        n += 1
        frame = annotate(frame, analyse(frame, a.tiles, state))
        put_label(frame, f"{n/(time.time()-t0):.1f} fps  |  q = quit", (10, 28), (200, 230, 255))
        if disp.show(frame, "expression_live") or (a.max_frames and n >= a.max_frames):
            break
    src.release(); disp.close()


if __name__ == "__main__":
    main()
