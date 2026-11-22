#!/usr/bin/env python3
"""Apparent-gender estimator from a face image (OPTIONAL, opt-in).

    python programs/gender_detector.py --image face.jpg --accept-limits
    python programs/gender_detector.py --camera 0 --accept-limits

Two estimators:
  * CNN  - Levi & Hassner (2015) network trained on web photos. Works out of the box.
  * texture model - NumPy logistic regression on LBP / Gabor / edge / skin-colour features; only if you trained it
                    with programs/train_texture_model.py --task gender.

READ THIS: the output is a binary guess about visual *appearance*. It is error-prone (the author's own test misjudged a
photo of a woman as male at 97%), is less reliable for some groups, and says nothing about anyone's gender identity.
Do not use it to make decisions about people. You must pass --accept-limits to run it.
"""
import argparse

from _common import *      # noqa: F401,F403
from app.vision import attributes, faces, texture_ml


def analyse(frame):
    f = faces.detect_faces(frame)
    if not f:
        return None
    crop = faces.crop(frame, f[0]["box"])
    out = {"box": f[0]["box"], "cnn": attributes.predict_gender(crop)}
    if texture_ml.available("gender"):
        out["texture"] = texture_ml.predict_gender(crop)
    return out


def main():
    ap = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--accept-limits", action="store_true", help="confirm you have read the limitations above")
    a = ap.parse_args()
    if not a.accept_limits:
        ap.error("pass --accept-limits to confirm you understand this is an error-prone appearance guess (see --help)")
    ensure_models("gender_levi_hassner", "face_ssd_res10")
    kind, src = open_source(a)
    if kind == "image":
        r = analyse(src)
        if not r:
            print("No face found."); return
        g = r["cnn"]
        print(f"CNN: {g['label']}  ({g['probability']*100:.0f}%)   male={g['probs']['male-presenting']*100:.0f}%  female={g['probs']['female-presenting']*100:.0f}%")
        if "texture" in r:
            t = r["texture"]; print(f"texture model: {t['label']} ({t['probability']*100:.0f}%)  held-out accuracy {t['validation']['accuracy']*100:.0f}% vs baseline {t['validation']['baseline_accuracy']*100:.0f}%")
        print("NOTE:", g["caveat"])
        x, y, w, h = r["box"]; cv2.rectangle(src, (x, y), (x + w, y + h), (255, 200, 60), 3)
        put_label(src, g["label"] + f' {g["probability"]*100:.0f}%', (x, max(24, y - 6)))
        OUT.mkdir(exist_ok=True); cv2.imwrite(str(OUT / "gender_result.jpg"), src); print("Saved:", OUT / "gender_result.jpg")
        return
    disp, n = Display("Apparent gender (limits apply)", not a.no_gui), 0
    while True:
        ok, frame = src.read()
        if not ok:
            break
        n += 1
        r = analyse(frame) if n % 3 == 0 else None
        if r:
            x, y, w, h = r["box"]; cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 60), 3)
            put_label(frame, r["cnn"]["label"] + f' {r["cnn"]["probability"]*100:.0f}%', (x, max(24, y - 6)))
        put_label(frame, "appearance guess only - can be wrong  |  q = quit", (10, 28), (80, 200, 255), scale=0.5)
        if disp.show(frame, "gender_live") or (a.max_frames and n >= a.max_frames):
            break
    src.release(); disp.close()


if __name__ == "__main__":
    main()
