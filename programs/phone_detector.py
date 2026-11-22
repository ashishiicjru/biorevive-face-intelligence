#!/usr/bin/env python3
"""Mobile-phone detector (YOLOv4-tiny, COCO 'cell phone'): live camera, video or image.

    python programs/phone_detector.py                       # webcam, beeps + logs when a phone is seen
    python programs/phone_detector.py --image room.jpg --size 1024
    python programs/phone_detector.py --video clip.mp4 --no-gui --save-frames

A phone must be seen in --confirm consecutive frames (default 2) before an alert fires. Detections go to
outputs/phone_log.csv. Larger --size finds smaller phones but is slower; 'remote' is a common look-alike.
"""
import argparse
import csv
import time

from _common import *      # noqa: F401,F403
from app.vision import phone


def draw(frame, dets):
    for d in dets:
        x, y, w, h = d["box"]
        col = (80, 80, 255) if d["is_phone"] else (60, 190, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), col, 3)
        put_label(frame, f'{d["label"]} {d["confidence"]*100:.0f}%', (x, max(24, y - 6)), col)
    return frame


def main():
    ap = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--conf", type=float, default=0.4, help="minimum confidence (default 0.4)")
    ap.add_argument("--size", type=int, default=608, help="detector input size 320-1024 (default 608)")
    ap.add_argument("--confirm", type=int, default=2, help="consecutive frames needed before alerting")
    ap.add_argument("--all-objects", action="store_true", help="show every COCO object, not just phones")
    ap.add_argument("--save-frames", action="store_true", help="save a JPEG whenever a phone alert fires")
    a = ap.parse_args()
    ensure_models("phone_yolov4_tiny")
    kind, src = open_source(a)
    OUT.mkdir(exist_ok=True)
    if kind == "image":
        dets = phone.detect(src, a.conf, a.size, all_objects=a.all_objects)
        cv2.imwrite(str(OUT / "phone_result.jpg"), draw(src, dets))
        print("PHONE DETECTED" if phone.phone_present(dets, a.conf) else "No phone detected")
        for d in dets:
            print(f"  {d['label']:<12} {d['confidence']*100:5.1f}%  box={d['box']}")
        print("Saved:", OUT / "phone_result.jpg")
        return
    log = open(OUT / "phone_log.csv", "a", newline="")
    wr = csv.writer(log)
    if log.tell() == 0:
        wr.writerow(["timestamp", "confidence", "x", "y", "w", "h"])
    disp, streak, active, n, t0 = Display("Phone detector", not a.no_gui), 0, False, 0, time.time()
    while True:
        ok, frame = src.read()
        if not ok:
            break
        n += 1
        dets = phone.detect(frame, a.conf, a.size, all_objects=a.all_objects)
        ph = [d for d in dets if d["is_phone"]]
        streak = streak + 1 if ph else 0
        if streak >= a.confirm and not active:
            active = True
            best = max(ph, key=lambda d: d["confidence"])
            print(f"[{time.strftime('%H:%M:%S')}] PHONE DETECTED  {best['confidence']*100:.0f}%")
            wr.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), best["confidence"], *best["box"]]); log.flush()
            beep()
            if a.save_frames:
                cv2.imwrite(str(OUT / f"phone_{int(time.time())}.jpg"), draw(frame.copy(), dets))
        elif not ph:
            active = False
        frame = draw(frame, dets)
        if active:
            put_label(frame, "PHONE DETECTED", (10, 60), (255, 255, 255), (40, 40, 200), 0.9)
        put_label(frame, f"{n/(time.time()-t0):.1f} fps  |  q = quit", (10, 28), (200, 230, 255))
        if disp.show(frame, "phone_live") or (a.max_frames and n >= a.max_frames):
            break
    src.release(); disp.close(); log.close()


if __name__ == "__main__":
    main()
