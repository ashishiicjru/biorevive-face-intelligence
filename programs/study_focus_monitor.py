#!/usr/bin/env python3
"""Study-focus monitor: is the person at the screen paying visible attention? (expression + pose + eyes + phone)

    python programs/study_focus_monitor.py                 # webcam, until q / Esc
    python programs/study_focus_monitor.py --video lecture_cam.mp4 --no-gui
    python programs/study_focus_monitor.py --minutes 25    # stop automatically after 25 min

Cues fused by app/vision/engagement.py (transparent rules, NOT a validated measure of attention):
  head direction, eyes open / blinks / long closures, gaze, expression, phone in view, face present.

Landmark source
  * FULL mode  - MediaPipe Face Landmarker (pip install mediapipe; model via
                 `python -m app.download_models face_landmarker_mediapipe`): head pose, eye-aspect ratio, yawns, gaze.
  * BASIC mode - OpenCV only: SSD face + Haar eyes. No head pose, so the app can never report 'Focused' with
                 certainty and caps the score ('limited mode').
Use it on yourself, for self-tracking. Do not use it to grade or monitor other people.
"""
import argparse
import csv
import json
import time

from _common import *      # noqa: F401,F403
from app.vision import engagement, expression, faces, phone, pose

STATE_COL = {"Focused": (113, 211, 52), "Partially attentive": (36, 191, 251), "Distracted": (113, 113, 251),
             "Drowsy": (246, 92, 139), "Phone in use": (113, 113, 251), "Away": (160, 160, 160)}


class Landmarks:
    """MediaPipe Face Landmarker wrapper; .ok is False when mediapipe or its model file is unavailable."""

    def __init__(self):
        self.ok = False
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpt
            from mediapipe.tasks.python import vision
            model = mr.path_of("face_landmarker.task")
            if not model.exists():
                try:
                    mr.install("face_landmarker_mediapipe")
                except Exception as e:       # noqa: BLE001
                    raise RuntimeError(f"landmarker model not available ({e})")
            opts = vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=str(model)),
                                                running_mode=vision.RunningMode.VIDEO, num_faces=1)
            self.lm, self.mp, self.ok = vision.FaceLandmarker.create_from_options(opts), mp, True
        except Exception as e:      # noqa: BLE001
            print(f"[landmarks] BASIC mode ({str(e)[:110]}).")

    def detect(self, bgr, ts_ms):
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        r = self.lm.detect_for_video(img, int(ts_ms))
        return np.array([[p.x, p.y, p.z] for p in r.face_landmarks[0]]) if r.face_landmarks else None


_eyes = None


def haar_eyes_open(bgr, box):
    """BASIC-mode proxy: 0.30 if at least one eye is found in the upper face half, else 0.12 (eyes closed / looking down)."""
    global _eyes
    if _eyes is None:
        _eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    x, y, w, h = box
    roi = cv2.equalizeHist(cv2.cvtColor(bgr[y:y + int(h * 0.6), x:x + w], cv2.COLOR_BGR2GRAY))
    return 0.30 if len(_eyes.detectMultiScale(roi, 1.1, 5, minSize=(max(12, w // 10), max(12, w // 10)))) else 0.12


def main():
    ap = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--minutes", type=float, default=0, help="stop after N minutes (0 = manual)")
    ap.add_argument("--stride", type=int, default=2, help="analyse every Nth frame (default 2)")
    ap.add_argument("--no-phone", action="store_true", help="skip phone detection (faster)")
    ap.add_argument("--phone-size", type=int, default=416)
    ap.add_argument("--alert-after", type=float, default=10, help="beep after N s of distraction (0 = never)")
    a = ap.parse_args()
    if a.image:
        ap.error("this program needs --camera or --video (use expression_detector.py / phone_detector.py for single images)")
    ensure_models("expression_effnet_b0", "face_ssd_res10", *(() if a.no_phone else ("phone_yolov4_tiny",)))
    _, cap = open_source(a)
    lms = Landmarks()
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    video_clock = bool(a.video) and fps > 1
    study, disp = engagement.StudySession(), Display("Study focus monitor", not a.no_gui)
    last_exp, last_phone, res, n, t0, bad_since, last_alert = None, 0.0, None, 0, time.time(), None, 0
    print("Full mode (landmarks)" if lms.ok else "Basic mode (no landmarks)", "- press q / Esc to finish.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        t = (n / fps) if video_clock else (time.time() - t0)
        if a.minutes and t > a.minutes * 60:
            break
        if n % a.stride == 0:
            feats, box, present = None, None, False
            if lms.ok:
                lm = lms.detect(frame, t * 1000)
                if lm is not None:
                    present, box = True, faces.box_from_landmarks(lm, frame.shape[1], frame.shape[0])
                    feats = pose.features(lm, frame.shape[1], frame.shape[0])
            else:
                f = faces.detect_faces(frame)
                if f:
                    present, box = True, f[0]["box"]
                    feats = {"pose": None, "ear": haar_eyes_open(frame, box), "mar": None, "gaze_h": None}
            if present and (n // a.stride) % 3 == 0:
                last_exp = expression.predict(faces.crop(frame, box))
            if not a.no_phone and (n // a.stride) % 4 == 0:
                d = phone.detect(frame, 0.4, a.phone_size)
                last_phone = max([x["confidence"] for x in d if x["is_phone"]], default=0.0)
            res = study.update(t, present, pose=(feats or {}).get("pose"), ear=(feats or {}).get("ear"), mar=(feats or {}).get("mar"),
                               gaze_h=(feats or {}).get("gaze_h"), expression=last_exp if present else None, phone_conf=last_phone)
            res["box"], res["pose"] = box, (feats or {}).get("pose")
            if a.alert_after and res["state"] in ("Distracted", "Phone in use", "Drowsy", "Away"):
                bad_since = bad_since if bad_since is not None else t
                if t - bad_since > a.alert_after and t - last_alert > a.alert_after:
                    last_alert = t; print(f"[{t:6.1f}s] {res['state']} - back to work!"); beep()
            else:
                bad_since = None
        if res:
            col = STATE_COL.get(res["state"], (255, 255, 255))
            if res.get("box"):
                x, y, w, h = res["box"]; cv2.rectangle(frame, (x, y), (x + w, y + h), col, 3)
            put_label(frame, f'{res["state"]}  {res["score"]:.0f}/100', (10, 30), col, scale=0.8)
            p = res.get("pose")
            put_label(frame, (f'yaw {p["yaw"]:.0f} pitch {p["pitch"]:.0f}  ' if p else "") + f'blinks/min {res["blinks_per_min"]}  yawns {res["yawns"]}  '
                      + (f'expr {last_exp["label"]}' if last_exp else ""), (10, 62), (230, 230, 230), scale=0.5)
            if res["limited_mode"]:
                put_label(frame, "limited mode: no head/eye cues", (10, 92), (80, 200, 255), scale=0.5)
        if disp.show(frame, "study_live") or (a.max_frames and n >= a.max_frames):
            break
    cap.release(); disp.close()
    s = study.summary()
    OUT.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (OUT / f"study_session_{stamp}.json").write_text(json.dumps(s, indent=2))
    with open(OUT / f"study_session_{stamp}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["seconds", "score", "state"]); w.writerows(s["timeline"])
    print("\n=== Session summary ===")
    print(f"duration {s['duration_s']:.0f}s | average attention proxy {s['average_attention_score']:.0f}/100 | distraction events {s['distraction_events']}")
    print("time in state:", ", ".join(f"{k} {v:.0f}%" for k, v in s["state_percent"].items() if v))
    print(f"phone in view {s['phone_seconds']:.0f}s | yawns {s['yawns']} | longest focus streak {s['longest_focus_streak_s']:.0f}s")
    print(s["disclaimer"], f"\nSaved to outputs/study_session_{stamp}.json / .csv")


if __name__ == "__main__":
    main()
