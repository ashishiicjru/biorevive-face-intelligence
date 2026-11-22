"""Study-focus / attention estimation from visual cues.

A TRANSPARENT, RULE-BASED heuristic - not a trained or validated model of learning or attention.
Visible cues (face present, head direction, eyes open, gaze direction, phone in view, expression)
are combined into a 0-100 'attention proxy'. People concentrate while looking away, think with eyes
closed, and wear neutral faces; the score will be wrong for many of them. Use it as a self-tracking
aid, never to grade, rank or discipline anyone, and never as a medical or psychological measure.
"""
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Config:
    yaw_limit: float = 25.0           # deg off-centre before the head score starts to fall
    pitch_down_limit: float = 20.0    # looking down (phone / desk) penalised sooner than up
    pitch_up_limit: float = 25.0
    ear_open: float = 0.22
    ear_closed: float = 0.16
    gaze_center: tuple = (0.32, 0.68)
    away_after_s: float = 3.0
    eyes_closed_s: float = 1.5
    perclos_window_s: float = 20.0
    perclos_drowsy: float = 0.35
    phone_conf: float = 0.35
    phone_hold_s: float = 1.5
    yawn_mar: float = 0.55
    yawn_hold_s: float = 1.2
    ema_tau_s: float = 2.5
    focused_at: float = 70.0
    partial_at: float = 45.0
    weights: dict = field(default_factory=lambda: {"eyes": 0.45, "gaze": 0.35, "expression": 0.20})   # head acts as a gate
    phone_factor: float = 0.35


def _ramp(x, good, bad):
    """1 when x is on the 'good' side of `good`, 0 beyond `bad`, linear between."""
    if good == bad:
        return 1.0
    t = (x - bad) / (good - bad)
    return max(0.0, min(1.0, t))


class StudySession:
    STATES = ["Focused", "Partially attentive", "Distracted", "Drowsy", "Phone in use", "Away"]

    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.t0 = None
        self.last_t = None
        self.score = None
        self.state = "Away"
        self.time_in_state = {s: 0.0 for s in self.STATES}
        self.emotion_time = {}
        self.timeline = []                         # (t_rel, score, state) ~1 sample / 5 s
        self._last_tl = -999.0
        self._no_face_since = None
        self._closed_since = None
        self._phone_since = None
        self._phone_seconds = 0.0
        self._eye_hist = deque()                   # (t, closed:bool) for PERCLOS
        self._blinks = deque()                     # blink timestamps
        self._in_blink = False
        self._yawn_since = None
        self._in_yawn = False
        self.yawns = 0
        self.distraction_events = 0
        self._prev_state = None
        self._pending = 0.0                        # dropout seconds not yet attributed to a state
        self._streak = 0.0
        self.longest_focus_streak = 0.0
        self._scores = []

    # -------------------------------------------------------------- per frame
    def update(self, t, face_present, pose=None, ear=None, mar=None, gaze_h=None, expression=None, phone_conf=0.0):
        c = self.cfg
        if self.t0 is None:
            self.t0 = t
        dt = 0.0 if self.last_t is None else max(0.0, min(2.0, t - self.last_t))
        self.last_t = t
        phone = phone_conf >= c.phone_conf
        parts, used = {}, {}

        if not face_present:
            self._no_face_since = self._no_face_since if self._no_face_since is not None else t
            self._closed_since = None
            self._yawn_since = None
            frame_score = 0.0
        else:
            self._no_face_since = None
            if pose:
                yaw, pitch = abs(pose["yaw"]), pose["pitch"]
                yaw_s = _ramp(yaw, c.yaw_limit * 0.5, c.yaw_limit * 1.8)
                lim = c.pitch_down_limit if pitch > 0 else c.pitch_up_limit
                pit_s = _ramp(abs(pitch), lim * 0.5, lim * 1.8)
                parts["head"] = min(yaw_s, pit_s)
            if ear is not None:
                parts["eyes"] = _ramp(ear, c.ear_open, c.ear_closed)
            if gaze_h is not None:
                lo, hi = c.gaze_center
                off = 0.0 if lo <= gaze_h <= hi else min(abs(gaze_h - lo), abs(gaze_h - hi))
                parts["gaze"] = _ramp(off, 0.0, 0.25)
            if expression:
                neg = expression.get("negative_affect", 0.0)
                parts["expression"] = 1.0 - 0.6 * neg
            rest = {k: v for k, v in parts.items() if k != "head"}
            wsum = sum(c.weights[k] for k in rest)
            base = sum(rest[k] * c.weights[k] for k in rest) / wsum if wsum else 0.6   # nothing measurable: neutral-ish
            if "head" in parts:            # head direction is a GATE: looking away caps the score whatever else is true
                base *= 0.25 + 0.75 * parts["head"]
            frame_score = 100.0 * base * (c.phone_factor if phone else 1.0)
            if "head" not in parts:      # no head-pose evidence (no landmarks): can never claim 'Focused'
                frame_score = min(frame_score, c.focused_at - 0.1)
            used = {k: round(v, 2) for k, v in parts.items()}

            # eye closure, PERCLOS, blinks
            if ear is not None:
                closed = ear < c.ear_closed
                self._eye_hist.append((t, closed))
                while self._eye_hist and t - self._eye_hist[0][0] > c.perclos_window_s:
                    self._eye_hist.popleft()
                if closed:
                    self._closed_since = self._closed_since if self._closed_since is not None else t
                    self._in_blink = True
                else:
                    if self._in_blink and self._closed_since is not None and (t - self._closed_since) < 0.7:
                        self._blinks.append(t)
                    self._in_blink = False
                    self._closed_since = None
                while self._blinks and t - self._blinks[0] > 60:
                    self._blinks.popleft()
            # yawns
            if mar is not None and mar > c.yawn_mar:
                self._yawn_since = self._yawn_since if self._yawn_since is not None else t
                if not self._in_yawn and t - self._yawn_since >= c.yawn_hold_s:
                    self.yawns += 1
                    self._in_yawn = True
            else:
                self._yawn_since, self._in_yawn = None, False

        # phone persistence
        if phone and face_present:
            self._phone_since = self._phone_since if self._phone_since is not None else t
            self._phone_seconds += dt
        else:
            self._phone_since = None

        # smoothing (time-constant EMA)
        if self.score is None:
            self.score = frame_score
        else:
            a = 1 - 2.718281828 ** (-dt / c.ema_tau_s) if dt > 0 else 0.0
            self.score += a * (frame_score - self.score)

        # state machine (priority order)
        perclos = self.perclos()
        if self._no_face_since is not None and t - self._no_face_since >= c.away_after_s:
            state = "Away"
        elif not face_present:
            state = self.state if self.state != "Away" else "Away"      # brief dropout: hold previous state
        elif self._phone_since is not None and t - self._phone_since >= c.phone_hold_s:
            state = "Phone in use"
        elif (self._closed_since is not None and t - self._closed_since >= c.eyes_closed_s) or \
                (perclos is not None and perclos >= c.perclos_drowsy):
            state = "Drowsy"
        elif self.score >= c.focused_at:
            state = "Focused"
        elif self.score >= c.partial_at:
            state = "Partially attentive"
        else:
            state = "Distracted"
        if self._prev_state is not None and state != self._prev_state and state in ("Distracted", "Phone in use", "Away", "Drowsy"):
            self.distraction_events += 1
        self._prev_state, self.state = state, state

        if not face_present and state != "Away":       # brief dropout: hold the decision, attribute the time later
            self._pending += dt
        else:
            self.time_in_state[state] += dt + self._pending
            self._pending = 0.0
        self._scores.append((dt, self.score))
        if state == "Focused":
            self._streak += dt
            self.longest_focus_streak = max(self.longest_focus_streak, self._streak)
        else:
            self._streak = 0.0
        if expression and face_present:
            lab = expression.get("label")
            if lab:
                self.emotion_time[lab] = self.emotion_time.get(lab, 0.0) + dt
        rel = t - self.t0
        if rel - self._last_tl >= 5.0:
            self.timeline.append((round(rel, 1), round(self.score, 1), state))
            self._last_tl = rel
        return {"score": round(self.score, 1), "frame_score": round(frame_score, 1), "state": state, "components": used,
                "perclos": None if perclos is None else round(perclos, 3), "blinks_per_min": len(self._blinks),
                "yawns": self.yawns, "phone": phone, "limited_mode": face_present and "head" not in parts}

    def perclos(self):
        if len(self._eye_hist) < 10:
            return None
        return sum(1 for _, cl in self._eye_hist if cl) / len(self._eye_hist)

    # -------------------------------------------------------------- summary
    def summary(self):
        total = sum(self.time_in_state.values())
        pct = {k: round(100 * v / total, 1) if total else 0.0 for k, v in self.time_in_state.items()}
        wsum = sum(d for d, _ in self._scores)
        avg = sum(d * s for d, s in self._scores) / wsum if wsum else 0.0
        et = sum(self.emotion_time.values())
        return {"duration_s": round(total, 1), "average_attention_score": round(avg, 1), "state_percent": pct,
                "distraction_events": self.distraction_events, "phone_seconds": round(self._phone_seconds, 1),
                "yawns": self.yawns, "blinks_last_minute": len(self._blinks),
                "longest_focus_streak_s": round(self.longest_focus_streak, 1),
                "expression_percent": {k: round(100 * v / et, 1) for k, v in sorted(self.emotion_time.items(), key=lambda kv: -kv[1])} if et else {},
                "timeline": self.timeline,
                "disclaimer": "Heuristic attention proxy from visible cues; not a validated measure of learning or attention."}
