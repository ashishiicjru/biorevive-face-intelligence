"""Face texture + skin-tone analysis with classical computer vision (no ML, fully transparent).

Everything here is a *measurement of image statistics* - local binary patterns, Gabor energy,
edge / fine-line density, smoothness, Lab skin colour. They depend heavily on camera, lighting,
makeup, compression and resolution. They are descriptive cues for a prototype, NOT a medical or
dermatological assessment and NOT a validated measure of age or health.
"""
import base64

import cv2
import numpy as np

# Regions as fractions of the face crop (x0, y0, x1, y1)
REGIONS = {"forehead": (0.22, 0.06, 0.78, 0.26), "eye_zone": (0.12, 0.28, 0.88, 0.44),
           "left_cheek": (0.10, 0.48, 0.38, 0.72), "right_cheek": (0.62, 0.48, 0.90, 0.72),
           "mouth_zone": (0.28, 0.70, 0.72, 0.88)}
FEATURE_NAMES = None          # filled below


def _prep(face_bgr, size=192):
    """Centred square crop (no stretching - stretching would distort texture) resized to size x size."""
    h, w = face_bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(face_bgr[y0:y0 + s, x0:x0 + s], (size, size), interpolation=cv2.INTER_AREA)


def skin_mask(face_bgr):
    """Skin pixels: YCrCb range inside a face ellipse, cleaned morphologically. Returns uint8 0/255."""
    ycc = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycc, (0, 133, 77), (255, 173, 127))
    h, w = m.shape
    ell = np.zeros_like(m)
    cv2.ellipse(ell, (w // 2, int(h * 0.52)), (int(w * 0.40), int(h * 0.46)), 0, 0, 360, 255, -1)
    m = cv2.bitwise_and(m, ell)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    # fallback for unusual colour casts: use the ellipse core if too little skin was found
    if m.mean() / 255.0 < 0.15:
        core = np.zeros_like(m)
        cv2.ellipse(core, (w // 2, int(h * 0.55)), (int(w * 0.28), int(h * 0.30)), 0, 0, 360, 255, -1)
        m = core
    return m


_UNI = None


def _uniform_lut():
    """Map the 256 LBP codes (P=8) to 10 'uniform' classes (rotation-invariant uniform patterns)."""
    global _UNI
    if _UNI is None:
        lut = np.zeros(256, np.uint8)
        for c in range(256):
            bits = [(c >> i) & 1 for i in range(8)]
            trans = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
            lut[c] = sum(bits) if trans <= 2 else 9
        _UNI = lut
    return _UNI


def lbp_image(gray):
    """8-neighbour, radius-1 LBP code image (uint8) for the interior pixels."""
    g = gray.astype(np.int16)
    c = g[1:-1, 1:-1]
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    code = np.zeros_like(c, dtype=np.uint8)
    for i, (dy, dx) in enumerate(shifts):
        nb = g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
        code |= ((nb >= c).astype(np.uint8) << i)
    return code


def lbp_histogram(gray, mask=None):
    code = _uniform_lut()[lbp_image(gray)]
    if mask is not None:
        sel = mask[1:-1, 1:-1] > 0
        code = code[sel]
    hist = np.bincount(code.ravel(), minlength=10).astype(np.float64)
    return hist / max(1.0, hist.sum())


def gabor_energy(gray, mask=None):
    """Mean |response| for 4 orientations x 2 wavelengths -> (8,) plus per-filter std -> (8,)."""
    g = gray.astype(np.float32) / 255.0
    feats_m, feats_s = [], []
    for lam in (6.0, 12.0):
        for th in np.arange(0, np.pi, np.pi / 4):
            k = cv2.getGaborKernel((21, 21), 4.0, th, lam, 0.5, 0, ktype=cv2.CV_32F)
            r = np.abs(cv2.filter2D(g, cv2.CV_32F, k))
            v = r[mask > 0] if mask is not None and (mask > 0).any() else r.ravel()
            feats_m.append(float(v.mean()))
            feats_s.append(float(v.std()))
    return np.array(feats_m), np.array(feats_s)


def _region_slice(img, frac):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = frac
    return slice(int(y0 * h), int(y1 * h)), slice(int(x0 * w), int(x1 * w))


def region_metrics(gray, mask):
    """Edge (fine-line) density and Laplacian variance per facial region, inside skin pixels only."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 110)
    lap = cv2.Laplacian(blur, cv2.CV_32F)
    out = {}
    for name, frac in REGIONS.items():
        ys, xs = _region_slice(gray, frac)
        m = mask[ys, xs] > 0
        if m.sum() < 30:
            out[name] = {"edge_density": 0.0, "roughness": 0.0, "skin_coverage": round(float(m.mean()), 3)}
            continue
        out[name] = {"edge_density": round(float((edges[ys, xs][m] > 0).mean()), 4),
                     "roughness": round(float(lap[ys, xs][m].var()), 2),
                     "skin_coverage": round(float(m.mean()), 3)}
    return out, edges


def skin_tone(face_bgr, mask):
    """Lab statistics of skin pixels + Individual Typology Angle (ITA).
    ITA = atan((L*-50)/b*) in degrees. Category bands follow Chardon et al. (1991). It is NOT a
    Fitzpatrick skin type and shifts with white balance, exposure and makeup."""
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    sel = mask > 0
    if sel.sum() < 50:
        return None
    L = lab[..., 0][sel] * 100.0 / 255.0
    a = lab[..., 1][sel] - 128.0
    b = lab[..., 2][sel] - 128.0
    Lm, am, bm = float(np.median(L)), float(np.median(a)), float(np.median(b))
    ita = float(np.degrees(np.arctan2(Lm - 50.0, max(bm, 1e-3))))
    bands = [(55, "very light"), (41, "light"), (28, "intermediate"), (10, "tan"), (-30, "brown")]
    cat = next((n for t, n in bands if ita > t), "dark")
    bgr = np.median(face_bgr[sel], axis=0)
    return {"lab": {"L": round(Lm, 1), "a": round(am, 1), "b": round(bm, 1)}, "ita_degrees": round(ita, 1),
            "ita_category": cat, "redness_a": round(am, 1), "evenness": round(float(100 - min(100, L.std() * 4)), 1),
            "swatch_hex": "#%02x%02x%02x" % (int(bgr[2]), int(bgr[1]), int(bgr[0])),
            "note": "Colour depends on lighting and camera white balance; not a Fitzpatrick type or medical measure."}


def smoothness_index(gray, mask):
    """0-100, higher = smoother skin (inverse of high-frequency energy in skin pixels)."""
    hp = gray.astype(np.float32) - cv2.GaussianBlur(gray, (0, 0), 2.0).astype(np.float32)
    v = hp[mask > 0]
    if v.size < 50:
        return None
    return round(float(max(0.0, 100.0 - min(100.0, v.std() * 4.0))), 1)


def feature_vector(face_bgr):
    """Fixed-length numeric vector for the trainable texture models (see texture_ml.py)."""
    f = _prep(face_bgr)
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    mask = skin_mask(f)
    lbp = lbp_histogram(gray, mask)
    gm, gs = gabor_energy(gray, mask)
    reg, _ = region_metrics(gray, mask)
    r = np.array([[v["edge_density"], np.log1p(v["roughness"])] for v in reg.values()]).ravel()
    tone = skin_tone(f, mask)
    t = np.array([tone["lab"]["L"], tone["lab"]["a"], tone["lab"]["b"], tone["evenness"]]) if tone else np.zeros(4)
    sm = smoothness_index(gray, mask) or 0.0
    return np.concatenate([lbp, gm, gs, r, t, [sm]]).astype(np.float64)


FEATURE_DIM = 10 + 8 + 8 + 2 * len(REGIONS) + 4 + 1


def _png_b64(img):
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def analyze(face_bgr, with_images=True):
    """Full texture report for one face crop. Returns a JSON-able dict (images as data-URLs)."""
    f = _prep(face_bgr)
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    mask = skin_mask(f)
    lbp = lbp_histogram(gray, mask)
    ent = float(-(lbp[lbp > 0] * np.log2(lbp[lbp > 0])).sum())
    gm, gs = gabor_energy(gray, mask)
    reg, edges = region_metrics(gray, mask)
    tone = skin_tone(f, mask)
    smooth = smoothness_index(gray, mask)
    # relative, un-calibrated 0-100 index from edge density (more fine lines/pores => higher)
    ed = np.mean([v["edge_density"] for v in reg.values()])
    index = round(float(min(100.0, ed * 200.0)), 1)
    warn = []
    if min(face_bgr.shape[:2]) < 110:
        warn.append("Low-resolution face (<110 px): texture metrics are unreliable. Use a closer, sharper photo.")
    if (mask > 0).mean() < 0.3:
        warn.append("Little skin detected (lighting, colour cast, hair or glasses): metrics may be off.")
    rep = {"warnings": warn, "skin_coverage_pct": round(float((mask > 0).mean() * 100), 1), "regions": reg,
           "lbp": {"uniform_histogram": [round(float(x), 4) for x in lbp], "entropy_bits": round(ent, 3),
                   "flat_fraction": round(float(lbp[8]), 3), "edge_like_fraction": round(float(lbp[1:3].sum()), 3)},
           "gabor": {"mean_energy": [round(float(x), 4) for x in gm], "dominant_orientation_deg": int(np.argmax(gm[:4]) * 45)},
           "smoothness_index": smooth, "fine_line_density_index": index, "skin_tone": tone,
           "disclaimer": "Descriptive image statistics only - not a medical, dermatological or age measurement. Indices are relative: compare only photos taken under the same camera and lighting."}
    if with_images:
        code = lbp_image(gray)
        lbp_vis = cv2.applyColorMap(cv2.normalize(code, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_VIRIDIS)
        edge_vis = cv2.applyColorMap(cv2.GaussianBlur(edges, (0, 0), 1.5), cv2.COLORMAP_INFERNO)
        ov = f.copy()
        ov[mask == 0] = (ov[mask == 0] * 0.25).astype(np.uint8)
        for name, frac in REGIONS.items():
            ys, xs = _region_slice(f, frac)
            cv2.rectangle(ov, (xs.start, ys.start), (xs.stop, ys.stop), (255, 210, 60), 1)
        rep["images"] = {"face": _png_b64(f), "skin_mask": _png_b64(ov), "lbp_map": _png_b64(lbp_vis),
                         "edge_map": _png_b64(edge_vis)}
    return rep
