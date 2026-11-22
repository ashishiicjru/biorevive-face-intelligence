#!/usr/bin/env python3
"""Age + facial-texture + skin-tone analysis from one face photo (computer vision, no cloud).

    python programs/age_texture_analyzer.py --image face.jpg
    python programs/age_texture_analyzer.py --image face.jpg --json report.json

Produces:
  * apparent age (Levi-Hassner CNN, 8 age groups, 10-crop)            - a trained model
  * texture report: LBP histogram, Gabor energy, edge / fine-line density per region (forehead, eyes, cheeks, mouth),
    smoothness index                                                   - classical CV measurements
  * skin tone: Lab colour, Individual Typology Angle (ITA) category    - lighting dependent
  * a 4-panel image (face | skin mask + regions | LBP map | edge map) saved to outputs/
  * texture-feature age estimate IF you trained one (programs/train_texture_model.py --task age)

These are descriptive image statistics, not dermatology, not a medical or health measure, and not a validated
way to determine age. Compare photos only when taken with the same camera and lighting.
"""
import argparse
import base64
import json

from _common import *      # noqa: F401,F403
from app.vision import attributes, faces, texture, texture_ml


def decode(data_url):
    return cv2.imdecode(np.frombuffer(base64.b64decode(data_url.split(",")[1]), np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--json", help="also write the full report as JSON")
    a = ap.parse_args()
    ensure_models("age_levi_hassner", "face_ssd_res10")
    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"Cannot read image: {a.image}")
    try:
        f = faces.primary_face(img)
    except faces.NoFaceError as e:
        sys.exit(str(e))
    crop = faces.crop(img, f["box"])
    rep = texture.analyze(crop, with_images=True)
    age = attributes.predict_age(crop)
    print(f"Face found ({f['detector']}), crop {crop.shape[1]}x{crop.shape[0]} px")
    print(f"\nAPPARENT AGE (CNN): ~{age['apparent_age']:.0f} years  | group {age['age_bucket']} ({age['bucket_probability']*100:.0f}%)")
    print("  " + age["uncertainty_note"])
    if texture_ml.available("age"):
        t = texture_ml.predict_age(crop); v = t["validation"]
        print(f"TEXTURE-FEATURE MODEL: ~{t['age']:.0f} years (held-out MAE {v['mae_years']:.1f} y vs baseline {v['baseline_mae_years']:.1f} y)")
    else:
        print("TEXTURE-FEATURE MODEL: not trained (python programs/train_texture_model.py --task age --data UTKFace/)")
    print(f"\nTEXTURE  smoothness index {rep['smoothness_index']} | fine-line density index {rep['fine_line_density_index']} | skin pixels {rep['skin_coverage_pct']}%")
    print(f"  LBP entropy {rep['lbp']['entropy_bits']} bits | dominant Gabor orientation {rep['gabor']['dominant_orientation_deg']} deg")
    print("  region            edge-density  roughness")
    for k, v in rep["regions"].items():
        print(f"  {k:<16}  {v['edge_density']*100:8.1f}%  {v['roughness']:10.1f}")
    t = rep["skin_tone"]
    if t:
        print(f"\nSKIN TONE  {t['ita_category']} (ITA {t['ita_degrees']} deg)  Lab L*={t['lab']['L']} a*={t['lab']['a']} b*={t['lab']['b']}  swatch {t['swatch_hex']}")
        print("  " + t["note"])
    for w in rep["warnings"]:
        print("WARNING:", w)
    print("\n" + rep["disclaimer"])
    sheet = np.hstack([cv2.resize(decode(rep["images"][k]), (256, 256)) for k in ("face", "skin_mask", "lbp_map", "edge_map")])
    OUT.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / "texture_analysis.png"), sheet)
    print("Saved:", OUT / "texture_analysis.png")
    if a.json:
        rep.pop("images"); Path(a.json).write_text(json.dumps({"age_cnn": age, "texture": rep}, indent=2))
        print("Saved:", a.json)


if __name__ == "__main__":
    main()
