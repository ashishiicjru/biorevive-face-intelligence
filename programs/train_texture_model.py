#!/usr/bin/env python3
"""Train the texture-feature age / gender models (pure NumPy: ridge regression / logistic regression).

    python programs/train_texture_model.py --task age    --data path/to/UTKFace
    python programs/train_texture_model.py --task gender --data path/to/UTKFace
    python programs/train_texture_model.py --task age --synthetic 300      # machinery self-test, saves nothing

UTKFace file names are  <age>_<gender>_<race>_<date>.jpg  (gender 0 = male, 1 = female). Only age and gender are
read; race is ignored. The trained model is written to models/texture_<task>.npz and the web app / programs pick
it up automatically. 20 % of the data is held out and the metrics are printed NEXT TO a trivial baseline (predict
the median age / the majority class) so you can see whether the texture features learned anything real.
Hand-crafted texture features are much weaker than the pretrained CNNs and reflect your dataset's population.
"""
import argparse

from _common import *      # noqa: F401,F403
from app.vision import texture_ml


def synthetic(n, task, seed=0):
    """Fake 'faces' whose fine-line noise grows with the label - ONLY to prove the pipeline learns something."""
    rng = np.random.RandomState(seed)
    X, y = [], []
    yy, xx = np.mgrid[0:128, 0:128]
    for _ in range(n):
        label = rng.randint(18, 80) if task == "age" else rng.randint(0, 2)
        tone = rng.randint(120, 200)
        base = np.stack([tone + 0.2 * (xx - 64) - 0.15 * (yy - 64)] * 3, -1) + np.array([-8, 0, 10])
        amp = (label - 18) / 6.0 if task == "age" else 2 + 5 * label
        base = base + rng.randn(128, 128, 1) * amp
        X.append(texture_ml.texture.feature_vector(np.clip(base, 0, 255).astype(np.uint8))); y.append(label)
    return np.array(X), np.array(y, float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["age", "gender"], required=True)
    ap.add_argument("--data", help="folder of UTKFace-style images")
    ap.add_argument("--limit", type=int, help="use at most N images")
    ap.add_argument("--synthetic", type=int, metavar="N", help="self-test on N synthetic images (nothing is saved)")
    a = ap.parse_args()
    if a.synthetic:
        X, y = synthetic(a.synthetic, a.task)
        print(f"[synthetic] {len(X)} generated samples - this validates the training code only, not real accuracy.")
    elif a.data:
        print("Extracting texture features ...")
        X, ages, genders = texture_ml.load_utkface(a.data, a.limit, lambda i, n: print(f"\r  {i}/{n}", end="", flush=True))
        print()
        if len(X) == 0:
            sys.exit("No usable images found (expected UTKFace-style names such as 25_0_1_20170116.jpg).")
        y = ages if a.task == "age" else genders
    else:
        ap.error("give --data FOLDER (or --synthetic N for a self-test)")
    model, m = texture_ml.train(a.task, X, y)
    print(f"Trained on {m['n_train']} / validated on {m['n_val']} held-out images")
    if a.task == "age":
        print(f"  held-out MAE: {m['mae_years']:.2f} years   (baseline 'always the median': {m['baseline_mae_years']:.2f} years)")
        print("  -> better than baseline" if m["mae_years"] < m["baseline_mae_years"] else "  -> NOT better than the baseline: do not rely on this model")
    else:
        print(f"  held-out accuracy: {m['accuracy']*100:.1f}%   (baseline 'majority class': {m['baseline_accuracy']*100:.1f}%)")
        print("  -> better than baseline" if m["accuracy"] > m["baseline_accuracy"] else "  -> NOT better than the baseline: do not rely on this model")
    if a.synthetic:
        print("Self-test only - model not saved.")
    else:
        print("Saved:", texture_ml.save(a.task, model, m))


if __name__ == "__main__":
    main()
