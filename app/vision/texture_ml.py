"""Trainable *texture-feature* models for age (ridge regression) and gender (logistic regression).

Pure numpy, no extra dependencies. Features come from app/vision/texture.feature_vector()
(LBP histogram, Gabor energy, region edge density, skin Lab colour, smoothness).

No pretrained weights ship with the app: you train on your own labelled faces, e.g. UTKFace
(file names  <age>_<gender>_<race>_<date>.jpg , gender 0 = male, 1 = female):

    python programs/train_texture_model.py --task age    --data path/to/UTKFace
    python programs/train_texture_model.py --task gender --data path/to/UTKFace

Hand-crafted texture features are much weaker than the CNNs; expect clearly worse accuracy than
the pretrained networks, and results that reflect whatever population your dataset covers.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from .. import model_registry as mr
from . import texture


def model_path(task):
    return mr.path_of(f"texture_{task}.npz")


def available(task):
    return model_path(task).exists()


def _standardise(X, mu=None, sd=None):
    mu = X.mean(0) if mu is None else mu
    sd = (X.std(0) if sd is None else sd) + 1e-8
    return (X - mu) / sd, mu, sd


def fit_ridge(X, y, lam=3.0):
    Xs, mu, sd = _standardise(X)
    A = np.hstack([Xs, np.ones((len(Xs), 1))])
    reg = lam * np.eye(A.shape[1]); reg[-1, -1] = 0
    w = np.linalg.solve(A.T @ A + reg, A.T @ y)
    return {"kind": "ridge", "w": w, "mu": mu, "sd": sd}


def fit_logistic(X, y, lam=1.0, iters=400, lr=0.1):
    Xs, mu, sd = _standardise(X)
    A = np.hstack([Xs, np.ones((len(Xs), 1))])
    w = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (p - y) / len(y) + lam * np.r_[w[:-1], 0] / len(y)
        w -= lr * g
    return {"kind": "logistic", "w": w, "mu": mu, "sd": sd}


def _apply(model, X):
    Xs = (X - model["mu"]) / model["sd"]
    return np.hstack([Xs, np.ones((len(Xs), 1))]) @ model["w"]


def evaluate(model, X, y):
    z = _apply(model, X)
    if model["kind"] == "ridge":
        return {"mae_years": float(np.abs(z - y).mean())}
    return {"accuracy": float(((z > 0).astype(int) == y).mean())}


def train(task, X, y, holdout=0.2, seed=0):
    """task: 'age' | 'gender'. Returns (model, metrics) with metrics measured on a held-out split."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    if len(X) < 30:
        raise ValueError("Need at least 30 labelled faces to train and hold out a validation split.")
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X)); n_val = max(5, int(holdout * len(X)))
    va, tr = idx[:n_val], idx[n_val:]
    fit = fit_ridge if task == "age" else fit_logistic
    model = fit(X[tr], y[tr])
    metrics = evaluate(model, X[va], y[va])
    metrics.update({"n_train": int(len(tr)), "n_val": int(len(va))})
    if task == "age":
        metrics["baseline_mae_years"] = float(np.abs(y[va] - np.median(y[tr])).mean())   # predict-the-median
    else:
        metrics["baseline_accuracy"] = float(max(y[va].mean(), 1 - y[va].mean()))         # majority class
    return model, metrics


def save(task, model, metrics):
    p = model_path(task)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, w=model["w"], mu=model["mu"], sd=model["sd"], kind=model["kind"], metrics=json.dumps(metrics))
    return p


def load(task):
    d = np.load(model_path(task), allow_pickle=False)
    return {"kind": str(d["kind"]), "w": d["w"], "mu": d["mu"], "sd": d["sd"]}, json.loads(str(d["metrics"]))


def predict_age(face_bgr):
    model, metrics = load("age")
    v = texture.feature_vector(face_bgr)[None]
    return {"age": round(float(np.clip(_apply(model, v)[0], 1, 100)), 1), "model": "texture-feature ridge regression",
            "validation": metrics}


def predict_gender(face_bgr):
    model, metrics = load("gender")
    z = float(_apply(model, texture.feature_vector(face_bgr)[None])[0])
    p_f = 1 / (1 + np.exp(-z))
    return {"label": "female-presenting" if p_f >= 0.5 else "male-presenting",
            "probability": round(float(max(p_f, 1 - p_f)), 3), "model": "texture-feature logistic regression",
            "validation": metrics}


def load_utkface(folder, limit=None, progress=None):
    """Read a UTKFace-style folder -> (X features, ages, genders)."""
    X, ages, genders = [], [], []
    files = sorted(Path(folder).glob("*.jpg")) + sorted(Path(folder).glob("*.png"))
    for i, f in enumerate(files[:limit] if limit else files):
        parts = f.name.split("_")
        try:
            age, gen = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            continue
        img = cv2.imread(str(f))
        if img is None or gen not in (0, 1):
            continue
        X.append(texture.feature_vector(img)); ages.append(age); genders.append(gen)
        if progress and i % 200 == 0:
            progress(i, len(files))
    return np.array(X), np.array(ages, float), np.array(genders, float)
