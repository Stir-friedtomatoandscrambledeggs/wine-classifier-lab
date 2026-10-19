"""训练脚本：比较 LogisticRegression 与 RandomForest，保存最佳 Pipeline + meta。

运行: python -m src.train
产物:
  models/model.joblib        完整 Pipeline
  models/model_meta.json     版本/特征/标签/指标/SHA256
  models/comparison.csv      两模型交叉验证对比
  models/confusion_*.png     混淆矩阵（可选，PPT 素材）
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, RANDOM_STATE, TARGET_NAMES, load_data, make_split

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def build_pipelines() -> dict:
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "rf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ]),
    }


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_precision": float(_macro_prf(y_true, y_pred)["precision"]),
        "macro_recall": float(_macro_prf(y_true, y_pred)["recall"]),
    }


def _macro_prf(y_true, y_pred) -> dict:
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "precision": rep["macro avg"]["precision"],
        "recall": rep["macro avg"]["recall"],
    }


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    X, y = load_data()
    split = make_split(X, y)

    # 同协议交叉验证（在 train 上），选模型，不碰 test
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    trained = {}
    for name, pipe in build_pipelines().items():
        t0 = time.time()
        scores = cross_val_score(pipe, split.X_train, split.y_train, cv=cv, scoring="f1_macro")
        pipe.fit(split.X_train, split.y_train)
        trained[name] = pipe
        rows.append({
            "model": name,
            "cv_macro_f1_mean": scores.mean(),
            "cv_macro_f1_std": scores.std(),
            "fit_seconds": time.time() - t0,
        })
    comp = pd.DataFrame(rows)
    comp.to_csv(MODEL_DIR / "comparison.csv", index=False)
    print("== 交叉验证对比 ==")
    print(comp.to_string(index=False))

    best_name = comp.sort_values("cv_macro_f1_mean", ascending=False).iloc[0]["model"]
    best = trained[best_name]
    print(f"\n选定模型: {best_name}（依据 train 5折 macro-F1）")

    # 最终测试集只评一次
    y_pred = best.predict(split.X_test)
    test_metrics = evaluate(split.y_test, y_pred)
    cm = confusion_matrix(split.y_test, y_pred)
    print("\n== 最终测试集（仅评一次）==")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    print("混淆矩阵：")
    print(pd.DataFrame(cm, index=TARGET_NAMES, columns=TARGET_NAMES).to_string())

    # 保存 Pipeline
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(best, model_path)

    meta = {
        "task": "Wine 三分类（红酒品种识别）",
        "positive_cost": "三类均衡，主指标 macro-F1；误判任意一类业务代价相近",
        "model_name": best_name,
        "sklearn_version": __import__("sklearn").__version__,
        "python_version": platform.python_version(),
        "git_commit": _git_commit(),
        "feature_order": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "random_state": RANDOM_STATE,
        "split": "train 60% / val 20% / test 20%, stratified, seed=42",
        "metrics_on_test": test_metrics,
        "confusion_matrix": cm.tolist(),
        "train_range": {c: [float(split.X_train[c].min()), float(split.X_train[c].max())]
                        for c in FEATURE_NAMES},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warning": "仅用于课程演示，不构成真实业务决策",
    }
    meta["model_sha256"] = sha256_of(model_path)
    meta_path = MODEL_DIR / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已保存: {model_path.name} / {meta_path.name}")
    print("model_sha256:", meta["model_sha256"])


if __name__ == "__main__":
    main()
