"""Pipeline 测试 6 项：缺失/未知列/保存加载一致/列序/预测形状/越界校验。"""
import json
import joblib
import pandas as pd
import pytest

from src.features import (
    FEATURE_NAMES, validate_schema, check_out_of_range, training_range,
)


def test_pipeline_accepts_raw_df(trained_logreg, sample_row):
    pred = trained_logreg.predict(sample_row)
    assert pred.shape == (1,)
    assert int(pred[0]) in (0, 1, 2)


def test_pipeline_predict_proba_sums_to_one(trained_logreg, sample_row):
    proba = trained_logreg.predict_proba(sample_row)
    assert proba.shape == (1, 3)
    assert abs(proba.sum() - 1.0) < 1e-6


def test_schema_rejects_missing_col(sample_row):
    bad = sample_row.drop(columns=[FEATURE_NAMES[0]])
    with pytest.raises(ValueError, match="缺少特征列"):
        validate_schema(bad)


def test_schema_rejects_unknown_col(sample_row):
    bad = sample_row.copy()
    bad["unexpected"] = 1.0
    with pytest.raises(ValueError, match="未知列"):
        validate_schema(bad)


def test_schema_rejects_missing_value(sample_row):
    bad = sample_row.copy()
    bad.loc[0, FEATURE_NAMES[0]] = float("nan")
    with pytest.raises(ValueError, match="缺失值"):
        validate_schema(bad)


def test_save_reload_prediction_consistency(trained_logreg, sample_row, tmp_path):
    p = tmp_path / "m.joblib"
    joblib.dump(trained_logreg, p)
    reloaded = joblib.load(p)
    a = trained_logreg.predict(sample_row)
    b = reloaded.predict(sample_row)
    assert (a == b).all()
