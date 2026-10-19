"""自主功能测试 3 项：批量 CSV 预测 + 越界统计 + 结果可下载。"""
import pandas as pd
import pytest

from src.features import FEATURE_NAMES, validate_schema, check_out_of_range, training_range


def _batch_df(split, n=5):
    return split.X_train.iloc[:n].copy()


def test_batch_valid_rows_predictable(split, trained_logreg):
    df = _batch_df(split)
    validate_schema(df)
    pred = trained_logreg.predict(df)
    assert len(pred) == len(df)


def test_batch_out_of_range_statistic(split):
    rng = training_range(split.X_train)
    df = _batch_df(split, n=3)
    df.iloc[0, 0] = -1.0   # 越下界
    df.iloc[1, 1] = 9999.0 # 越上界
    out = check_out_of_range(df, rng)
    assert out
    assert sum(out.values()) >= 2


def test_batch_result_is_serializable_csv(split, trained_logreg):
    df = _batch_df(split, n=3)
    proba = trained_logreg.predict_proba(df)
    out = df.copy()
    out["pred"] = trained_logreg.predict(df)
    for i in range(proba.shape[1]):
        out[f"p{i}"] = proba[:, i]
    csv_text = out.to_csv(index=False)
    assert "pred" in csv_text
    assert csv_text.count("\n") == len(df) + 1  # 表头 + n 行
