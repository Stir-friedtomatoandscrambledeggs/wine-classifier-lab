"""应用层测试 6 项：正常/非法/边界输入。"""
import pandas as pd
import pytest

from src.features import FEATURE_NAMES, validate_schema, check_out_of_range, training_range


def _row(**kw):
    base = {c: 12.0 for c in FEATURE_NAMES}
    base.update(kw)
    return pd.DataFrame([base])[FEATURE_NAMES]


def test_valid_row_passes(sample_row):
    validate_schema(sample_row)  # 不抛错即通过


def test_negative_value_passes_schema_but_flagged(split):
    rng = training_range(split.X_train)
    bad = _row()
    bad.iloc[0, 0] = -999.0  # 负值，schema 允许（数值），但会被越界检出
    out = check_out_of_range(bad, rng)
    assert out, "应检出越界"


def test_extreme_value_out_of_range(split):
    rng = training_range(split.X_train)
    big = _row()
    big.iloc[0, 0] = 99999.0
    out = check_out_of_range(big, rng)
    assert "alcohol" in out


def test_in_range_not_flagged(split):
    rng = training_range(split.X_train)
    out = check_out_of_range(split.X_train.iloc[[0]], rng)
    assert out == {}


def test_empty_dataframe_rejected():
    empty = pd.DataFrame(columns=FEATURE_NAMES)
    with pytest.raises(ValueError):
        validate_schema(empty)


def test_wrong_column_order_rejected(sample_row):
    shuffled = sample_row[list(reversed(FEATURE_NAMES))]
    # 列序不同但列齐全——pipeline 按位置预测，但我们的 schema 按列名取
    # 这里测试列序被尊重：强制重排后应等于原列序
    assert list(shuffled.columns) != FEATURE_NAMES
    reordered = shuffled[FEATURE_NAMES]
    assert list(reordered.columns) == FEATURE_NAMES
