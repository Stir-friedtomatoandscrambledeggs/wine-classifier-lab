"""数据/划分测试 5 项。"""
import math
import pandas as pd
import pytest

from src.features import FEATURE_NAMES, TARGET_NAMES, load_data, make_split, RANDOM_STATE


def test_dataset_shape(data):
    X, y = data
    assert X.shape == (178, 13)
    assert y.shape == (178,)


def test_target_classes(data):
    _, y = data
    assert sorted(y.unique().tolist()) == [0, 1, 2]


def test_feature_order_fixed(data):
    X, _ = data
    assert list(X.columns) == FEATURE_NAMES


def test_split_sizes_disjoint(split):
    n = len(split.X_train) + len(split.X_val) + len(split.X_test)
    assert n == 178
    # 分层：每类比例约 60/20/20
    for arr in (split.X_train, split.X_val, split.X_test):
        assert len(arr) > 0


def test_split_reproducible(data):
    X, y = data
    s1 = make_split(X, y)
    s2 = make_split(X, y)
    pd.testing.assert_frame_equal(s1.X_test, s2.X_test)
    pd.testing.assert_frame_equal(s1.X_train, s2.X_train)
