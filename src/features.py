"""Wine 数据集加载、Schema 校验与可复现划分。

样本单位：一瓶红酒的 13 项理化化验值。
标签：wine 类别 (0=class_0, 1=class_1, 2=class_2)。
划分策略：分层抽样，固定 random_state。
- train 60% / val 20% / test 20%
预处理只在 train 拟合，避免数据泄漏。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

FEATURE_NAMES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue",
    "od280/od315_of_diluted_wines", "proline",
]

TARGET_NAMES = ["class_0", "class_1", "class_2"]


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """加载 Wine 数据集，返回 (X, y)。X 列顺序固定为 FEATURE_NAMES。"""
    bunch = load_wine(as_frame=True)
    X: pd.DataFrame = bunch.data[FEATURE_NAMES].copy()
    y: pd.Series = bunch.target.copy()
    y.name = "target"
    return X, y


def validate_schema(df: pd.DataFrame) -> None:
    """对推理输入做 Schema 校验：列齐全、全数值、无缺失。

    越界（数值超出训练范围）不在此抛错，由 check_range 提示。
    """
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"缺少特征列: {missing}")
    extra = [c for c in df.columns if c not in FEATURE_NAMES]
    if extra:
        raise ValueError(f"出现未知列: {extra}")
    if df[FEATURE_NAMES].isna().any().any():
        raise ValueError("输入存在缺失值，请补全后再预测")
    if not pd.api.types.is_numeric_dtype(df[FEATURE_NAMES].to_numpy().dtype) and \
       not all(pd.api.types.is_numeric_dtype(df[c]) for c in FEATURE_NAMES):
        raise ValueError("所有特征列必须为数值")


def make_split(X: pd.DataFrame, y: pd.Series) -> SplitData:
    """分层划分 train 60 / val 20 / test 20。"""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25,
        random_state=RANDOM_STATE, stratify=y_trainval
    )
    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)


def training_range(X_train: pd.DataFrame) -> pd.DataFrame:
    """返回每列训练集 min/max，用于推理时越界提示。"""
    return pd.DataFrame({"min": X_train.min(), "max": X_train.max()})


def check_out_of_range(df: pd.DataFrame, rng: pd.DataFrame) -> dict:
    """对推理样本逐列检查是否越界，返回 {col: 越界行数}。"""
    out: dict = {}
    for col in FEATURE_NAMES:
        below = (df[col] < rng.loc[col, "min"]).sum()
        above = (df[col] > rng.loc[col, "max"]).sum()
        if below or above:
            out[col] = int(below + above)
    return out
