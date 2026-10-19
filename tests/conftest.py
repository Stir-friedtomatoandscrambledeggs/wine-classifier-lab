import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from src.features import (
    FEATURE_NAMES, TARGET_NAMES, load_data, make_split,
    validate_schema, check_out_of_range, training_range, RANDOM_STATE,
)


@pytest.fixture(scope="session")
def data():
    X, y = load_data()
    return X, y


@pytest.fixture(scope="session")
def split(data):
    X, y = data
    return make_split(X, y)


@pytest.fixture(scope="session")
def trained_logreg(split):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    pipe.fit(split.X_train, split.y_train)
    return pipe


@pytest.fixture(scope="session")
def trained_rf(split):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ])
    pipe.fit(split.X_train, split.y_train)
    return pipe


@pytest.fixture(scope="session")
def sample_row(data):
    X, _ = data
    return X.iloc[[0]].copy()
