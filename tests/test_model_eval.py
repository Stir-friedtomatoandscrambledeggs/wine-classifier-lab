"""模型评价测试 2 项：两模型同协议比较。"""
from sklearn.metrics import f1_score


def test_logreg_f1_above_07(split, trained_logreg):
    pred = trained_logreg.predict(split.X_test)
    f1 = f1_score(split.y_test, pred, average="macro")
    assert f1 > 0.7, f"logreg macro-F1={f1:.3f} 过低"


def test_rf_f1_above_07(split, trained_rf):
    pred = trained_rf.predict(split.X_test)
    f1 = f1_score(split.y_test, pred, average="macro")
    assert f1 > 0.7, f"rf macro-F1={f1:.3f} 过低"
