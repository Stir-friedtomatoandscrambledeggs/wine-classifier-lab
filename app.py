"""Streamlit 推理界面：加载固定 Pipeline，单条表单 + 批量 CSV 预测。

启动: streamlit run app.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.features import FEATURE_NAMES, TARGET_NAMES, check_out_of_range, validate_schema

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.joblib"
META_PATH = ROOT / "models" / "model_meta.json"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, meta


def predict_frame(model, df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """对已通过 schema 的 DataFrame 做预测，返回带概率与类别的结果表。"""
    proba = model.predict_proba(df)
    pred_idx = model.predict(df)
    out = df.copy()
    out["predicted"] = [TARGET_NAMES[i] for i in pred_idx]
    for i, name in enumerate(TARGET_NAMES):
        out[f"p_{name}"] = proba[:, i].round(4)
    # 越界提示
    rng = pd.DataFrame(meta["train_range"], index=["min", "max"]).T
    out_of_range = check_out_of_range(df, rng)
    return out, out_of_range


def main() -> None:
    st.set_page_config(page_title="Wine 品种分类", layout="wide")
    st.title("红酒品种智能分类器")
    st.caption("课程演示项目 · 模型仅用于教学，不构成业务决策")

    if not MODEL_PATH.exists():
        st.error("未找到 models/model.joblib，请先运行 python -m src.train")
        return

    model, meta = load_model()
    with st.expander("模型信息", expanded=False):
        st.json(meta)

    tab1, tab2 = st.tabs(["单条预测", "批量 CSV 预测（自主功能）"])

    with tab1:
        st.subheader("输入 13 项理化指标")
        cols = st.columns(4)
        values = {}
        for i, feat in enumerate(FEATURE_NAMES):
            with cols[i % 4]:
                values[feat] = st.number_input(feat, value=12.0, step=0.1, key=feat)
        if st.button("预测", type="primary"):
            row = pd.DataFrame([values])[FEATURE_NAMES]
            try:
                validate_schema(row)
                result, oor = predict_frame(model, row, meta)
                st.success(f"预测类别: **{result.iloc[0]['predicted']}**")
                st.dataframe(result[[c for c in result.columns if c.startswith('p_')]],
                             use_container_width=True)
                if oor:
                    st.warning(f"以下特征超出训练范围，结果可能不可靠: {oor}")
            except Exception as e:
                st.error(f"输入校验失败: {e}")

    with tab2:
        st.subheader("上传 CSV 批量预测并下载结果")
        st.write(f"CSV 需包含列: {', '.join(FEATURE_NAMES)}")
        up = st.file_uploader("选择 CSV", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            try:
                validate_schema(df)
                result, oor = predict_frame(model, df, meta)
                st.success(f"共 {len(df)} 行，全部预测成功")
                st.dataframe(result.head(20), use_container_width=True)
                if oor:
                    st.warning(f"越界特征统计: {oor}")
                buf = io.StringIO()
                result.to_csv(buf, index=False)
                st.download_button(
                    "下载预测结果 CSV",
                    data=buf.getvalue().encode("utf-8-sig"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"批量预测失败: {e}")


if __name__ == "__main__":
    main()
