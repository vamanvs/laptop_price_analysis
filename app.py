# app.py — robust UI for Laptop Price Predictor
import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Load data (use CSV, not pickles) ----------
DATA_CSV = "laptop_data.csv"
if not os.path.exists(DATA_CSV):
    st.error(f"Data file not found: {os.path.abspath(DATA_CSV)}")
    st.stop()

df = pd.read_csv(DATA_CSV)

# Figure out target column
target_candidates = ["Price", "price"]
target_col = next((c for c in target_candidates if c in df.columns), None)
if target_col is None:
    st.error(f"Could not find target column (tried {target_candidates}). Columns found: {list(df.columns)}")
    st.stop()

# Features used for prediction
X = df.drop(columns=[target_col])
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# ---------- Load trained pipeline ----------
PIPE_PATH = "pipe.pkl"
pipe = None
if os.path.exists(PIPE_PATH):
    try:
        with open(PIPE_PATH, "rb") as f:
            pipe = pickle.load(f)
    except Exception as e:
        st.error(
            "Failed to load trained pipeline from 'pipe.pkl'. "
            "Re-train with your current environment (you already have train_model.py)."
        )
        st.exception(e)
        st.stop()
else:
    st.warning(
        "No 'pipe.pkl' found. Train a model first:\n\n"
        "```powershell\npython train_model.py\n```"
    )
    st.stop()

# ---------- Page UI ----------
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")
st.title("💻 Laptop Price Predictor")

st.markdown(
    "Select laptop specifications below. "
    "This app builds inputs from your dataset columns, so it adapts to your CSV automatically."
)

# Make two columns for a tidy layout
left, right = st.columns(2)

# Build a default/template row using the most frequent values
default_row = {}
for c in X.columns:
    if c in num_cols:
        # numeric default = median
        val = float(df[c].median()) if pd.api.types.is_float_dtype(df[c]) else int(df[c].median())
    else:
        # categorical default = mode (most frequent)
        try:
            val = df[c].mode(dropna=True).iloc[0]
        except Exception:
            val = df[c].dropna().iloc[0] if df[c].dropna().shape[0] else ""
    default_row[c] = val

# Widgets: numeric on the left, categorical on the right
user_vals = {}

with left:
    st.subheader("Numeric Features")
    if len(num_cols) == 0:
        st.info("No numeric columns detected in your features.")
    for c in num_cols:
        col_min = float(df[c].min())
        col_max = float(df[c].max())
        col_val = float(default_row[c])
        # Choose slider granularity
        step = 0.1 if pd.api.types.is_float_dtype(df[c]) else 1.0
        user_vals[c] = st.slider(
            c, min_value=float(col_min), max_value=float(col_max), value=float(col_val), step=step
        )

with right:
    st.subheader("Categorical Features")
    if len(cat_cols) == 0:
        st.info("No categorical columns detected in your features.")
    for c in cat_cols:
        options = sorted([str(x) for x in df[c].dropna().unique().tolist()])
        default = str(default_row[c]) if default_row[c] is not None else (options[0] if options else "")
        user_vals[c] = st.selectbox(c, options, index=(options.index(default) if default in options else 0))

st.divider()

# ---------- Prediction ----------
col_a, col_b = st.columns([1, 3])
with col_a:
    predict_btn = st.button("Predict Price", type="primary")

if predict_btn:
    try:
        # Build one-row DataFrame in the exact feature order used in training
        input_df = pd.DataFrame([[user_vals[c] for c in X.columns]], columns=X.columns)

        # Run prediction
        pred = pipe.predict(input_df)
        price_pred = float(pred[0])

        st.success(f"💰 Estimated Price: **{price_pred:,.2f}**")
        st.caption("Prediction based on your trained pipeline (OneHot + RandomForest by default).")

        with st.expander("See input row used for prediction"):
            st.dataframe(input_df)

    except Exception as e:
        st.error("Prediction failed. See details below.")
        st.exception(e)

# ---------- Debug info ----------
with st.expander("🔧 DEBUG (env & model details)"):
    try:
        import sklearn  # noqa: F401

        st.write("Python:", sys.version.split()[0])
        st.write("Python exe:", sys.executable)
        st.write("Working directory:", os.getcwd())
        st.write("CSV path:", os.path.abspath(DATA_CSV))
        st.write("Data shape:", df.shape)
        st.write("Features (X) shape:", X.shape)
        st.write("Numeric cols:", num_cols)
        st.write("Categorical cols:", cat_cols)
        try:
            import sklearn as sk  # type: ignore

            st.write("scikit-learn:", sk.__version__)
        except Exception:
            pass
        st.write("pipe.pkl exists:", os.path.exists(PIPE_PATH))
        st.write("Model type:", type(pipe).__name__)
    except Exception as e:
        st.write("Debug failed:", repr(e))
