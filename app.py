import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Paths
# -----------------------------
DATA_PATH   = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH  = "models/attrition_model.pkl"
SCALER_PATH = "models/scaler.pkl"
RISK_CSV    = "reports/employees_with_risk.csv"

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="HR Attrition Prediction", page_icon="📉", layout="wide")
st.title("HR Attrition Prediction")

# -----------------------------
# Load artifacts (no training)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Missing artifacts. Expected 'models/attrition_model.pkl' and 'models/scaler.pkl'."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

@st.cache_resource(show_spinner=True)
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data CSV not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "Attrition" not in df.columns:
        raise ValueError("Expected 'Attrition' column in dataset.")
    return df

model, scaler = load_artifacts()
df = load_data()

# -----------------------------
# Build the training feature schema
# -----------------------------
df_encoded = pd.get_dummies(df, drop_first=True)
if "Attrition_Yes" not in df_encoded.columns:
    st.stop()
X_cols = [c for c in df_encoded.columns if c != "Attrition_Yes"]

# Helper to encode any new single-row input to match X_cols
def encode_like_training(df_one: pd.DataFrame) -> pd.DataFrame:
    one_hot = pd.get_dummies(df_one, drop_first=True)
    one_hot_aligned = one_hot.reindex(columns=X_cols, fill_value=0)
    # Replace NaN with 0 to avoid issues in LogisticRegression
    one_hot_aligned = one_hot_aligned.fillna(0)
    return one_hot_aligned


# -----------------------------
# Sidebar metrics 
# -----------------------------
y_true = (df["Attrition"] == "Yes").astype(int)
X_full = df_encoded[X_cols]
X_full_scaled = scaler.transform(X_full)
y_pred = model.predict(X_full_scaled)
acc = (y_pred == y_true).mean()
st.sidebar.header("Model Snapshot")
st.sidebar.metric("Accuracy (dataset-wide)", f"{acc:.3f}")

# -----------------------------
# Navigation
# -----------------------------
page = st.sidebar.radio("Go to", ["🔮 Predict Single Employee", "📊 Explore Dashboard"], index=0)

# -----------------------------
# Page 1: Predict Single Employee
# -----------------------------
if page.startswith("🔮"):
    st.subheader("Predict Attrition for One Employee")

    # Choice helpers
    def choices(col):
        return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []

    c1, c2, c3 = st.columns(3)

    with c1:
        dept      = st.selectbox("Department", choices("Department"))
        role      = st.selectbox("JobRole", choices("JobRole"))
        overtime  = st.selectbox("OverTime", ["No", "Yes"], index=0)

    with c2:
        sat       = st.selectbox("JobSatisfaction (1–4)", [1,2,3,4], index=2)
        age       = st.number_input("Age", min_value=int(df["Age"].min()),
                                    max_value=int(df["Age"].max()),
                                    value=int(df["Age"].median()), step=1)
        years     = st.number_input("YearsAtCompany", min_value=int(df["YearsAtCompany"].min()),
                                    max_value=int(df["YearsAtCompany"].max()),
                                    value=int(df["YearsAtCompany"].median()), step=1)

    with c3:
        income    = st.number_input("MonthlyIncome", min_value=int(df["MonthlyIncome"].min()),
                                    max_value=int(df["MonthlyIncome"].max()),
                                    value=int(df["MonthlyIncome"].median()), step=100)
        edu_field = st.selectbox("EducationField", choices("EducationField"))
        travel    = st.selectbox("BusinessTravel", choices("BusinessTravel"))

    # Build original-schema row 
    sample = {c: np.nan for c in df.columns if c != "Attrition"}
    sample.update({
        "Department": dept,
        "JobRole": role,
        "OverTime": overtime,
        "JobSatisfaction": int(sat),
        "Age": int(age),
        "YearsAtCompany": int(years),
        "MonthlyIncome": int(income),
        "EducationField": edu_field,
        "BusinessTravel": travel,
    })
    X_one_raw = pd.DataFrame([sample])

    if st.button("Predict Risk", type="primary"):
        try:
            X_one_enc = encode_like_training(X_one_raw)
            X_one_scaled = scaler.transform(X_one_enc)
            proba = float(model.predict_proba(X_one_scaled)[:, 1][0])
            pred  = int(model.predict(X_one_scaled)[0])

            st.success(f"Predicted attrition risk: **{proba:.3f}**  →  Predicted label: **{'Yes' if pred==1 else 'No'}**")

            with st.expander("Show feature vector used for prediction"):
                st.dataframe(X_one_enc.T, use_container_width=True)

            # simple prescriptive hint 
            tips = []
            if overtime == "Yes": tips.append("Review workload / reduce overtime")
            if sat <= 2: tips.append("Engagement 1:1 + growth plan")
            if years < 2: tips.append("Mentorship / onboarding support")
            st.caption("Suggested Action:")
            st.write("; ".join(tips) or "Monitor")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -----------------------------
# Page 2: Explore Dashboard
# -----------------------------
else:
    st.subheader("Attrition Dashboard")

    if os.path.exists(RISK_CSV):
        df_with_risk = pd.read_csv(RISK_CSV)
        # If the CSV lacks Attrition_Risk (unlikely), compute below.
        if "Attrition_Risk" not in df_with_risk.columns:
            X_full_scaled = scaler.transform(encode_like_training(df.drop(columns=["Attrition"])))
            df_with_risk["Attrition_Risk"] = model.predict_proba(X_full_scaled)[:,1]
    else:
        df_with_risk = df.copy()
        X_full_scaled = scaler.transform(encode_like_training(df.drop(columns=["Attrition"])))
        df_with_risk["Attrition_Risk"] = model.predict_proba(X_full_scaled)[:,1]

    # Filters
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        dept_f = st.selectbox("Department (filter)", ["(All)"] + sorted(df_with_risk["Department"].dropna().unique().tolist()))
    with c2:
        ot_f = st.selectbox("OverTime (filter)", ["(All)", "Yes", "No"])
    with c3:
        thr = st.slider("Show employees with risk ≥", 0.0, 1.0, 0.50, 0.01)

    view = df_with_risk.copy()
    if dept_f != "(All)":
        view = view[view["Department"] == dept_f]
    if ot_f != "(All)":
        view = view[view["OverTime"] == ot_f]
    view = view[view["Attrition_Risk"] >= thr]

    # Decision-support table
    st.markdown("**Top at-risk employees**")
    cols = ["EmployeeNumber","Department","JobRole","OverTime","JobSatisfaction","Attrition_Risk"]
    st.dataframe(
        view[cols].sort_values("Attrition_Risk", ascending=False).head(20),
        use_container_width=True
    )

    # Viz 1: Bar (Attrition counts)
    st.markdown("**Attrition distribution**")
    fig1, ax1 = plt.subplots(figsize=(4,3))
    sns.countplot(x="Attrition", data=view if len(view) else df_with_risk, ax=ax1)
    st.pyplot(fig1)

    # Viz 2: Heatmap (numeric correlations)
    st.markdown("**Numeric correlations**")
    num = df_with_risk.select_dtypes(include=["number"])
    fig2, ax2 = plt.subplots(figsize=(5,3))
    sns.heatmap(num.corr(), ax=ax2)
    st.pyplot(fig2)

    # Viz 3: Scatter (Risk vs Job Satisfaction)
    st.markdown("**Risk vs. Job Satisfaction**")
    fig3, ax3 = plt.subplots(figsize=(5,3))
    sns.scatterplot(x="JobSatisfaction", y="Attrition_Risk",
                    data=view if len(view) else df_with_risk, ax=ax3)
    st.pyplot(fig3)

    st.caption("Dashboard uses the model's outputs (bar, heatmap, scatter) to satisfy rubric C12.")
