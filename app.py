import streamlit as st
import pandas as pd
import numpy as np
import joblib
from eda import (load_data, churn_distribution, churn_by_contract,
                 churn_by_tenure, churn_by_monthly_charges,
                 correlation_heatmap)

# --- Page Config ---
st.set_page_config(page_title="Churn Prediction Dashboard",
                   layout="wide", page_icon="📉")

# --- Load Assets ---
df = load_data("data/telco_churn.csv")
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# --- Sidebar Navigation ---
st.sidebar.title("📉 Churn Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", [
    "📊 EDA & Insights",
    "🔮 Predict Churn",
    "📋 Business Recommendations"
])

# ============================================================
# PAGE 1 — EDA & INSIGHTS
# ============================================================
if page == "📊 EDA & Insights":
    st.title("📊 Customer Churn Analysis")
    st.markdown("Explore churn patterns across customer segments.")
    st.markdown("---")

    # KPI Metrics
    total = len(df)
    churned = int(df["Churn"].sum())
    retained = total - churned
    churn_rate = round(df["Churn"].mean() * 100, 2)
    avg_charges = round(df["MonthlyCharges"].mean(), 2)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churned", f"{churned:,}")
    c3.metric("Retained", f"{retained:,}")
    c4.metric("Churn Rate", f"{churn_rate}%")
    c5.metric("Avg Monthly Charge", f"${avg_charges}")
    st.markdown("---")

    # Row 1 Charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(churn_distribution(df), use_container_width=True)
    with col2:
        st.plotly_chart(churn_by_contract(df), use_container_width=True)

    # Row 2 Charts
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(churn_by_tenure(df), use_container_width=True)
    with col4:
        st.plotly_chart(churn_by_monthly_charges(df),
                        use_container_width=True)

    # Heatmap
    st.subheader("🔥 Correlation Heatmap")
    st.pyplot(correlation_heatmap(df))

    # Model Performance Charts
    st.markdown("---")
    st.subheader("🤖 Model Performance")
    col5, col6 = st.columns(2)
    with col5:
        st.image("roc_curve.png", caption="ROC Curve")
    with col6:
        st.image("confusion_matrix.png", caption="Confusion Matrix")

    if st.checkbox("Show Feature Importance"):
        st.image("feature_importance.png", caption="Top 10 Features")

# ============================================================
# PAGE 2 — PREDICT CHURN
# ============================================================
elif page == "🔮 Predict Churn":
    st.title("🔮 Predict Customer Churn")
    st.markdown("Enter customer details to predict if they will churn.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Account Info")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type",
                                ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method",
                               ["Electronic check", "Mailed check",
                                "Bank transfer (automatic)",
                                "Credit card (automatic)"])

    with col2:
        st.subheader("Services")
        internet = st.selectbox("Internet Service",
                                ["DSL", "Fiber optic", "No"])
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col3:
        st.subheader("Charges & Profile")
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, 800.0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    # Build input dictionary matching feature names
    input_dict = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if phone == "Yes" else 0,
        "MultipleLines": ["No", "No phone service", "Yes"].index(multiple),
        "InternetService": ["DSL", "Fiber optic", "No"].index(internet),
        "OnlineSecurity": ["No", "No internet service", "Yes"].index(security),
        "OnlineBackup": ["No", "No internet service", "Yes"].index(backup),
        "DeviceProtection": 0,
        "TechSupport": 0,
        "StreamingTV": 0,
        "StreamingMovies": 0,
        "Contract": ["Month-to-month", "One year", "Two year"].index(contract),
        "PaperlessBilling": 1 if paperless == "Yes" else 0,
        "PaymentMethod": ["Bank transfer (automatic)",
                          "Credit card (automatic)",
                          "Electronic check",
                          "Mailed check"].index(payment),
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    input_df = pd.DataFrame([input_dict])[feature_names]
    input_scaled = scaler.transform(input_df)

    st.markdown("---")
    if st.button("🔮 Predict Now", use_container_width=True):
        prob = model.predict_proba(input_scaled)[0][1]
        risk_level = "🔴 HIGH RISK" if prob > 0.7 else \
                     "🟡 MEDIUM RISK" if prob > 0.4 else "🟢 LOW RISK"

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Churn Probability", f"{prob * 100:.1f}%")
        with col_b:
            st.metric("Risk Level", risk_level)

        if prob > 0.7:
            st.error("⚠️ This customer is very likely to churn. "
                     "Immediate retention action is recommended.")
        elif prob > 0.4:
            st.warning("⚠️ This customer shows moderate churn risk. "
                       "Consider a proactive check-in or offer.")
        else:
            st.success("✅ This customer appears stable and unlikely to churn.")

# ============================================================
# PAGE 3 — BUSINESS RECOMMENDATIONS
# ============================================================
elif page == "📋 Business Recommendations":
    st.title("📋 Business Recommendations")
    st.markdown("Strategic actions the business should take based on "
                "churn analysis findings.")
    st.markdown("---")

    st.subheader("🔑 Key Findings from the Data")
    st.markdown("""
    - **~26% overall churn rate** — 1 in 4 customers leaves, which is high
    - **Month-to-month contracts** drive the most churn (~43% churn rate)
    - **First 12 months are critical** — new customers are highest risk
    - **High monthly charges** are strongly linked to churn
    - **Customers without partners or dependents** churn significantly more
    """)

    st.markdown("---")
    st.subheader("✅ Recommended Business Actions")

    col1, col2 = st.columns(2)

    with col1:
        st.success("**1. Incentivise Long-Term Contracts**\n\n"
                   "Offer month-to-month customers a 15–20% discount to "
                   "switch to annual contracts. This single action "
                   "could cut churn by up to 30%.")

        st.success("**2. Improve Early Onboarding (First 90 Days)**\n\n"
                   "Assign new customers a dedicated support contact in "
                   "their first 3 months. Customer satisfaction in this "
                   "window is the strongest predictor of retention.")

    with col2:
        st.success("**3. Introduce Loyalty Pricing**\n\n"
                   "Reduce monthly charges progressively for long-tenure "
                   "customers. Reward loyalty before customers start "
                   "comparing competitors.")

        st.success("**4. Use This Model Monthly**\n\n"
                   "Run churn predictions on all active customers every "
                   "month. Have the retention team personally contact "
                   "anyone above 70% churn probability.")

    st.markdown("---")
    st.subheader("📈 Estimated Business Impact")
    impact_data = pd.DataFrame({
        "Action": ["Contract Incentives", "Better Onboarding",
                   "Loyalty Pricing", "Proactive Outreach"],
        "Estimated Churn Reduction": ["25-30%", "15-20%",
                                       "10-15%", "20-25%"],
        "Difficulty to Implement": ["Low", "Medium", "Low", "Medium"]
    })
    st.dataframe(impact_data, use_container_width=True)