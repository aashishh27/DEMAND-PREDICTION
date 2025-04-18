import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import plotly.express as px
import os

# ─────────────────────────────────────────────────────────────
# 🔧 Load Data + Model
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("region_client_df (1).csv", parse_dates=["pickup_date", "first_visit_date"])
model = joblib.load("final_rf_model_seasonal.joblib")
features = joblib.load("features.pkl")
holiday_dates = joblib.load("holiday_dates.pkl")

# ─────────────────────────────────────────────────────────────
# 📈 Forecast Simulation Function
# ─────────────────────────────────────────────────────────────
def simulate_forecast(df, model, holiday_dates):
    results = []
    start_date = df['pickup_date'].max() + pd.Timedelta(days=1)
    global_start = df['pickup_date'].min()
    future_dates = pd.date_range(start=start_date, end="2025-12-31")

    for reg in tqdm(df['region'].unique(), desc="Forecasting Regions"):
        hist = df[df['region'] == reg].sort_values('pickup_date')
        daily = hist.set_index('pickup_date')['daily_pickups'].copy()
        first_visit = pd.to_datetime(hist['first_visit_date'].min())

        for d in future_dates:
            feat = {
                'lag_1': daily.get(d - pd.Timedelta(1), 0.0),
                'rolling_7d':  daily.loc[d - pd.Timedelta(7):d - pd.Timedelta(1)].mean(),
                'rolling_14d': daily.loc[d - pd.Timedelta(14):d - pd.Timedelta(1)].mean(),
                'rolling_30d': daily.loc[d - pd.Timedelta(30):d - pd.Timedelta(1)].mean(),
                'days_since_first_visit': (d - first_visit).days,
                'dow_sin': np.sin(2*np.pi * d.weekday()/7),
                'dow_cos': np.cos(2*np.pi * d.weekday()/7),
                'moy_sin': np.sin(2*np.pi * (d.month-1)/12),
                'moy_cos': np.cos(2*np.pi * (d.month-1)/12),
                'day_index': (d - global_start).days,
                'is_holiday': int(d in holiday_dates),
                'pre_holiday': int(d - pd.Timedelta(1) in holiday_dates),
                'pickup_month': d.month
            }
            for m in range(2,13):  # One-hot encode months
                feat[f"mon_{m}"] = 1 if d.month == m else 0

            X = pd.DataFrame([feat])[features]
            pred = model.predict(X)[0]
            results.append({'region': reg, 'pickup_date': d, 'predicted_daily': pred})
            daily.at[d] = pred

    return pd.DataFrame(results)

# Simulate once on app load
forecast_df = simulate_forecast(df, model, holiday_dates)

# ─────────────────────────────────────────────────────────────
# 📋 Streamlit Layout
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="📦 Demand Prediction Studio", layout="wide")
st.title("📦 2025 Geospatial Demand Forecasting")

st.markdown("""
**🎯 Problem Statement**  
Predict daily hamper demand across Edmonton regions to help **Islamic Family** plan mobile food distribution and outreach.

**📊 KPI Goals**  
- Predict upcoming demand across regions  
- Identify clusters of rising/falling demand  
- Show model interpretability with SHAP  

Model Used: `final_rf_model_seasonal.joblib` (Tuned Random Forest)
""")

# ─────────────────────────────────────────────────────────────
# 🔎 Sidebar Filters
# ─────────────────────────────────────────────────────────────
regions = sorted(forecast_df["region"].unique())
selected_regions = st.sidebar.multiselect("Select Region(s)", regions, default=regions[:4])
min_date = forecast_df["pickup_date"].min()
max_date = forecast_df["pickup_date"].max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

mask = (
    forecast_df['region'].isin(selected_regions) &
    (forecast_df['pickup_date'] >= pd.to_datetime(date_range[0])) &
    (forecast_df['pickup_date'] <= pd.to_datetime(date_range[1]))
)
filtered = forecast_df[mask]

# ─────────────────────────────────────────────────────────────
# 📊 Tabs Layout
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecast Explorer",
    "📍 Regional Clusters",
    "📊 Model Performance",
    "🧠 SHAP Insights"
])

# ───────── Tab 1: Forecast Trend ─────────
with tab1:
    st.subheader("📈 Forecasted Daily Pickups (2025)")
    fig = px.line(filtered, x="pickup_date", y="predicted_daily", color="region")
    fig.update_layout(title="Daily Forecast per Region", xaxis_title="Date", yaxis_title="Predicted Pickups")
    st.plotly_chart(fig, use_container_width=True)

# ───────── Tab 2: Region Cluster Map ─────────
with tab2:
    st.subheader("📍 Average Daily Pickups by Region")
    region_avg = filtered.groupby("region")["predicted_daily"].mean().reset_index()
    fig = px.bar(region_avg.sort_values("predicted_daily", ascending=False),
                 x="region", y="predicted_daily", color="predicted_daily")
    fig.update_layout(title="Average Predicted Demand by Region", xaxis_title="Region", yaxis_title="Pickups")
    st.plotly_chart(fig, use_container_width=True)

# ───────── Tab 3: Model Comparison ─────────
with tab3:
    st.subheader("📊 Model Performance Summary")
    st.markdown("""
    - **Random Forest (Tuned)**  
      - RMSE: `0.74`  
      - R²: `0.11` ✅  
      
    - **XGBoost (Tuned)**  
      - RMSE: `0.76`  
      - R²: `0.05`  

    ✅ Random Forest selected due to better interval performance and SHAP clarity.
    """)

# ───────── Tab 4: SHAP Explanations ─────────
with tab4:
    st.subheader("🧠 SHAP Visual Insights")
    shap_dir = "shap_images"
    shap_files = sorted([f for f in os.listdir(shap_dir) if f.endswith(".png")])

    titles = [
        "SHAP Summary Plot",
        "SHAP Value by Feature",
        "Partial Dependence: lag_1 vs rolling_7d",
        "Residuals vs Fitted",
        "Residual Histogram",
        "Prediction Intervals (Test Set)",
        "Seasonal Decomposition"
    ]

    for i, file in enumerate(shap_files):
        if i < len(titles):
            st.image(f"{shap_dir}/{file}", caption=titles[i], use_column_width=True)

# ─────────────────────────────────────────────────────────────
# ✅ Footer
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("📁 Final Model: `final_rf_model_seasonal.joblib` | By Aashish Arora")
