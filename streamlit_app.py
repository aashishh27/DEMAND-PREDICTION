
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📦 Edmonton Hamper Forecast 2025", layout="wide")
st.title("📦 Hamper Demand Forecast Studio – Edmonton 2025")

# ─────────────────────────────────────────────────────────────────────────────
# Load all resources
@st.cache_data
def load_all():
    df = pd.read_csv("region_client_df (1).csv", parse_dates=["pickup_date", "first_visit_date"])
    model = joblib.load("tuned_rf_pipe.joblib")
    features = joblib.load("features.pkl")
    holidays = joblib.load("holiday_dates.pkl")
    return df, model, features, holidays

df, model, FEATURES, holiday_dates = load_all()

# ─────────────────────────────────────────────────────────────────────────────
# Forecast function
def simulate_2025_by_region(df, model, holiday_dates):
    results = []
    global_start = df["pickup_date"].min()
    start_date = df["pickup_date"].max() + pd.Timedelta(days=1)
    future_dates = pd.date_range(start_date, "2025-12-31", freq="D")

    for reg in tqdm(df["region"].unique(), desc="Forecasting Regions"):
        hist = df[df["region"] == reg].sort_values("pickup_date")
        daily = hist.set_index("pickup_date")["daily_pickups"].astype(float).copy()
        first_visit = pd.to_datetime(hist["first_visit_date"]).min()

        for d in future_dates:
            feat = {
                'lag_1': daily.get(d - pd.Timedelta(days=1), 0.0),
                'rolling_7d':  daily.loc[d - pd.Timedelta(7):d - pd.Timedelta(1)].mean(),
                'rolling_14d': daily.loc[d - pd.Timedelta(14):d - pd.Timedelta(1)].mean(),
                'rolling_30d': daily.loc[d - pd.Timedelta(30):d - pd.Timedelta(1)].mean(),
                'days_since_first_visit': (d - first_visit).days,
                'dow_sin': np.sin(2*np.pi * d.weekday() / 7),
                'dow_cos': np.cos(2*np.pi * d.weekday() / 7),
                'moy_sin': np.sin(2*np.pi * (d.month - 1) / 12),
                'moy_cos': np.cos(2*np.pi * (d.month - 1) / 12),
                'day_index': (d - global_start).days,
                'is_holiday': int(d in holiday_dates),
                'pre_holiday': int((d + pd.Timedelta(days=1)) in holiday_dates),
                'monthly_trend': df["monthly_trend"].median(),
                'seasonal_scale': df["seasonal_scale"].median()
            }
            for m in range(2, 13):
                feat[f"mon_{m}"] = 1 if d.month == m else 0
            Xp = pd.DataFrame([feat])[FEATURES]
            pred = model.predict(Xp)[0]
            results.append({"region": reg, "pickup_date": d, "predicted_daily": pred})
            daily.at[d] = pred
    return pd.DataFrame(results)

# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Forecast Trend", "🗺️ Regional Demand Map", "📥 Download"])

with tab1:
    st.subheader("📅 Daily Citywide Forecast (2025)")
    if st.button("🚀 Generate 2025 Forecast"):
        forecast_df = simulate_2025_by_region(df, model, holiday_dates)
        forecast_df.to_csv("forecast_df.csv", index=False)
        daily = forecast_df.groupby("pickup_date")["predicted_daily"].sum().reset_index()
        fig = px.line(daily, x="pickup_date", y="predicted_daily",
                      title="Total Forecasted Pickups by Date", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        st.session_state["forecast_df"] = forecast_df

with tab2:
    st.subheader("📍 Forecasted Demand by Region")
    if "forecast_df" in st.session_state:
        region_total = st.session_state["forecast_df"].groupby("region")["predicted_daily"].sum().reset_index()
        coords = {
            'North Edmonton': (53.624, -113.489),
            'Northeast': (53.592, -113.425),
            'South Edmonton': (53.473, -113.520),
            'West Edmonton': (53.544, -113.652),
            'Central': (53.546, -113.491),
            'Southeast': (53.500, -113.410),
            'Far South': (53.410, -113.500),
            'Unknown / Islamic Family Pickup': (53.544, -113.490)
        }
        region_total["lat"] = region_total["region"].map(lambda x: coords.get(x, (53.54, -113.49))[0])
        region_total["lon"] = region_total["region"].map(lambda x: coords.get(x, (53.54, -113.49))[1])

        fig_map = px.scatter_mapbox(
            region_total, lat="lat", lon="lon", size="predicted_daily",
            color="predicted_daily", hover_name="region", zoom=9,
            size_max=40, mapbox_style="carto-positron",
            title="Predicted Pickup Volume by Region"
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Please run the forecast first in the 'Forecast Trend' tab.")

with tab3:
    if "forecast_df" in st.session_state:
        st.download_button("📥 Download Full Forecast CSV",
                           data=st.session_state["forecast_df"].to_csv(index=False),
                           file_name="forecast_2025.csv", mime="text/csv")
    else:
        st.warning("No forecast available to download. Please generate one first.")
