import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

st.set_page_config(page_title="2025 Food Hamper Forecast", layout="wide")
st.title("📦 Forecasting Daily Hamper Demand in 2025")

# --- Load model and data ---
@st.cache_data
def load_data():
    return pd.read_csv("region_client_df (1).csv", parse_dates=["pickup_date", "first_visit_date"])

def load_model():
    return joblib.load("final_rf_model_seasonal.joblib")

# --- Setup ---
df = load_data()
model = load_model()
df = df.sort_values("pickup_date")
holiday_dates = pd.to_datetime([
    "2023-01-01", "2023-04-07", "2023-05-22", "2023-07-01", "2023-08-07",
    "2023-09-04", "2023-10-09", "2023-11-11", "2023-12-25", "2023-12-26"
])

# --- Forecast Function ---
def simulate_2025_by_region_rf(df, model, holiday_dates):
    records = []
    global_start = df['pickup_date'].min()
    start_date = df['pickup_date'].max() + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=start_date, end="2025-12-31", freq='D')

    for reg in tqdm(df['region'].unique(), desc="Forecasting Regions"):
        hist = df[df['region'] == reg].sort_values('pickup_date')
        daily = hist.set_index('pickup_date')['daily_pickups'].astype(float).copy()
        first_visit = pd.to_datetime(hist['first_visit_date']).min()

        for d in future_dates:
            feat = {
                'lag_1': daily.get(d - pd.Timedelta(days=1), 0.0),
                'rolling_7d': daily.loc[d - pd.Timedelta(days=7):d - pd.Timedelta(days=1)].mean(),
                'rolling_14d': daily.loc[d - pd.Timedelta(days=14):d - pd.Timedelta(days=1)].mean(),
                'rolling_30d': daily.loc[d - pd.Timedelta(days=30):d - pd.Timedelta(days=1)].mean(),
                'days_since_first_visit': (d - first_visit).days,
                'dow_sin': np.sin(2 * np.pi * d.weekday() / 7),
                'dow_cos': np.cos(2 * np.pi * d.weekday() / 7),
                'moy_sin': np.sin(2 * np.pi * (d.month - 1) / 12),
                'moy_cos': np.cos(2 * np.pi * (d.month - 1) / 12),
                'day_index': (d - global_start).days,
                'is_holiday': int(d in holiday_dates),
                'pre_holiday': int((d + pd.Timedelta(days=1)) in holiday_dates),
                'monthly_trend': 0.0,
                'seasonal_scale': 0.5,
                'lag1_x_season': 0.0,
                'trend_x_holiday': 0.0,
                'lag_ratio': 0.0,
                'delta_lag': 0.0,
                'region_trend': 0.0
            }
            for m in range(2, 13):
                feat[f"mon_{m}"] = int(d.month == m)

            Xp = pd.DataFrame([feat])[model.feature_names_in_]
            pred = model.predict(Xp)[0]

            records.append({
                'region': reg,
                'pickup_date': d,
                'predicted_daily': pred
            })
            daily.at[d] = pred

    return pd.DataFrame(records)

# --- Run Forecast ---
st.info("Generating forecast... This may take up to 30 seconds.")
forecast_df = simulate_2025_by_region_rf(df, model, holiday_dates)

# --- Aggregate and Plot ---
total_by_day = (
    forecast_df.groupby("pickup_date")["predicted_daily"]
    .sum().reset_index(name="total_predicted_daily")
)

st.subheader("📊 Total Forecasted Pickups per Day (2025)")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(total_by_day["pickup_date"], total_by_day["total_predicted_daily"], lw=1.5)
ax.set_title("Forecasted Daily Demand for Food Hampers in 2025")
ax.set_xlabel("Date"); ax.set_ylabel("Predicted Pickups")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

st.success("Forecast complete and displayed.")

