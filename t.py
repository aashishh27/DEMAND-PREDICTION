import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium

# ─── App Config ─────────────────────────────────────────────────────────
st.set_page_config(page_title="📦 2025 Forecast Dashboard", layout="wide")
st.title("📅 2025 Hamper Demand Forecast")

# ─── Load Data ──────────────────────────────────────────────────────────
@st.cache_data
def load_forecast():
    df = pd.read_csv("forecast_2025_rf.csv", parse_dates=["pickup_date"])
    df["pickup_date"] = df["pickup_date"].dt.normalize()
    df["week"] = df["pickup_date"].dt.to_period("W").apply(lambda r: r.start_time)
    df["month"] = df["pickup_date"].dt.to_period("M").apply(lambda r: r.start_time)
    return df

df = load_forecast()

# ─── Sidebar Controls ───────────────────────────────────────────────────
st.sidebar.header("🔧 Filter Forecast")

# Date range picker
min_date, max_date = df["pickup_date"].min(), df["pickup_date"].max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Granularity toggle
view_mode = st.sidebar.radio("View by", ["Daily", "Weekly", "Monthly"])

# Region selector
regions = df["region"].unique().tolist()
selected_regions = st.sidebar.multiselect("Select Regions", regions, default=regions)

# ─── Filter Data Based on Inputs ────────────────────────────────────────
df_filtered = df[(df["pickup_date"] >= date_range[0]) & (df["pickup_date"] <= date_range[1])]
df_filtered = df_filtered[df_filtered["region"].isin(selected_regions)]

# ─── Dynamic Chart ──────────────────────────────────────────────────────
st.subheader(f"📈 {view_mode} Forecast Trend")

if view_mode == "Daily":
    chart_data = df_filtered.copy()
    x_col = "pickup_date"
elif view_mode == "Weekly":
    chart_data = df_filtered.groupby(["week", "region"])["predicted_daily"].sum().reset_index()
    x_col = "week"
else:  # Monthly
    chart_data = df_filtered.groupby(["month", "region"])["predicted_daily"].sum().reset_index()
    x_col = "month"

fig = px.line(chart_data, x=x_col, y="predicted_daily", color="region",
              labels={x_col: view_mode, "predicted_daily": f"{view_mode} Pickups"},
              title=f"{view_mode} Forecasted Pickups per Region")
st.plotly_chart(fig, use_container_width=True)

# ─── Summary Table ──────────────────────────────────────────────────────
st.subheader("📊 Region Forecast Summary (Full 2025)")

summary = df[df["region"].isin(selected_regions)].groupby("region")["predicted_daily"].agg(
    Total_Pickups="sum", Average_Daily="mean", Max_Day="max", Min_Day="min").reset_index()
st.dataframe(summary, use_container_width=True)

# ─── Map View ───────────────────────────────────────────────────────────
st.subheader("🌍 Map of Average Forecasted Pickups per Region")

region_coords = {
    "Central":        [53.5461, -113.4938],
    "North Edmonton": [53.6081, -113.5035],
    "Northeast":      [53.5820, -113.4190],
    "South Edmonton": [53.4690, -113.5102],
    "Southeast":      [53.4955, -113.4100],
    "Far South":      [53.4080, -113.5095],
    "West Edmonton":  [53.5444, -113.6426],
}

avg_demand = df[df["region"].isin(selected_regions)].groupby("region")["predicted_daily"].mean().reset_index()

m = folium.Map(location=[53.5461, -113.4938], zoom_start=10)
for _, row in avg_demand.iterrows():
    region = row["region"]
    demand = row["predicted_daily"]
    coords = region_coords.get(region)
    if coords:
        folium.CircleMarker(
            location=coords,
            radius=5 + demand * 2,
            popup=f"{region}: {demand:.2f} avg pickups/day",
            color="blue",
            fill=True,
            fill_opacity=0.6
        ).add_to(m)

st_data = st_folium(m, width=700)

# ─── Footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built by Aashish • Powered by Streamlit, Plotly, and Folium")

