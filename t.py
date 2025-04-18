import streamlit as st
import pandas as pd
import plotly.express as px

# ─── Page Configuration ───────────────────────────────────────────────
st.set_page_config(page_title="📈 2025 Demand Forecast", layout="wide")
st.title("📦 2025 Food Hamper Demand Forecast Viewer")

# ─── Load and Prepare Data ─────────────────────────────────────────────
@st.cache_data
def load_forecast(path):
    df = pd.read_csv(path, parse_dates=["pickup_date"])
    df = df.dropna(subset=["pickup_date", "region", "daily_pickups"])
    return df

uploaded = st.file_uploader("Upload Forecast CSV", type="csv")
if uploaded:
    forecast_df = load_forecast(uploaded)

    # Sidebar Controls
    st.sidebar.header("🔍 Filter Options")
    all_regions = sorted(forecast_df["region"].unique())
    selected_regions = st.sidebar.multiselect("Select Region(s):", all_regions, default=all_regions)

    # Filter by region
    filtered_df = forecast_df[forecast_df["region"].isin(selected_regions)].copy()

    # Tabs for Daily, Weekly, Monthly
    tab1, tab2, tab3 = st.tabs(["📅 Daily Forecast", "📊 Weekly Aggregation", "📆 Monthly Aggregation"])

    # ── Daily Forecast ────────────────────────────────────────────────
    with tab1:
        st.subheader("📅 Daily Forecast")
        daily_fig = px.line(filtered_df, x="pickup_date", y="daily_pickups", color="region",
                            labels={"pickup_date": "Date", "daily_pickups": "Forecasted Pickups"},
                            title="Daily Forecasted Pickups per Region")
        st.plotly_chart(daily_fig, use_container_width=True)

    # ── Weekly Aggregation ────────────────────────────────────────────
    with tab2:
        st.subheader("📊 Weekly Forecast (Sum of Daily Pickups)")
        filtered_df["week"] = filtered_df["pickup_date"].dt.to_period("W").apply(lambda r: r.start_time)
        weekly_df = filtered_df.groupby(["week", "region"])["daily_pickups"].sum().reset_index()

        weekly_fig = px.line(weekly_df, x="week", y="daily_pickups", color="region",
                             labels={"week": "Week", "daily_pickups": "Weekly Total Pickups"},
                             title="Weekly Aggregated Forecast per Region")
        st.plotly_chart(weekly_fig, use_container_width=True)

    # ── Monthly Aggregation ───────────────────────────────────────────
    with tab3:
        st.subheader("📆 Monthly Forecast (Sum of Daily Pickups)")
        filtered_df["month"] = filtered_df["pickup_date"].dt.to_period("M").apply(lambda r: r.start_time)
        monthly_df = filtered_df.groupby(["month", "region"])["daily_pickups"].sum().reset_index()

        monthly_fig = px.line(monthly_df, x="month", y="daily_pickups", color="region",
                              labels={"month": "Month", "daily_pickups": "Monthly Total Pickups"},
                              title="Monthly Aggregated Forecast per Region")
        st.plotly_chart(monthly_fig, use_container_width=True)

    # ─── Summary ──────────────────────────────────────────────────────
    st.markdown("### 📌 Forecast Summary Table")
    summary = filtered_df.groupby("region")["daily_pickups"].sum().sort_values(ascending=False).reset_index()
    summary.columns = ["Region", "Total Forecasted Pickups (2025)"]
    st.dataframe(summary, use_container_width=True)

else:
    st.info("Please upload a forecast CSV file to begin.")

