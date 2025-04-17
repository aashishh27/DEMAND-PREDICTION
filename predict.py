import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk

# 1) Load & cache data + model
@st.cache_data
def load_assets():
    df = pd.read_csv(
        "region_client_df (1).csv",
        parse_dates=["pickup_date", "first_visit_date"]
    )
    df = df.sort_values(["region", "pickup_date"]).reset_index(drop=True)
    model = joblib.load("optimized_random_forest.joblib")
    return df, model

df, model = load_assets()

# 2) Forecast function (cached) — includes lat/lon in output
@st.cache_data(show_spinner=False)
def forecast_2025(df, _model):
    results = []
    static = (
        df.groupby("region")
          [["latitude", "longitude", "dist_to_hub_km", "fsa_cluster"]]
          .first()
          .reset_index()
    )

    for reg in df.region.unique():
        hist = df[df.region == reg]
        daily = hist.set_index("pickup_date")["daily_pickups"].astype(float).copy()
        future = pd.date_range(daily.index.max() + pd.Timedelta(1), "2025-12-31", freq="D")

        for date in future:
            lag_1 = daily.get(date - pd.Timedelta(1), 0.0)
            win7  = daily.loc[date - pd.Timedelta(7):date - pd.Timedelta(1)].mean()
            win14 = daily.loc[date - pd.Timedelta(14):date - pd.Timedelta(1)].mean()
            win30 = daily.loc[date - pd.Timedelta(30):date - pd.Timedelta(1)].mean()
            days_since = (date - hist.first_visit_date.min()).days

            feat = {
                "lag_1": lag_1,
                "daily_pickups": lag_1,
                "rolling_7d": win7,
                "rolling_14d": win14,
                "rolling_30d": win30,
                "days_since_first_visit": days_since
            }
            feat.update(static[static.region == reg].iloc[0].to_dict())

            Xp = pd.DataFrame([feat])[_model.feature_names_in_]
            pred = _model.predict(Xp)[0]

            # ✅ append prediction with coordinates
            results.append({
                "pickup_date": date,
                "region": reg,
                "predicted_pickups": pred,
                "latitude": feat["latitude"],
                "longitude": feat["longitude"]
            })

            daily.at[date] = pred

    return pd.DataFrame(results)

# 3) Streamlit UI
st.title("🔮 2025 Forecast with Trend, Map & Filters")

# Sidebar filters
st.sidebar.header("Filters")
all_regions = list(df.region.unique())
selected_regions = st.sidebar.multiselect(
    "Select region(s)", all_regions, default=all_regions
)
start_date, end_date = st.sidebar.date_input(
    "Date range", 
    value=(pd.to_datetime("2025-01-01"), pd.to_datetime("2025-05-16")),
    min_value=pd.to_datetime("2025-01-01"),
    max_value=pd.to_datetime("2025-12-31")
)

if st.button("Run 2025 Forecast"):
    with st.spinner("Generating forecasts…"):
        preds_2025 = forecast_2025(df, model)
    st.success("Forecast complete!")

    # Apply filters
    mask = (
        preds_2025["region"].isin(selected_regions) &
        (preds_2025["pickup_date"] >= pd.to_datetime(start_date)) &
        (preds_2025["pickup_date"] <= pd.to_datetime(end_date))
    )
    filtered = preds_2025[mask]

    # --- Trend line chart: total daily pickups ---
    daily_totals = (
        filtered
        .groupby("pickup_date")["predicted_pickups"]
        .sum()
        .rename("Total Pickups")
        .reset_index()
        .set_index("pickup_date")
    )
    st.subheader("📈 Daily Total Pickups for 2025 (Filtered)")
    st.line_chart(daily_totals)

    # --- Map: total 2025 pickups by region centroid ---
    centroids = (
        filtered
        .groupby("region")
        .agg({
            "latitude": "first",
            "longitude": "first",
            "predicted_pickups": "sum"
        })
        .reset_index()
    )
    centroids = centroids.rename(columns={
        "latitude": "lat",
        "longitude": "lon",
        "predicted_pickups": "total_2025_pickups"
    })

    st.subheader("🗺️ Map of Total 2025 Pickups by Region (Filtered)")
    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=centroids.lat.mean(),
            longitude=centroids.lon.mean(),
            zoom=10
        ),
        layers=[
            pdk.Layer(
                "ColumnLayer",
                data=centroids,
                get_position=["lon", "lat"],
                get_elevation="total_2025_pickups",
                elevation_scale=0.1,
                radius=500,
                get_fill_color="[255 - total_2025_pickups * 0.02, 100, total_2025_pickups * 0.02, 160]"
            )
        ]
    )
    st.pydeck_chart(deck)

