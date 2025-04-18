import streamlit as st
import pandas as pd
import numpy as np
from math import pi
import folium
from streamlit_folium import st_folium
import plotly.express as px
import os  # needed for dynamic image loading

# ─── Data Loading ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('forecast_2025_rf.csv', parse_dates=['pickup_date'])
    df['pickup_date'] = df['pickup_date'].dt.normalize()
    df['week'] = df['pickup_date'].dt.to_period('W').apply(lambda r: r.start_time)
    df['month'] = df['pickup_date'].dt.to_period('M').apply(lambda r: r.start_time)
    return df

# Load data
df = load_data()

# ─── Sidebar Filters ───────────────────────────────────────────────────────
st.sidebar.header("🔧 Filters & View Options")
min_date, max_date = df['pickup_date'].min().date(), df['pickup_date'].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
view_mode = st.sidebar.radio("View by", ["Daily", "Weekly", "Monthly"])
regions = df['region'].unique().tolist()
selected_regions = st.sidebar.multiselect("Regions", regions, default=regions)

# Apply filters
df_filt = df[
    (df['pickup_date'] >= pd.to_datetime(date_range[0])) &
    (df['pickup_date'] <= pd.to_datetime(date_range[1])) &
    (df['region'].isin(selected_regions))
]

# ─── Tabs ─────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Forecast Trends",
    "🗺️ Demand Map",
    "📊 EDA Insights",
    "📊 Model Comparison",
    "🧠 Model Insights",
    "🧪 Residuals",
    "📤 SHAP"
])

# Tab 1: Forecast Trends
with tabs[0]:
    st.header("📈 Forecast Trends (2025)")
    # Smoothing option for noisy daily lines
    smooth = st.sidebar.checkbox("Apply 7-day smoothing", value=False, help="Smooth daily trends with a 7-day moving average")

    if view_mode == "Daily":
        data = df_filt.copy()
        x_col = 'pickup_date'
        y_col = 'predicted_daily'
        if smooth:
            # aggregate by date & region then smooth
            tmp = data.groupby(['pickup_date','region'])['predicted_daily'].sum().reset_index()
            tmp['smoothed'] = tmp.groupby('region')['predicted_daily']\
                .transform(lambda x: x.rolling(7, min_periods=1).mean())
            data = tmp
            y_col = 'smoothed'
    elif view_mode == "Weekly":
        data = df_filt.groupby(['week','region'])['predicted_daily'].sum().reset_index()
        x_col = 'week'
        y_col = 'predicted_daily'
    else:
        data = df_filt.groupby(['month','region'])['predicted_daily'].sum().reset_index()
        x_col = 'month'
        y_col = 'predicted_daily'

    fig = px.line(
        data,
        x=x_col,
        y=y_col,
        color='region',
        line_shape='spline' if smooth and view_mode=='Daily' else 'linear',
        labels={x_col: view_mode, y_col: 'Forecasted Pickups'},
        title=f"{view_mode} Forecasted Pickups per Region"
    )
    fig.update_traces(mode='lines')
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=1))
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Demand Map
with tabs[1]:
    st.header(f"🗺️ 2025 Forecast Map ({view_mode})")
    # Aggregate based on view_mode
    if view_mode == 'Daily':
        agg = df_filt.groupby('region')['predicted_daily'].mean().reset_index()
    elif view_mode == 'Weekly':
        weekly = df_filt.groupby(['week','region'])['predicted_daily'].sum().reset_index()
        agg = weekly.groupby('region')['predicted_daily'].mean().reset_index()
    else:
        monthly = df_filt.groupby(['month','region'])['predicted_daily'].sum().reset_index()
        agg = monthly.groupby('region')['predicted_daily'].mean().reset_index()
    agg.columns = ['region', 'avg_pickup']

    # Dynamic thresholds based on percentiles
    q1, q2 = np.percentile(agg['avg_pickup'], [33, 66])
    agg['level'] = agg['avg_pickup'].apply(lambda v: 'Low' if v < q1 else 'Medium' if v < q2 else 'High')
    color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}

    coords = {
        'Central': [53.5461, -113.4938], 'North Edmonton': [53.6081, -113.5035],
        'Northeast': [53.5820, -113.4190], 'South Edmonton': [53.4690, -113.5102],
        'Southeast': [53.4955, -113.4100], 'Far South': [53.4080, -113.5095],
        'West Edmonton': [53.5444, -113.6426]
    }

    # Scale circle radii with square-root scaling, max 5000 m
    max_radius = 5000
    max_val = agg['avg_pickup'].max()
    agg['radius'] = np.sqrt(agg['avg_pickup'] / max_val) * max_radius
    agg['area_km2'] = (pi * (agg['radius']/1000)**2).round(2)

    m = folium.Map(location=[53.5461, -113.4938], zoom_start=10)
    for _, row in agg.iterrows():
        loc = coords.get(row['region'])
        if not loc: continue
        folium.Circle(
            location=loc,
            radius=row['radius'],
            color=color_map[row['level']],
            fill=True, fill_color=color_map[row['level']], fill_opacity=0.5,
            popup=(f"{row['region']}<br>Avg: {row['avg_pickup']:.2f}<br>"
                   f"Level: {row['level']}<br>Area: {row['area_km2']} km²")
        ).add_to(m)
    st_folium(m, width=800)
    st.markdown(
        f"**Thresholds:** Low < {q1:.2f}, Medium < {q2:.2f}, High ≥ {q2:.2f}<br>"
        f"**Area:** Circle radius sqrt-scaled, with max {max_radius/1000:.1f} km corresponding to largest value."
    , unsafe_allow_html=True)
# Tab 3: EDA Insights
with tabs[2]:
    st.header("📊 EDA Insights & Problem Statement")
    st.markdown(
        "**Problem Statement:**  \n> Identify geographic areas in Edmonton with higher or lower food hamper demand to support Islamic Family’s outreach and mobile distribution planning."
    )
    # Additional Data Overview
    st.subheader("Data Overview & Summary Statistics")
    st.markdown(
        f"- Total forecast records: {len(df)}  \n"
        f"- Date range: {min_date} to {max_date}  \n"
        f"- Regions covered: {len(regions)}  \n"
    )
    summary = df.groupby('region')['predicted_daily'].agg(
        Total='sum', Average='mean', Minimum='min', Maximum='max'
    ).round(2).reset_index()
    st.dataframe(summary, use_container_width=True)
    # Heatmap: Monthly average forecasted pickups per region
    st.subheader("📈 Monthly Avg Forecast Heatmap")
    # Prepare pivot table
    heat_df = df.groupby([df['pickup_date'].dt.to_period('M').apply(lambda r: r.start_time), 'region'])['predicted_daily']\
        .mean().round(2).reset_index()
    heat_pivot = heat_df.pivot(index='region', columns='pickup_date', values='predicted_daily')
    fig_heat = px.imshow(
        heat_pivot,
        labels=dict(x='Month', y='Region', color='Avg Pickups'),
        x=heat_pivot.columns,
        y=heat_pivot.index,
        aspect='auto',
        title='Monthly Avg Forecasted Pickups Heatmap'
    )
    fig_heat.update_xaxes(tickangle=45)
    st.plotly_chart(fig_heat, use_container_width=True)
    st.subheader("Key Findings From Data Inspection")
    st.markdown(
        "- Core features: age, sex_new, dependents_qty, preferred_languages, latitude, longitude, dist_to_hub_km, daily_pickups.  \n"
        "- Behavioral: visit_count_90d, days_since_first_visit; Data Quality: missing ages/address JSON."
    )
        # Static EDA visuals
    eda_images = [
        ('images/stats.png', 'Stats of Clients'),
        ('images/dependents.png', 'Dependents Quantity Distribution'),
        ('images/lang_top10.png', 'Top 10 Primary Languages'),
        ('images/revisit_dependants.png', 'Revisit Rate by Dependents Group'),
        ('images/pickup_age_group.png', 'Pickup Rate by Age Group')
    ]
    for img_path, caption in eda_images:
        if os.path.exists(img_path):
            try:
                st.image(img_path, caption=caption)
            except Exception as e:
                st.warning(f"Error loading {img_path}: {e}")
        else:
            st.warning(f"EDA image not found: {img_path}")
# Tab 4: Model Comparison
with tabs[3]:
    st.header("📊 Model Comparison & RMSE Metrics")
    st.markdown("**Why We Don’t Need Encoding or Normalization (Right Now):**")
    st.markdown(
        "All input features are numeric lagged or rolling counts.  \n"
        "Tree-based models are scale-invariant, so encoding and normalization aren’t required for RF and XGB models."
    )
    st.subheader("Static Region‑level RMSEs")
    rmse_static = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost'],
        'Region-level RMSE': [0.7695657763249005, 0.8783934389922908]
    })
    st.table(rmse_static)
    st.subheader("Aggregated Daily RMSEs")
    rmse_daily = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'SARIMAX', 'Prophet'],
        'Daily RMSE': [102.6961924148502, 102.69619181108908, 104.56170104736358, 166.03155426794353]
    })
    st.table(rmse_daily)
    st.markdown(
        "**Note:** If categorical or scale-sensitive models are introduced, we’d incorporate encoding and scaling then."
    )
# Tab 4: Model Insights
with tabs[4]:
    st.header("🧠 Model Diagnostics")
    st.image('images/feature_importance.png', caption='RF Feature Importance')
    st.image('images/acf.png', caption='ACF of Δ Daily Pickups')
    st.image('images/pacf.png', caption='PACF of Δ Daily Pickups')
    st.image('images/sarima.png', caption='14-Day SARIMA Forecast')

# Tab 5: Residuals
with tabs[5]:
    st.header("🧪 Residual Analysis")
    st.image('images/rf_interval.png', caption='Prediction Intervals – RF')
    st.image('images/decomposition.png', caption='STL Decomposition')
    st.image('images/residual_hist.png', caption='Residual Histogram')
    st.image('images/residual_fitted.png', caption='Residuals vs Fitted')

# Tab 6: SHAP Interpretation
with tabs[6]:
    st.header("📤 SHAP Interpretability")
    st.image('images/shap4.png', caption='SHAP results')

    st.markdown(
        "**Insights:** lag_1 drives forecasts; days_since_first_visit boosts pickups; day_index captures weekly patterns."
    )

