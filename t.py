import streamlit as st
import pandas as pd
import plotly.express as px

# Page setup
st.set_page_config(page_title="📊 Demand Prediction Studio", layout="wide")
st.title("📦 Food Hamper Demand – Forecast & EDA Insights")

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
date_range = st.sidebar.date_input(
    "Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
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
    "📈 Forecast Trends", "🗺️ Demand Map", "📊 EDA Insights",
    "🧠 Model Insights", "🧪 Residuals", "📤 SHAP"
])

# Tab 1: Forecast Trends
with tabs[0]:
    st.header("📈 Forecast Trends (2025)")
    if view_mode == "Daily":
        chart_df = df_filt.copy()
        x = 'pickup_date'
    elif view_mode == "Weekly":
        chart_df = df_filt.groupby(['week', 'region'])['predicted_daily'].sum().reset_index()
        x = 'week'
    else:
        chart_df = df_filt.groupby(['month', 'region'])['predicted_daily'].sum().reset_index()
        x = 'month'
    fig = px.line(
        chart_df, x=x, y='predicted_daily', color='region',
        labels={x: view_mode, 'predicted_daily': 'Forecasted Pickups'},
        title=f"{view_mode} Forecasted Pickups per Region"
    )
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=1))
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Demand Map
with tabs[1]:
    st.header("🗺️ 2025 Average Forecast Map")
    # Use filtered data to compute average
    map_df = (
        df_filt.groupby('region')['predicted_daily']
        .mean().reset_index().rename(columns={'predicted_daily': 'avg_pickup'})
    )
    coords = {
        'Central': [53.5461, -113.4938], 'North Edmonton': [53.6081, -113.5035],
        'Northeast': [53.5820, -113.4190], 'South Edmonton': [53.4690, -113.5102],
        'Southeast': [53.4955, -113.4100], 'Far South': [53.4080, -113.5095],
        'West Edmonton': [53.5444, -113.6426]
    }
    map_df['lat'] = map_df['region'].map(lambda r: coords[r][0])
    map_df['lon'] = map_df['region'].map(lambda r: coords[r][1])
    map_df['level'] = map_df['avg_pickup'].apply(
        lambda v: 'Low' if v < 1.5 else 'Medium' if v < 2.5 else 'High'
    )
    color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    fig_map = px.scatter_mapbox(
        map_df, lat='lat', lon='lon', size='avg_pickup', color='level', hover_name='region',
        color_discrete_map=color_map, size_max=30, zoom=10, mapbox_style='open-street-map'
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown(
        "**Legend:** Red=High (>=2.5), Orange=Medium (1.5–2.5), Green=Low (<1.5)"
    )

# Tab 3: EDA Insights
with tabs[2]:
    st.header("📊 EDA Insights & Problem Statement")
    st.markdown(
        "**Problem Statement:**  \n> Identify geographic areas in Edmonton with higher or lower food hamper demand to support Islamic Family’s outreach and mobile distribution planning."
    )
    st.subheader("Key Findings From Data Inspection")
    st.markdown(
        "- Clients: 25,505 rows, 44 cols; Food Hampers: 16,605 rows, 39 cols.  \n"
        "- Core features: age, sex_new, dependents_qty, preferred_languages, latitude, longitude, dist_to_hub_km, daily_pickups.  \n"
        "- Behavioral: visit_count_90d, days_since_first_visit; Data Quality: missing ages/address JSON."
    )

# Tab 4: Model Insights
with tabs[3]:
    st.header("🧠 Model Diagnostics")
    st.image('images/feature_importance.png', caption='RF Feature Importance')
    st.image('images/acf.png', caption='ACF of Δ Daily Pickups')
    st.image('images/pacf.png', caption='PACF of Δ Daily Pickups')
    st.image('images/sarima.png', caption='14-Day SARIMA Forecast')

# Tab 5: Residuals
with tabs[4]:
    st.header("🧪 Residual Analysis")
    st.image('images/rf_interval.png', caption='Prediction Intervals – RF')
    st.image('images/decomposition.png', caption='STL Decomposition')
    st.image('images/residual_hist.png', caption='Residual Histogram')
    st.image('images/residual_fitted.png', caption='Residuals vs Fitted')

# Tab 6: SHAP Interpretation
with tabs[5]:
    st.header("📤 SHAP Interpretability")
    for img in ['shap1', 'shap2', 'shap4', 'shap5', 'shap6', 'shap7']:
        st.image(f'shap/{img}.png')
    st.markdown(
        "**Insights:** lag_1 drives forecasts; days_since_first_visit boosts pickups; day_index captures weekly patterns."
    )


