import streamlit as st
import pandas as pd
import numpy as np
from math import pi
import folium
from streamlit_folium import st_folium
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
    "📈 Forecast Trends", "🗺️ Demand Map", "📊 EDA Insights",
    "🧠 Model Insights", "🧪 Residuals", "📤 SHAP"
])

# Tab 1: Forecast Trends
with tabs[0]:
    st.header("📈 Forecast Trends (2025)")
    if view_mode == "Daily":
        data = df_filt.copy()
        x_col = 'pickup_date'
    elif view_mode == "Weekly":
        data = df_filt.groupby(['week','region'])['predicted_daily'].sum().reset_index()
        x_col = 'week'
    else:
        data = df_filt.groupby(['month','region'])['predicted_daily'].sum().reset_index()
        x_col = 'month'
    fig = px.line(
        data, x=x_col, y='predicted_daily', color='region',
        labels={x_col: view_mode, 'predicted_daily': 'Forecasted Pickups'},
        title=f"{view_mode} Forecasted Pickups per Region"
    )
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=1))
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Demand Map
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
    st.subheader("Key Findings From Data Inspection")
    st.markdown(
        "- Clients: 25,505 rows, 44 cols; Food Hampers: 16,605 rows, 39 cols.  \n"
        "- Core features: age, sex_new, dependents_qty, preferred_languages, latitude, longitude, dist_to_hub_km, daily_pickups.  \n"
        "- Behavioral: visit_count_90d, days_since_first_visit; Data Quality: missing ages/address JSON."
    )
    # Dynamic load EDA images from images folder
    eda_dir = 'images'
    eda_prefixes = ['stats', 'dependents_dist', 'lang_top10', 'revisit_dependents', 'pickup_age_group', 'revisit_flag']
    if os.path.isdir(eda_dir):
        eda_files = sorted([f for f in os.listdir(eda_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                            and any(f.lower().startswith(prefix) for prefix in eda_prefixes)])
        if eda_files:
            for img in eda_files:
                path = os.path.join(eda_dir, img)
                try:
                    st.image(path, caption=img)
                except Exception as e:
                    st.warning(f"Could not load EDA image {img}: {e}")
        else:
            st.info("No EDA images found in 'images' matching expected prefixes.")
    else:
        st.warning("Images folder 'images' not found. Please add your EDA visuals there.")
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
    st.image('images/shap4.png', caption='SHAP results')

    st.markdown(
        "**Insights:** lag_1 drives forecasts; days_since_first_visit boosts pickups; day_index captures weekly patterns."
    )


