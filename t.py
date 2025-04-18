import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium

# Page setup
st.set_page_config(page_title="📊 Demand Prediction Studio", layout="wide")
st.title("📦 Food Hamper Demand – Forecast & EDA Insights")

# ─── Data Loading ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Forecast data
    forecast = pd.read_csv('forecast_2025_rf.csv', parse_dates=['pickup_date'])
    forecast['pickup_date'] = forecast['pickup_date'].dt.normalize()
    # Add week/month for grouping
    forecast['week'] = forecast['pickup_date'].dt.to_period('W').apply(lambda r: r.start_time)
    forecast['month'] = forecast['pickup_date'].dt.to_period('M').apply(lambda r: r.start_time)
    return forecast

# Load data
forecast_df = load_data()

# ─── Sidebar Filters ───────────────────────────────────────────────────────
st.sidebar.header("🔧 Filters & View Options")
# Date range picker
min_date = forecast_df['pickup_date'].min().date()
max_date = forecast_df['pickup_date'].max().date()
date_range = st.sidebar.date_input(
    "Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
# Granularity
view_mode = st.sidebar.radio("View by", ["Daily", "Weekly", "Monthly"])
# Regions
regions = forecast_df['region'].unique().tolist()
selected_regions = st.sidebar.multiselect("Regions", regions, default=regions)

# Filter forecast data based on inputs
df_filt = forecast_df[
    (forecast_df['pickup_date'] >= pd.to_datetime(date_range[0])) &
    (forecast_df['pickup_date'] <= pd.to_datetime(date_range[1])) &
    (forecast_df['region'].isin(selected_regions))
]

# ─── Tabs ─────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Forecast Trends",
    "🗺️ Demand Map",
    "📊 EDA Insights",
    "🧠 Model Insights",
    "🧪 Residuals",
    "📤 SHAP"
])

# Tab 1: Forecast Trends
with tabs[0]:
    st.header("📈 Forecast Trends (2025)")
    if view_mode == "Daily":
        chart_df = df_filt.copy()
        x_col = 'pickup_date'
        y_col = 'predicted_daily'
    elif view_mode == "Weekly":
        chart_df = df_filt.groupby(['week', 'region'])['predicted_daily'].sum().reset_index()
        x_col = 'week'
        y_col = 'predicted_daily'
    else:
        chart_df = df_filt.groupby(['month', 'region'])['predicted_daily'].sum().reset_index()
        x_col = 'month'
        y_col = 'predicted_daily'
    fig = px.line(
        chart_df,
        x=x_col,
        y=y_col,
        color='region',
        labels={x_col: view_mode, y_col: 'Forecasted Pickups'},
        title=f"{view_mode} Forecasted Pickups per Region"
    )
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=1))
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Demand Map
with tabs[1]:
    st.header("🗺️ 2025 Forecast Map")
    avg = forecast_df.groupby('region')['predicted_daily'].mean().reset_index()
    # Categorize demand levels
    def categorize(val):
        if val < 1.5:
            return 'Low'
        elif val < 2.5:
            return 'Medium'
        else:
            return 'High'
    avg['level'] = avg['predicted_daily'].apply(categorize)
    # Color and size settings
    color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    coords = {
        'Central': [53.5461, -113.4938],
        'North Edmonton': [53.6081, -113.5035],
        'Northeast': [53.5820, -113.4190],
        'South Edmonton': [53.4690, -113.5102],
        'Southeast': [53.4955, -113.4100],
        'Far South': [53.4080, -113.5095],
        'West Edmonton': [53.5444, -113.6426],
    }
    m = folium.Map(location=[53.5461, -113.4938], zoom_start=10)
    for _, row in avg.iterrows():
        loc = coords.get(row['region'])
        if not loc:
            continue
        folium.CircleMarker(
            location=loc,
            radius=5 + row['predicted_daily'] * 2,
            color='black',             # border color
            fill=True,
            fill_color=color_map[row['level']],  # fill color
            fill_opacity=0.7,
            popup=f"{row['region']}: {row['predicted_daily']:.2f} avg pickups ({row['level']})"
        ).add_to(m)
    st_folium(m, width=800)
    with st.expander("Legend"):
        st.markdown(
            "- **Red**: High (>=2.5 avg)  \n"
            "- **Orange**: Medium (1.5–2.5 avg)  \n"
            "- **Green**: Low (<1.5 avg)"
        )


# Tab 3: EDA Insights
with tabs[2]:
    st.header("📊 EDA Insights & Problem Statement")
    st.markdown("**Problem Statement:**  \n> Identify geographic areas in Edmonton with higher or lower food hamper demand to support Islamic Family’s outreach and mobile distribution planning.")
    st.subheader("Key Findings from Initial Data Inspection")
    st.markdown(
        "- **Clients dataset**: 25,505 rows, 44 columns; one record per unique client.  \n"
        "- **Food Hampers dataset**: 16,605 rows, 39 columns; one record per pickup appointment.  \n"
        "- **Primary client features**: `age`, `sex_new`, `dependents_qty`, `preferred_languages`.  \n"
        "- **Geospatial variables**: `FSA`, `final_FSA`, `latitude`, `longitude`, `dist_to_hub_km`.  \n"
        "- **Temporal and target**: `pickup_date`, `daily_pickups`, `target_pickup_count_14d`.  \n"
        "- **Data quality notes**: Some missing ages/birthdates; address fields in JSON; empty emergency contact columns.  \n"
        "- **Behavioral features** computed: `visit_count_90d`, `days_since_first_visit`."
    )
    st.image('images/stats.png', caption='Stats of Clients')
    st.image('images/dependents_dist.png', caption='Dependents Quantity Distribution')
    st.image('images/lang_top10.png', caption='Top 10 Primary Languages')
    st.image('images/revisit_dependents.png', caption='Revisit Rate by Dependents Group')
    st.image('images/pickup_age_group.png', caption='Pickup Rate by Age Group')
    st.image('images/revisit_flag.png', caption='Revisit Behavior (First vs Repeat)')
   
    

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
    st.image('images/decomposition.png', caption='STL Decomposition – Trend/Season/Noise')
    st.image('images/residual_hist.png', caption='Histogram of Residuals')
    st.image('images/residual_fitted.png', caption='Residuals vs Fitted Predictions')

# Tab 6: SHAP
with tabs[5]:
    st.header("📤 SHAP Interpretability")
    st.image('images/shap4.png', caption='SHAP Interaction Effects')
    st.markdown("**Key SHAP Insights:**  \n"
                "+ **lag_1**: Previous day pickups strongly increase forecast.  \n"
                "+ **days_since_first_visit**: Longer client history boosts pickup rate.  \n"
                "+ **day_index**: Captures weekly patterns effectively.")


