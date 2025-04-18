import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium

# Page setup
st.set_page_config(page_title="📊 Demand Prediction Studio", layout="wide")
st.title("📦 Food Hamper Demand – Forecast & Client Insights")

# ─── Data Loading ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Historical actuals
    actual = pd.read_csv('region_client_df (1).csv', parse_dates=['pickup_date'])
    actual['pickup_date'] = actual['pickup_date'].dt.normalize()
    actual_2024 = actual[actual['pickup_date'].dt.year == 2024]
    actual_daily = (actual_2024
        .groupby(['pickup_date', 'region'])['daily_pickups']
        .sum()
        .reset_index()
        .rename(columns={'daily_pickups': 'actual_daily'}))
    # Align to 2025
    def safe_replace_year(dt):
        try: return dt.replace(year=2025)
        except: return pd.NaT
    actual_daily['aligned_date'] = actual_daily['pickup_date'].apply(safe_replace_year)
    actual_daily = actual_daily.dropna(subset=['aligned_date'])

    # Forecast data
    forecast = pd.read_csv('forecast_2025_rf.csv', parse_dates=['pickup_date'])
    forecast['pickup_date'] = forecast['pickup_date'].dt.normalize()
    forecast_daily = (forecast
        .groupby(['pickup_date', 'region'])['predicted_daily']
        .sum()
        .reset_index())
    forecast_daily['aligned_date'] = forecast_daily['pickup_date']

    # Merge for comparison
    df_compare = pd.merge(forecast_daily, actual_daily,
                          on=['aligned_date', 'region'], how='inner')
    df_compare = df_compare.rename(columns={
        'pickup_date_x': 'forecast_date',
        'pickup_date_y': 'actual_date'
    })
    # Add aggregates
    df_compare['week'] = df_compare['aligned_date'].dt.to_period('W').apply(lambda r: r.start_time)
    df_compare['month'] = df_compare['aligned_date'].dt.to_period('M').apply(lambda r: r.start_time)

    return df_compare, forecast

# Load data
df_compare, forecast_df = load_data()

# ─── Sidebar Filters ───────────────────────────────────────────────────────
st.sidebar.header("🔧 Filters & View Options")
# Date range
min_date = df_compare['aligned_date'].min().date()
max_date = df_compare['aligned_date'].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date],
                                   min_value=min_date, max_value=max_date)
# Granularity
view_mode = st.sidebar.radio("View by", ["Daily", "Weekly", "Monthly"])
# Regions
regions = df_compare['region'].unique().tolist()
selected_regions = st.sidebar.multiselect("Regions", regions, default=regions)

# Apply filters
df_filt = df_compare[
    (df_compare['aligned_date'] >= pd.to_datetime(date_range[0])) &
    (df_compare['aligned_date'] <= pd.to_datetime(date_range[1])) &
    (df_compare['region'].isin(selected_regions))
]

# ─── Tabs ─────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Forecast vs Actual",
    "🗺️ Demand Map",
    "🧠 Model Insights",
    "🧪 Residuals",
    "📤 SHAP",
    "👥 Client Insights"
])

# Tab 1: Forecast vs Actual
with tabs[0]:
    st.header("📈 Forecast vs 2024 Actuals")
    if view_mode == "Daily":
        chart_df = df_filt.copy(); x_col = 'aligned_date'
    elif view_mode == "Weekly":
        chart_df = df_filt.groupby(['week','region'])[['actual_daily','predicted_daily']].sum().reset_index(); x_col = 'week'
    else:
        chart_df = df_filt.groupby(['month','region'])[['actual_daily','predicted_daily']].sum().reset_index(); x_col = 'month'
    fig = px.line(chart_df, x=x_col,
                  y=['actual_daily','predicted_daily'], color='region',
                  labels={'actual_daily':'Actual','predicted_daily':'Forecast'},
                  title=f"{view_mode} Forecast vs Actual by Region")
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=1))
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Demand Map
with tabs[1]:
    st.header("🗺️ 2025 Forecast Map")
    avg = forecast_df.groupby('region')['predicted_daily'].mean().reset_index()
    def cat(x): return 'Low' if x<1.5 else 'Medium' if x<2.5 else 'High'
    avg['level'] = avg['predicted_daily'].apply(cat)
    color_map = {'Low':'green','Medium':'orange','High':'red'}
    coords = {
        'Central':[53.5461,-113.4938],'North Edmonton':[53.6081,-113.5035],
        'Northeast':[53.5820,-113.4190],'South Edmonton':[53.4690,-113.5102],
        'Southeast':[53.4955,-113.4100],'Far South':[53.4080,-113.5095],
        'West Edmonton':[53.5444,-113.6426]
    }
    m = folium.Map(location=[53.5461,-113.4938], zoom_start=10)
    for _,r in avg.iterrows():
        loc = coords.get(r['region'])
        if loc: folium.CircleMarker(location=loc,
                     radius=5+r['predicted_daily']*2,
                     color=color_map[r['level']],fill=True,fill_opacity=0.6,
                     popup=f"{r['region']}: {r['predicted_daily']:.1f} ({r['level']})").add_to(m)
    st_folium(m, width=800)
    with st.expander("Legend"):
        st.markdown("- **Red**: High≥2.5  \n- **Orange**: 1.5–2.5  \n- **Green**: <1.5")

# Tab 3: Model Insights
with tabs[2]:
    st.header("🧠 Model Diagnostics")
    st.image('images/feature_importance.png', caption='RF Feature Importance')
    st.image('images/acf.png', caption='ACF of Δ Daily Pickups')
    st.image('images/pacf.png', caption='PACF of Δ Daily Pickups')
    st.image('images/sarima.png', caption='14-Day SARIMA Forecast')

# Tab 4: Residuals
with tabs[3]:
    st.header("🧪 Residual Analysis")
    st.image('images/rf_interval.png', caption='Prediction Intervals – RF')
    st.image('images/decomposition.png', caption='STL Decomposition – Trend/Season/Noise')
    st.image('images/residual_hist.png', caption='Histogram of Residuals')
    st.image('images/residual_fitted.png', caption='Residuals vs Fitted Predictions')

# Tab 5: SHAP
with tabs[4]:
    st.header("📤 SHAP Interpretability")
    st.markdown("**Key SHAP Insights:**")
    st.markdown(
        "- **lag_1** drives most predictions: higher previous-day pickups increase outlook.  "
        "- **days_since_first_visit**: clients with longer histories tend to pick up more.  "
        "- **day_index** captures intra-week patterns.  "
    )
    shap_table = pd.DataFrame({
        'Feature': ['lag_1', 'days_since_first_visit', 'day_index', 'dow_sin', 'dow_cos', 'is_holiday'],
        'Mean |SHAP|':    [0.65,       0.09,                   0.05,         0.01,     0.005,      0.001]
    })
    st.table(shap_table)
   

# Tab 6: Client Insights
with tabs[5]:
    st.header("👥 Client Demographics & Insights")
    st.image('images/age_dist.png', caption='Age Distribution of Clients')
    st.image('images/dependents_dist.png', caption='Dependents Quantity Distribution')
    st.image('images/lang_top10.png', caption='Top 10 Primary Languages')
    st.image('images/revisit_dependents.png', caption='Revisit Rate by Dependents Group')
    st.image('images/pickup_age_group.png', caption='Pickup Rate by Age Group')
    st.image('images/revisit_flag.png', caption='Revisit Behavior (First vs Repeat)')
    st.image('images/collected_flag.png', caption='Client Collected Flag')
    st.image('images/dependents_box.png', caption='Boxplot of Dependents Qty')
    
    st.subheader("🔗 Feature Correlation Heatmap")
    num_cols = actual_df.select_dtypes(include='number').columns
    corr = actual_df[num_cols].corr()
    corr_fig = px.imshow(corr, text_auto=True, aspect='auto', title='Correlation Heatmap')
    st.plotly_chart(corr_fig, use_container_width=True)

