# streamlit_demand_prediction_studio.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="🌐 Geospatial Demand Prediction Studio", layout="wide")

# ─── Helper Functions & Initialization ─────────────────────────────────────────

@st.cache_data
def load_data(path="region_client_df (1).csv"):
    df = pd.read_csv(path, parse_dates=["pickup_date"])
    return df

@st.cache_resource
def load_model(path="optimized_random_forest.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def forecast(model, df, horizon):
    # Simple rolling‑mean forecast as placeholder—replace with your true forecasting logic
    last_values = df.sort_values("pickup_date")["target_pickup_count_14d"].values[-horizon:]
    dates = [df.pickup_date.max() + pd.Timedelta(days=i+1) for i in range(horizon)]
    return pd.DataFrame({
        "date": dates,
        "forecast": model.predict(df.tail(horizon).drop(["pickup_date","quantity"], axis=1)),
        "actual": [np.nan]*horizon,
        "lower_ci": last_values * 0.9,
        "upper_ci": last_values * 1.1
    })

@st.cache_resource
def init_shap(model):
    # For RandomForest / other tree models, use TreeExplainer
    explainer = shap.TreeExplainer(model)
    return explainer

explainer = init_shap(model)
X = df_filt.select_dtypes(np.number)
shap_vals = explainer.shap_values(X)

def init_rag(chroma_dir="chroma_db"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ Please set OPENAI_API_KEY")
        st.stop()
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    llm = ChatOpenAI(temperature=0)
    return RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# ─── Load Everything ──────────────────────────────────────────────────────────────

df = load_data()
model = load_model()
shap_vals = init_shap(model, df)
qa_chain = init_rag()

# ─── Sidebar Controls ────────────────────────────────────────────────────────────

st.sidebar.header("Filters")
min_date, max_date = df.pickup_date.min(), df.pickup_date.max()
start_date, end_date = st.sidebar.date_input(
    "Date Range", [min_date, min_date + pd.Timedelta(days=13)],
    min_value=min_date, max_value=max_date
)
start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)

regions = sorted(df.region.unique())
selected_regions = st.sidebar.multiselect("Regions", regions, default=regions)

dep_min, dep_max = int(df.dependents_qty.min()), int(df.dependents_qty.max())
dependents_range = st.sidebar.slider(
    "Dependents", dep_min, dep_max, (dep_min, dep_max)
)

house_types = sorted(df.household.unique())
household_map = {0.0: "Single‑Person", 1.0: "Multi‑Person"}
selected_households = st.sidebar.multiselect(
    "Household Type",
    [household_map[h] for h in house_types],
    default=[household_map[h] for h in house_types]
)
# remap back to numeric codes
sel_house_codes = [k for k,v in household_map.items() if v in selected_households]

revisit_only = st.sidebar.selectbox(
    "Client Status", ["All", "First‑Timers", "Returning"]
)

horizon = st.sidebar.slider("Forecast Horizon (days ahead)", 7, 60, 14)
mode = st.sidebar.radio("Mode", ["Exploration", "Optimization", "Story"])

# ─── Data Filtering ───────────────────────────────────────────────────────────────

mask = (
    df.pickup_date.between(start_date, end_date) &
    df.region.isin(selected_regions) &
    df.dependents_qty.between(*dependents_range) &
    df.household.isin(sel_house_codes)
)
if revisit_only == "First‑Timers":
    mask &= df.revisit == 0
elif revisit_only == "Returning":
    mask &= df.revisit == 1
df_filt = df.loc[mask].reset_index(drop=True)

# ─── Tabs ────────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "Overview & KPIs","EDA","Features",
    "Model Comparison","Forecast & Performance",
    "XAI","Geospatial","Optimization Playground",
    "Story Mode","Chatbot"
])

# ── Tab 1: Overview & KPIs ───────────────────────────────────────────────────────
with tabs[0]:
    st.header("📦 Overview & KPIs")
    st.markdown(
        "**Geospatial Analysis for Demand Prediction**\n\n"
        "Predict regions of rising or falling hamper demand in Edmonton to guide outreach."
    )
    demand = int(df_filt.quantity.sum())
    days = (end_date - start_date).days + 1
    prev_start = start_date - pd.Timedelta(days=days)
    prev_end = start_date - pd.Timedelta(days=1)
    prev_mask = (
        df.pickup_date.between(prev_start, prev_end) &
        df.region.isin(selected_regions)
    )
    prev_demand = int(df.loc[prev_mask,"quantity"].sum())
    col1, col2 = st.columns(2)
    col1.metric(f"{start_date.date()} → {end_date.date()}", f"{demand:,}")
    delta = (demand - prev_demand) / prev_demand if prev_demand else None
    col2.metric(f"Change vs {prev_start.date()}→{prev_end.date()}",
                f"{demand - prev_demand:,}",
                f"{delta:.1%}" if delta is not None else "N/A")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=demand,
        delta={'reference': prev_demand, 'relative': True},
        title={'text': "Demand Momentum"}
    ))
    st.plotly_chart(fig, use_container_width=True, key="kpi_gauge")

# ── Tab 2: EDA ──────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("📊 Exploratory Data Analysis")
    bins = max(10, (end_date - start_date).days)
    hist = px.histogram(df_filt, x="pickup_date", nbins=bins)
    hist.update_layout(title="Pickup Distribution Over Time")
    st.plotly_chart(hist, use_container_width=True)
    corr = df_filt.select_dtypes("number").corr()
    heat = px.imshow(corr, text_auto=True)
    heat.update_layout(title="Feature Correlation Matrix")
    st.plotly_chart(heat, use_container_width=True)

# ── Tab 3: Features ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("🛠️ Feature Engineering Insights")
    feat_imp = pd.Series(model.feature_importances_,
                         index=df_filt.select_dtypes("number").columns).sort_values(ascending=False)
    imp_df = feat_imp.head(10).reset_index()
    imp_df.columns = ["Feature","Importance"]
    st.bar_chart(imp_df.set_index("Feature"), use_container_width=True)
    sm = px.scatter_matrix(df_filt, dimensions=feat_imp.head(3).index.tolist())
    sm.update_layout(title="Top 3 Feature Scatter Matrix")
    st.plotly_chart(sm, use_container_width=True)

# ── Tab 4: Model Comparison ─────────────────────────────────────────────────────
with tabs[3]:
    st.header("🤖 Model Comparison")
    # These values come from your optimization notebook
    rf_cv_rmse, rf_test_rmse = 1.775, 2.301
    xgb_cv_rmse, xgb_test_rmse = 2.080, 2.613
    comp_df = pd.DataFrame({
        "Random Forest":[rf_cv_rmse, rf_test_rmse],
        "XGBoost":[xgb_cv_rmse, xgb_test_rmse]
    }, index=["CV RMSE","Test RMSE"])
    st.bar_chart(comp_df, use_container_width=True)
    st.caption("Lower RMSE indicates better performance")

# ── Tab 5: Forecast & Performance ────────────────────────────────────────────────
with tabs[4]:
    st.header("⏱️ Forecast & Performance")
    fc_df = forecast(model, df_filt, horizon)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc_df.date, y=fc_df.forecast, name="Forecast"))
    fig.add_trace(go.Scatter(x=fc_df.date, y=fc_df.lower_ci, name="Lower CI", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=fc_df.date, y=fc_df.upper_ci, name="Upper CI", fill="tonexty", opacity=0.2))
    fig.update_layout(title="Forecast vs. Confidence Interval", xaxis_title="Date", yaxis_title="Predicted Demand")
    st.plotly_chart(fig, use_container_width=True)
    residuals = fc_df.forecast - fc_df.actual.fillna(fc_df.forecast)
    st.line_chart(residuals, use_container_width=True)
    st.caption("Residuals (Forecast – Actual)")

# ── Tab 6: XAI ──────────────────────────────────────────────────────────────────
with tabs[5]:
    st.header("🔍 Explainable AI (XAI)")
    X = df_filt.select_dtypes(np.number)
    explainer = init_shap(model)
    shap_vals = explainer.shap_values(X)

    c1, c2 = st.columns(2)
    with c1:
        # summary plot
        shap.summary_plot(shap_vals, X, show=False)
        st.pyplot(plt.gcf())

    with c2:
        # partial dependence for top feature
        top_feat = X.columns[np.argmax(np.abs(shap_vals).mean(0))]
        fig_pd, ax = plt.subplots()
        from sklearn.inspection import PartialDependenceDisplay
        PartialDependenceDisplay.from_estimator(
            model, X, [top_feat], ax=ax
        )
        ax.set_title(f"Partial Dependence: {top_feat}")
        st.pyplot(fig_pd)

# ── Tab 7: Geospatial ───────────────────────────────────────────────────────────
with tabs[6]:
    st.header("🌐 Geospatial Insights")
    hex_layer = pdk.Layer(
        "HexagonLayer", data=df_filt, get_position="[longitude, latitude]",
        radius=500, elevation_scale=50, pickable=True
    )
    scatter = pdk.Layer(
        "ScatterplotLayer", data=df_filt,
        get_position="[longitude, latitude]", get_radius=200, pickable=True
    )
    view = pdk.ViewState(latitude=53.5461, longitude=-113.4938, zoom=10, pitch=45)
    st.pydeck_chart(pdk.Deck(
        layers=[hex_layer, scatter], initial_view_state=view,
        tooltip={"text":"Region: {region}\nDemand: {quantity}"}
    ))

# ── Tab 8: Optimization Playground ───────────────────────────────────────────────
with tabs[7]:
    st.header("⚙️ Optimization Playground")
    n_vehicles = st.number_input("Number of Vehicles", 1, 10, 3)
    capacity = st.number_input("Vehicle Capacity", 1, 100, 50)
    if st.button("Solve Routes", key="solve_routes"):
        with st.spinner("Solving vehicle routes…"):
            # build demand list & locations
            regions_agg = df_filt.groupby("region").agg({
                "quantity":"sum","latitude":"mean","longitude":"mean"
            }).reset_index()
            coords = [(lat, lon) for lat, lon in zip(regions_agg.latitude, regions_agg.longitude)]
            demands = regions_agg.quantity.astype(int).tolist()
            depot = (53.5461, -113.4938)
            locations = [depot] + coords
            dist_matrix = [[
                int(np.hypot(loc1[0]-loc2[0], loc1[1]-loc2[1])*111000)
                for loc2 in locations] for loc1 in locations]
            manager = pywrapcp.RoutingIndexManager(len(dist_matrix), n_vehicles, 0)
            routing = pywrapcp.RoutingModel(manager)
            def dist_callback(i,j):
                return dist_matrix[manager.IndexToNode(i)][manager.IndexToNode(j)]
            routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(dist_callback))
            routing.AddDimensionWithVehicleCapacity(
                routing.RegisterUnaryTransitCallback(lambda idx: demands[manager.IndexToNode(idx)]),
                0, [capacity]*n_vehicles, True, "Capacity"
            )
            routing.SetGuessPolicy(routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
            sol = routing.Solve()
            if not sol:
                st.error("No feasible routes—try adjusting capacity or vehicles.")
            else:
                layers = []
                for v in range(n_vehicles):
                    idx = routing.Start(v)
                    route = []
                    while not routing.IsEnd(idx):
                        node = manager.IndexToNode(idx)
                        route.append(locations[node])
                        idx = sol.Value(routing.NextVar(idx))
                    route.append(depot)
                    layers.append(pdk.Layer(
                        "LineLayer", data=[{"path":route}],
                        get_path="path", get_width=4
                    ))
                st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view))

# ── Tab 9: Story Mode ────────────────────────────────────────────────────────────
with tabs[8]:
    st.header("📖 Story Mode")
    if "story_step" not in st.session_state:
        st.session_state.story_step = 0
    prev, next_ = st.columns([1,1])
    with prev:
        if st.button("◀ Previous", key="story_prev") and st.session_state.story_step>0:
            st.session_state.story_step -= 1
    with next_:
        if st.button("Next ▶", key="story_next") and st.session_state.story_step<2:
            st.session_state.story_step += 1

    step = st.session_state.story_step
    if step == 0:
        st.subheader("1. Historical Demand Trends")
        fig = px.histogram(df_filt, x="pickup_date", nbins=bins)
        st.plotly_chart(fig, use_container_width=True)
    elif step == 1:
        st.subheader("2. Key Drivers of Demand")
        imp_df = feat_imp.head(5).reset_index()
        imp_df.columns = ["Feature","Importance"]
        st.bar_chart(imp_df.set_index("Feature"), use_container_width=True)
    else:
        st.subheader("3. Recommendations")
        st.markdown("""
        - Deploy **2 mobile distribution points** in the highest‑demand regions.  
        - Prioritize outreach to families with **3+ dependents**.  
        - Allocate extra pickups on **weekends** for returning clients.
        """)

# ── Tab 10: Chatbot ───────────────────────────────────────────────────────────────
with tabs[9]:
    st.header("💬 Ask the Data")
    query = st.text_input("Your question…", key="chat_input")
    if query:
        result = qa_chain.run(query)
        st.markdown(f"**Answer:** {result['result']}" if isinstance(result, dict) else f"**Answer:** {result}")
        if isinstance(result, dict) and "source_documents" in result:
            for doc in result["source_documents"]:
                with st.expander(f"Source: {doc.metadata.get('source','')}"):
                    st.write(doc.page_content)
