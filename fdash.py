import os
import glob
import pickle

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

from sklearn.inspection import PartialDependenceDisplay
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# LangChain FAISS imports (no sqlite required)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

st.set_page_config(page_title="🌐 Geospatial Demand Prediction Studio", layout="wide")

# ─── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data(path="region_client_df (1).csv"):
    return pd.read_csv(path, parse_dates=["pickup_date"])

@st.cache_resource
def load_model(path="optimized_random_forest.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
@st.cache_resource
def get_shap_explainer(model, X_ref):
    """
    Return a unified SHAP Explainer that auto‐selects Tree vs Kernel.
    """
    # shap.Explainer will choose TreeExplainer for tree models,
    # otherwise fall back to the generic kernel method.
    return shap.Explainer(model, X_ref)

@st.cache_data
def prepare_daily_hist(df):
    return (
        df.groupby(["region","pickup_date"])["quantity"]
          .sum().reset_index()
          .sort_values(["region","pickup_date"])
    )

def make_features(daily, df_full, windows=(7,14,30)):
    out = daily.copy()
    out["dow"]   = out.pickup_date.dt.weekday
    out["month"] = out.pickup_date.dt.month
    med_dep = df_full.groupby("region")["dependents_qty"].median().rename("med_dep")
    out = out.merge(med_dep, on="region", how="left")
    out["lag_1"] = out.groupby("region")["quantity"].shift(1).fillna(0)
    for w in windows:
        out[f"roll_{w}"] = (
            out.groupby("region")["quantity"]
               .transform(lambda x: x.shift(1).rolling(w,1).mean())
        )
    return out

def generate_2025(df, model):
    hist = prepare_daily_hist(df)
    last = hist.pickup_date.max()
    future = pd.MultiIndex.from_product(
        [hist.region.unique(),
         pd.date_range(last+pd.Timedelta(days=1), "2025-12-31", freq="D")],
        names=["region","pickup_date"]
    ).to_frame(index=False)
    future["quantity"] = np.nan
    all_days = pd.concat([hist,future], ignore_index=True)
    all_days = make_features(all_days, df)
    Xf = all_days.loc[all_days.pickup_date > last].drop("quantity", axis=1)
    Xf = Xf[model.feature_names_in_]
    all_days.loc[all_days.pickup_date > last, "predicted_qty"] = model.predict(Xf)
    return all_days

# ─── RAG / Chatbot via FAISS ─────────────────────────────────────────────────────
@st.cache_data
def load_knowledge_base(path_pattern="knowledge/*.txt"):
    docs = []
    for fname in glob.glob(path_pattern):
        with open(fname, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append(Document(page_content=text, metadata={"source": fname}))
    return docs

@st.cache_resource
def init_rag_faiss(index_path="faiss_index.pkl"):
   # supply your OpenAI key to the embeddings constructor
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
       st.warning("⚠️ Chatbot disabled: missing OPENAI_API_KEY")
       return None
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    docs = load_knowledge_base()
    if os.path.exists(index_path):
        vs = FAISS.load(index_path, embeddings)
    else:
        vs = FAISS.from_documents(docs, embeddings)
        vs.save(index_path)
    llm = ChatOpenAI(temperature=0)
    return RetrievalQA.from_chain_type(
        llm, chain_type="stuff", retriever=vs.as_retriever()
    )

# ─── Load data, model, RAG ―────────────────────────────────────────────────────

df       = load_data()
model    = load_model()
qa_chain = init_rag_faiss()

# ─── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
min_dt, max_dt = df.pickup_date.min(), df.pickup_date.max()
start_dt, end_dt = st.sidebar.date_input(
    "History Range", [min_dt, max_dt],
    min_value=min_dt, max_value=max_dt
)
start_dt, end_dt = pd.to_datetime(start_dt), pd.to_datetime(end_dt)

regions = sorted(df.region.unique())
sel_regs = st.sidebar.multiselect("Regions", regions, default=regions)

dep_min, dep_max = int(df.dependents_qty.min()), int(df.dependents_qty.max())
sel_dep = st.sidebar.slider("Dependents", dep_min, dep_max, (dep_min, dep_max))

house_map = {0.0: "Single", 1.0: "Multi"}
sel_hh = st.sidebar.multiselect(
    "Household", [house_map[x] for x in sorted(df.household.unique())],
    default=[house_map[x] for x in sorted(df.household.unique())]
)
sel_codes = [k for k, v in house_map.items() if v in sel_hh]

mask = (
    df.pickup_date.between(start_dt, end_dt) &
    df.region.isin(sel_regs) &
    df.dependents_qty.between(*sel_dep) &
    df.household.isin(sel_codes)
)
hist_filt = df[mask]

# ─── Compute SHAP ───────────────────────────────────────────────────────────────
X_num      = hist_filt.select_dtypes(include=[np.number])
explainer  = get_shap_explainer(model, X_num)
# run the explainer to get an Explanation object
shap_exp   = explainer(X_num)

# ─── Build Tabs ―────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "2025 Forecast & Map",
    "Overview & KPIs","Features","Model Comparison",
    "XAI","Geospatial (History)","Optimization","Chatbot"
])

# ── Tab 1: 2025 Forecast & Map ───────────────────────────────────────────────────
with tabs[0]:
    st.header("📅 2025 Demand Predictions by Region")
    all_days = generate_2025(df, model)
    preds = all_days.query("pickup_date.dt.year==2025 & region in @sel_regs")
    fig_ts = px.line(
        preds, x="pickup_date", y="predicted_qty", color="region",
        title="Daily Predicted Quantity (2025)"
    )
    fig_ts.update_layout(xaxis_title="Date", yaxis_title="Predicted Demand")
    st.plotly_chart(fig_ts, use_container_width=True)

    agg = (
        preds.groupby("region")
             .agg({"predicted_qty":"sum","latitude":"mean","longitude":"mean"})
             .reset_index()
    )
    st.subheader("🔍 Aggregate 2025 Demand Clusters")
    view = pdk.ViewState(latitude=53.5461, longitude=-113.4938, zoom=10, pitch=45)
    hex_layer = pdk.Layer(
        "HexagonLayer", data=agg,
        get_position="[longitude, latitude]",
        get_elevation="predicted_qty",
        elevation_scale=0.01,
        radius=1000,
        pickable=True
    )
    st.pydeck_chart(pdk.Deck(layers=[hex_layer], initial_view_state=view))

# ── Tab 2: Overview & KPIs ───────────────────────────────────────────────────────
with tabs[1]:
    st.header("📦 Historical Overview & KPIs")
    total = int(hist_filt.quantity.sum())
    days = (end_dt-start_dt).days+1
    prev = hist_filt.assign(prev=hist_filt.quantity).loc[
        hist_filt.pickup_date.between(start_dt-pd.Timedelta(days=days),start_dt-pd.Timedelta(days=1))
    ]
    prev_tot = int(prev.prev.sum())
    col1, col2 = st.columns(2)
    col1.metric(f"{start_dt.date()}→{end_dt.date()}", f"{total:,}")
    delta = (total-prev_tot)/prev_tot if prev_tot else None
    col2.metric(f"Change vs Prev Period", f"{total-prev_tot:,}", f"{delta:.1%}" if delta else "N/A")

# ── Tab 3: Features ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("🛠️ Feature Importance (History)")
    imp = pd.Series(model.feature_importances_, index=X_num.columns)
    imp = imp.sort_values(ascending=False).head(10).reset_index()
    imp.columns = ["Feature","Importance"]
    st.bar_chart(imp.set_index("Feature"))

# ── Tab 4: Model Comparison ─────────────────────────────────────────────────────
with tabs[3]:
    st.header("🤖 Model Comparison")
    st.write("CV vs Test RMSE (RF vs XGBoost)")
    comp = pd.DataFrame({
        "RandomForest":[1.78,2.30],
        "XGBoost":[2.08,2.61]
    },index=["CV RMSE","Test RMSE"])
    st.bar_chart(comp)

# ── Tab 5: XAI ──────────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("🔍 Explainable AI (SHAP)")
    c1,c2 = st.columns(2)
    with c1:
        shap.summary_plot(shap_exp.values, X_num, show=False)
        st.pyplot(plt.gcf())
    with c2:
        top = X_num.columns[np.argmax(np.abs(shap_exp.values).mean(0))]
        fig_pd, ax = plt.subplots()
        PartialDependenceDisplay.from_estimator(model, X_num, [top], ax=ax)
        ax.set_title(f"Partial Dependence: {top}")
        st.pyplot(fig_pd)

# ── Tab 6: Geospatial (History) ─────────────────────────────────────────────────
with tabs[5]:
    st.header("🌐 Geospatial Insights (Historical)")
    view = pdk.ViewState(latitude=53.5461, longitude=-113.4938, zoom=10, pitch=45)
    hex_h = pdk.Layer(
        "HexagonLayer", data=hist_filt,
        get_position="[longitude, latitude]",
        radius=500, elevation_scale=50, pickable=True
    )
    st.pydeck_chart(pdk.Deck(layers=[hex_h], initial_view_state=view))

# ── Tab 7: Optimization ─────────────────────────────────────────────────────────
with tabs[6]:
    st.header("⚙️ Route Optimization")
    nveh = st.number_input("Vehicles",1,10,3)
    cap  = st.number_input("Capacity",1,200,50)
    if st.button("Solve"):
        regs = hist_filt.groupby("region").agg({"quantity":"sum","latitude":"mean","longitude":"mean"}).reset_index()
        coords = list(zip(regs.latitude,regs.longitude))
        demands= regs.quantity.astype(int).tolist()
        depot=(53.5461,-113.4938)
        locs=[depot]+coords
        dist=[[int(np.hypot(a[0]-b[0],a[1]-b[1])*111000)
               for b in locs] for a in locs]
        mgr = pywrapcp.RoutingIndexManager(len(dist),nveh,0)
        rt  = pywrapcp.RoutingModel(mgr)
        cb  = rt.RegisterTransitCallback(lambda i,j:dist[mgr.IndexToNode(i)][mgr.IndexToNode(j)])
        rt.SetArcCostEvaluatorOfAllVehicles(cb)
        rt.AddDimensionWithVehicleCapacity(
            rt.RegisterUnaryTransitCallback(lambda idx:demands[mgr.IndexToNode(idx)]),
            0,[cap]*nveh,True,"Capacity"
        )
        rt.SetGuessPolicy(routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        sol=rt.Solve()
        if not sol: st.error("No solution")
        else:
            layers=[]
            for v in range(nveh):
                idx=rt.Start(v);route=[]
                while not rt.IsEnd(idx):
                    route.append(locs[mgr.IndexToNode(idx)])
                    idx=sol.Value(rt.NextVar(idx))
                route.append(depot)
                layers.append(pdk.Layer("LineLayer",data=[{"path":route}],get_path="path",get_width=4))
            st.pydeck_chart(pdk.Deck(layers=layers,initial_view_state=view))

# ── Tab 8: Chatbot ──────────────────────────────────────────────────────────────
with tabs[7]:
    st.header("💬 Ask the Data")
    query = st.text_input("Enter a question…")
    if query:
        result = qa_chain.run(query)
        if isinstance(result, dict):
            st.markdown(f"**Answer:** {result['result']}")
            for doc in result.get("source_documents", []):
                with st.expander(f"Source: {doc.metadata.get('source','')}"):
                    st.write(doc.page_content)
        else:
            st.markdown(f"**Answer:** {result}")
