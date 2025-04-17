import os
import glob
import pickle

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px

from sklearn.inspection import PartialDependenceDisplay
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# LangChain FAISS imports
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
    raw = pickle.load(open(path,"rb"))
    # if it is directly a scikit‑learn model
    if hasattr(raw, "predict"):
        return raw
    # if you pickled a tuple/list that contains the model
    if isinstance(raw, (list,tuple)):
        for cand in raw:
            if hasattr(cand, "predict"):
                return cand
    st.error(f"⚠️ Loaded object from {path!r} has no .predict() method. "
             "Please repickle your sklearn estimator, not an array.")
    st.stop()

@st.cache_data
def prepare_daily_hist(df):
    return (
        df.groupby(["region","pickup_date"])["daily_pickups"]
          .sum().reset_index()
          .sort_values(["region","pickup_date"])
    )

def make_features(daily, df_full, windows=(7,14,30)):
    out = daily.copy()
    out["dow"]   = out.pickup_date.dt.weekday
    out["month"] = out.pickup_date.dt.month
    med_dep = df_full.groupby("region")["dependents_qty"].median().rename("med_dep")
    out = out.merge(med_dep, on="region", how="left")
    out["lag_1"] = out.groupby("region")["daily_pickups"].shift(1).fillna(0)
    for w in windows:
        out[f"roll_{w}"] = (
            out.groupby("region")["daily_pickups"]
               .transform(lambda x: x.shift(1).rolling(w,1).mean())
        )
    return out

def generate_2025(df, model):
    hist = prepare_daily_hist(df)
    last = hist.pickup_date.max()
    future = pd.MultiIndex.from_product(
        [hist.region.unique(),
         pd.date_range(last+pd.Timedelta(days=1),"2025-12-31",freq="D")],
        names=["region","pickup_date"]
    ).to_frame(index=False)
    future["daily_pickups"] = np.nan
    all_days = pd.concat([hist, future], ignore_index=True)
    all_days = make_features(all_days, df)

    # build Xf for prediction
    mask_future = all_days.pickup_date > last
    Xf = all_days.loc[mask_future].drop("daily_pickups", axis=1)

    # only reindex if feature_names_in_ exists
    if hasattr(model, "feature_names_in_"):
        Xf = Xf[model.feature_names_in_]

    all_days.loc[mask_future, "predicted_daily"] = model.predict(Xf)
    return all_days

# ─── RAG / Chatbot via FAISS ─────────────────────────────────────────────────────
@st.cache_data
def load_knowledge_base(path_pattern="knowledge/*.txt"):
    docs = []
    for fname in glob.glob(path_pattern):
        with open(fname, "r", encoding="utf-8") as f:
            docs.append(Document(page_content=f.read(), metadata={"source": fname}))
    return docs

@st.cache_resource
def init_rag_faiss(index_path="faiss_index.pkl"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
       st.warning("⚠️ Chatbot disabled: missing OPENAI_API_KEY")
       return None
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    docs       = load_knowledge_base()
    if os.path.exists(index_path):
        vs = FAISS.load(index_path, embeddings)
    else:
        vs = FAISS.from_documents(docs, embeddings)
        vs.save(index_path)
    llm = ChatOpenAI(temperature=0)
    return RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vs.as_retriever())

# ─── Load everything ─────────────────────────────────────────────────────────────
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
sel_codes = [k for k,v in house_map.items() if v in sel_hh]

mask = (
    df.pickup_date.between(start_dt, end_dt) &
    df.region.isin(sel_regs) &
    df.dependents_qty.between(*sel_dep) &
    df.household.isin(sel_codes)
)
hist_filt = df[mask]

# ─── Build Tabs ―────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "2025 Forecast & Map",
    "Overview & KPIs","Features","Model Comparison",
    "XAI","Geospatial (History)","Optimization","Chatbot"
])

# ── Tab 0: 2025 Forecast & Map ───────────────────────────────────────────────────
with tabs[0]:
    st.header("📅 2025 Demand Predictions by Region")
    all_days = generate_2025(df, model)
    preds    = all_days.query("pickup_date.dt.year==2025 & region in @sel_regs")

    # Time series chart
    fig_ts = px.line(
        preds, x="pickup_date", y="predicted_daily", color="region",
        title="Daily Predicted Quantity (2025)"
    )
    fig_ts.update_layout(xaxis_title="Date", yaxis_title="Predicted Demand")
    st.plotly_chart(fig_ts, use_container_width=True)

    # Geospatial clusters
    agg = (
        preds.groupby("region")
             .agg({"predicted_daily":"sum"})
             .reset_index()
             .merge(
                 df.groupby("region")[["latitude","longitude"]].mean().reset_index(),
                 on="region"
              )
    )
    st.subheader("🔍 Aggregate 2025 Demand Clusters")
    view = pdk.ViewState(latitude=53.5461, longitude=-113.4938, zoom=10, pitch=45)
    hex_layer = pdk.Layer(
        "HexagonLayer", data=agg,
        get_position="[longitude, latitude]",
        get_elevation="predicted_daily", elevation_scale=0.01,
        radius=1000, pickable=True
    )
    scatter_2025 = pdk.Layer(
        "ScatterplotLayer", data=agg,
        get_position="[longitude, latitude]",
        get_radius="predicted_daily * 0.005",
        get_fill_color="[255, 0, 0, 180]", pickable=True
    )
    st.pydeck_chart(pdk.Deck(
        layers=[hex_layer, scatter_2025],
        initial_view_state=view,
        tooltip={"text":"Region: {region}\nTotal 2025 Demand: {predicted_daily:.0f}"}
    ))

# ── Tab 1: Overview & KPIs ───────────────────────────────────────────────────────
with tabs[1]:
    st.header("📦 Historical Overview & KPIs")
    total   = int(hist_filt.daily_pickups.sum())
    days    = (end_dt - start_dt).days + 1
    prev    = hist_filt.assign(prev=hist_filt.daily_pickups).loc[
                 hist_filt.pickup_date.between(
                     start_dt-pd.Timedelta(days=days),
                     start_dt-pd.Timedelta(days=1)
                 )
             ]
    prev_tot = int(prev.prev.sum())
    col1, col2 = st.columns(2)
    col1.metric(f"{start_dt.date()}→{end_dt.date()}", f"{total:,}")
    delta = (total - prev_tot) / prev_tot if prev_tot else None
    col2.metric("Change vs Prev Period", f"{total-prev_tot:,}",
                f"{delta:.1%}" if delta else "N/A")

# ── Tab 2: Features ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("🛠️ Feature Importance (History)")
    imp = pd.Series(model.feature_importances_,
                    index=hist_filt.select_dtypes(include=np.number).columns)
    imp = imp.sort_values(ascending=False).head(10).reset_index()
    imp.columns = ["Feature","Importance"]
    st.bar_chart(imp.set_index("Feature"))

# ── Tab 3: Model Comparison ─────────────────────────────────────────────────────
with tabs[3]:
    st.header("🤖 Model Comparison")
    comp = pd.DataFrame({
        "RandomForest":[1.78,2.30],
        "XGBoost":[2.08,2.61]
    }, index=["CV RMSE","Test RMSE"])

    st.subheader("📊 RMSE Comparison")
    fig_bar = px.bar(comp, barmode="group", title="Cross‑Validation vs Test RMSE")
    fig_bar.update_layout(xaxis_title="Metric", yaxis_title="RMSE")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("📈 RMSE Heatmap")
    fig_heat = px.imshow(comp, text_auto=True, aspect="auto", title="RMSE Heatmap")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("📋 RMSE Table")
    st.dataframe(comp)

# ── Tab 4: XAI ────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("🔍 Explainable AI (SHAP)")
    img_paths   = sorted(glob.glob(os.path.join("shap_images", "*.png")))
    caption_map = {
        "shap7.png": "Residuals vs. Fitted Values",
        "shap6.png": "Histogram of Residuals",
        "shap5.png": "Random Forest Feature Importances",
        "shap4.png": "SHAP Summary Plot (Impact vs. Feature Value)",
        "shap3.png": "Partial Dependence Plots",
        "shap2.png": "Time Series Decomposition",
        "shap1.png": "RF Prediction Intervals"
    }
    if img_paths:
        for img in img_paths:
            fname = os.path.basename(img)
            title = caption_map.get(
                fname,
                fname.replace(".png","").replace("_"," ").title()
            )
            st.subheader(title)
            st.image(img, use_column_width=True)
    else:
        st.warning("No SHAP images found in `shap_images/`. Add shap1.png…shap7.png.")

# ── Tab 5: Geospatial (History) ─────────────────────────────────────────────────
with tabs[5]:
    st.header("🌐 Geospatial Insights (Historical)")
    view = pdk.ViewState(latitude=53.5461, longitude=-113.4938, zoom=10, pitch=45)
    hex_h = pdk.Layer(
        "HexagonLayer", data=hist_filt,
        get_position="[longitude, latitude]",
        radius=500, elevation_scale=50, pickable=True
    )
    st.pydeck_chart(pdk.Deck(layers=[hex_h], initial_view_state=view))

# ── Tab 6: Optimization ─────────────────────────────────────────────────────────
with tabs[6]:
    st.header("⚙️ Route Optimization")
    nveh = st.number_input("Vehicles", 1, 10, 3)
    cap  = st.number_input("Capacity", 1, 200, 50)
    if st.button("Solve Routes"):
        regs    = hist_filt.groupby("region").agg(
                    pickups="daily_pickups", latitude="mean", longitude="mean"
                 ).reset_index()
        coords  = list(zip(regs.latitude, regs.longitude))
        demands = regs.pickups.astype(int).tolist()
        depot   = (53.5461, -113.4938)
        locs    = [depot] + coords
        dist    = [
            [int(np.hypot(a[0]-b[0], a[1]-b[1])*111000) for b in locs]
            for a in locs
        ]
        mgr = pywrapcp.RoutingIndexManager(len(dist), nveh, 0)
        rt  = pywrapcp.RoutingModel(mgr)
        cb  = rt.RegisterTransitCallback(
                 lambda i,j: dist[mgr.IndexToNode(i)][mgr.IndexToNode(j)]
              )
        rt.SetArcCostEvaluatorOfAllVehicles(cb)
        rt.AddDimensionWithVehicleCapacity(
            rt.RegisterUnaryTransitCallback(
                lambda idx: demands[mgr.IndexToNode(idx)]
            ), 0, [cap]*nveh, True, "Capacity"
        )
        rt.SetGuessPolicy(routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        sol = rt.Solve()
        if not sol:
            st.error("No feasible routes—try adjusting capacity or vehicles.")
        else:
            layers = []
            for v in range(nveh):
                idx, route = rt.Start(v), []
                while not rt.IsEnd(idx):
                    route.append(locs[mgr.IndexToNode(idx)])
                    idx = sol.Value(rt.NextVar(idx))
                route.append(depot)
                layers.append(pdk.Layer(
                    "LineLayer", data=[{"path":route}],
                    get_path="path", get_width=4
                ))
            st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view))

# ── Tab 7: Chatbot ──────────────────────────────────────────────────────────────
with tabs[7]:
    st.header("💬 Ask the Data")
    q = st.text_input("Enter a question…")
    if q and qa_chain:
        res = qa_chain.run(q)
        if isinstance(res, dict):
            st.markdown(f"**Answer:** {res['result']}")
            for d in res.get("source_documents", []):
                with st.expander(f"Source: {d.metadata.get('source','')}"):
                    st.write(d.page_content)
        else:
            st.markdown(f"**Answer:** {res}")
    elif q:
        st.info("Chatbot is disabled (missing API key).")


