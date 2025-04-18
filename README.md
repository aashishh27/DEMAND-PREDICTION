<p align="center" draggable="false">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR8HNB-ex4xb4H3-PXRcywP5zKC_3U8VzQTPA&usqp=CAU" width="200px" height="auto" />
</p>

# Food Hamper Demand Prediction  
**Team Members:** Aashish Arora, Uchenna Mgbaja

---

### PROJECT TITLE: Food Hamper Demand Prediction

Welcome to the repository for our Capstone project at Norquest College. This project aims to forecast and analyze food hamper demand across Edmonton to support Islamic Family’s outreach and optimize mobile distribution planning.

### Problem Statement

Identify geographic areas in Edmonton with higher or lower food hamper demand to support Islamic Family’s outreach and mobile distribution planning.

### Solution

We implement and compare multiple forecasting approaches—Random Forest, XGBoost, SARIMAX, and Prophet—using historical pickup data combined with engineered time-series and behavioral features (lagged pickups, rolling windows, client demographics). Results are showcased in an interactive Streamlit dashboard featuring:

- Custom date-range and multi-region demand metrics
- Demand momentum gauge comparing current vs. prior period
- Exploratory Data Analysis (histograms, heatmaps)
- Model comparison with RMSE metrics
- Forecast vs. actual trends and confidence intervals
- Explainable AI insights via SHAP
- Geospatial demand mapping
- RAG-powered chatbot for ad-hoc notebook queries

### Repository Structure

```
DEMAND-PREDICTION/
├── data/                        # CSV datasets (raw & processed)
│   └── forecast_2025_rf.csv     # 2025 forecast data
├── notebooks/                   # Jupyter notebooks
│   └── RAG_Implementation.ipynb  # Retrieval-augmented notebook
├── app.py                       # Streamlit dashboard application
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview and instructions
```

### Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/aashishh27/DEMAND-PREDICTION.git
   cd DEMAND-PREDICTION
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

### Link to Application

[Open the Streamlit App](https://your-streamlit-app-url.streamlit.app)

### Team Members

- [Aashish Arora](https://github.com/aashishh27)  
- [Uchenna Mgbaja](https://www.linkedin.com/in/marianmgbaja/)

---

© 2025 Norquest College Capstone


