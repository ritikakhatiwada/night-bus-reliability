"""
06_dashboard.py
───────────────
Streamlit dashboard for Night Bus Service Reliability Prediction.

Run with:  streamlit run src/06_dashboard.py

Features:
  • Live delay prediction using best saved model
  • Operator reliability comparison
  • Route filter & delay map
  • Disruption breakdown

ST5011CEM | Night Bus Service Reliability Prediction
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sqlalchemy import create_engine, text
from config.config import DATABASE_URL

st.set_page_config(
    page_title="Night Bus Reliability Platform",
    page_icon="🚌",
    layout="wide",
)

# ── Load data & model ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("SELECT * FROM timetables_clean", engine)
    df_dis = pd.read_sql("SELECT * FROM disruptions_clean", engine)
    return df, df_dis

@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "models", "Random_Forest.pkl")
    return joblib.load(path) if os.path.exists(path) else None

df, df_dis = load_data()
model = load_model()

# ── Header ───────────────────────────────────────────────────────────────
st.title("🚌 Night Bus Service Reliability Prediction Platform")
st.caption("ST5011CEM | Big Data Programming Project | Softwarica College")

# ── KPI row ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Trips",    f"{len(df):,}")
k2.metric("% On-Time",      f"{(1-df['is_delayed'].mean())*100:.1f}%",
          delta=f"{(1-df['is_delayed'].mean())*100-85:.1f}% vs 85% target")
k3.metric("Avg Delay",      f"{df['delay_minutes'].mean():.2f} min")
k4.metric("Disruptions",    f"{len(df_dis):,}")

st.divider()

# ── Sidebar filters ──────────────────────────────────────────────────────
st.sidebar.header("Filters")
operators = st.sidebar.multiselect(
    "Operator", df["operator"].unique(), default=list(df["operator"].unique()))
routes = st.sidebar.multiselect(
    "Route", sorted(df["service_code"].unique()),
    default=sorted(df["service_code"].unique())[:5])
hours = st.sidebar.slider("Hour of Night", 0, 23, (22, 5))

mask = (df["operator"].isin(operators)) & (df["service_code"].isin(routes))
df_f = df[mask]

# ── Tab layout ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🗺️ Route Analysis", "⚡ Predict Delay", "🚧 Disruptions"])

# ──────────────────────────────────────
with tab1:
    st.subheader("Operator Reliability vs 85% Threshold")
    op = (df_f.groupby("operator")
             .agg(pct_on_time=("is_delayed", lambda x: (1-x.mean())*100),
                  trips=("trip_id","count"))
             .reset_index()
             .sort_values("pct_on_time"))
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#C44E52" if v < 85 else "#55A868" for v in op["pct_on_time"]]
    ax.barh(op["operator"], op["pct_on_time"], color=colors)
    ax.axvline(85, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("% On-Time")
    st.pyplot(fig)
    plt.close()

    st.subheader("Delay Distribution")
    fig2, ax2 = plt.subplots(figsize=(9, 3))
    ax2.hist(df_f["delay_minutes"].clip(-15, 30), bins=50,
             color="#4C72B0", edgecolor="white")
    ax2.axvline(0, color="red", linestyle="--")
    ax2.axvline(2, color="orange", linestyle="--")
    ax2.set_xlabel("Delay (min)")
    st.pyplot(fig2)
    plt.close()

# ──────────────────────────────────────
with tab2:
    st.subheader("Route Reliability Heatmap")
    route_hour = (df_f.groupby(["service_code","hour_of_night"])
                      .agg(pct_delayed=("is_delayed","mean"))
                      .reset_index()
                      .pivot(index="service_code",
                             columns="hour_of_night",
                             values="pct_delayed"))
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.heatmap(route_hour, cmap="RdYlGn_r", center=0.15,
                linewidths=0.3, ax=ax3, fmt=".0%", annot=False)
    ax3.set_title("% Delayed by Route × Hour")
    st.pyplot(fig3)
    plt.close()

# ──────────────────────────────────────
with tab3:
    st.subheader("Predict Trip Delay (Random Forest)")
    if model is None:
        st.warning("Model not found. Run 04_ml_pipeline.py first.")
    else:
        c1, c2, c3 = st.columns(3)
        weather_code      = c1.selectbox("Weather", [0,1,2,3,4],
                                          format_func=lambda x: ["Clear","Rain","Fog","Snow","Windy"][x])
        hour_of_night     = c2.slider("Hour", 0, 23, 23)
        load_factor       = c3.slider("Load Factor", 0.0, 1.0, 0.5)
        vehicle_age       = c1.slider("Vehicle Age (yrs)", 1, 15, 5)
        driver_exp        = c2.slider("Driver Experience (yrs)", 1, 30, 10)
        sched_dur         = c3.slider("Scheduled Duration (min)", 20, 90, 45)
        dis_count         = c1.slider("Disruption Count", 0, 20, 0)
        hi_sev            = c2.slider("High Severity Disruptions", 0, 5, 0)
        cv_tt             = c3.slider("Travel Time CV", 0.0, 0.5, 0.12)
        pct_ot            = c1.slider("Route % On-Time", 0.0, 1.0, 0.85)
        pax               = c2.slider("Passenger Count", 0, 75, 30)

        if st.button("Predict"):
            X_input = np.array([[weather_code, hour_of_night, load_factor,
                                  vehicle_age, driver_exp, sched_dur,
                                  dis_count, hi_sev, cv_tt, pct_ot, pax]])
            pred  = model.predict(X_input)[0]
            prob  = model.predict_proba(X_input)[0][1]
            label = "🔴 DELAYED" if pred == 1 else "🟢 ON-TIME"
            st.metric("Prediction", label, delta=f"Delay probability: {prob:.1%}")

# ──────────────────────────────────────
with tab4:
    st.subheader("Disruption Analysis")
    col1, col2 = st.columns(2)
    dis_type = df_dis["disruption_type"].value_counts()
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    dis_type.plot(kind="barh", ax=ax4, color="#4C72B0")
    ax4.set_title("Disruptions by Type")
    col1.pyplot(fig4)
    plt.close()

    sev = df_dis["severity"].value_counts()
    fig5, ax5 = plt.subplots(figsize=(5, 4))
    ax5.pie(sev, labels=sev.index, autopct="%1.1f%%",
            colors=["#55A868","#F0A500","#C44E52"])
    ax5.set_title("Disruption Severity")
    col2.pyplot(fig5)
    plt.close()

    st.dataframe(df_dis.head(20), use_container_width=True)
