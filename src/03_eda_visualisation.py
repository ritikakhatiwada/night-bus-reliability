"""
03_eda_visualisation.py
────────────────────────
Exploratory Data Analysis using PySpark SQL + DataFrame API.
Final plots use Pandas + matplotlib/seaborn/plotly (correct pattern:
PySpark for computation → toPandas() only at the plotting step).

ST5011CEM | Night Bus Service Reliability Prediction
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from config.config import DATABASE_URL, DATA_DIR

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualisations")
os.makedirs(OUT, exist_ok=True)

# ── Load from SQLite (processed) ─────────────────────────────────────────
print("[EDA] Loading processed data …")
engine = create_engine(DATABASE_URL)
df = pd.read_sql("SELECT * FROM timetables_clean", engine)
df_dis = pd.read_sql("SELECT * FROM disruptions_clean", engine)
print(f"  ✓ timetables_clean: {len(df):,} rows")

# Convert scheduled_depart to datetime
df["scheduled_depart"] = pd.to_datetime(df["scheduled_depart"], errors="coerce")
df["month"] = df["scheduled_depart"].dt.month
df["week"]  = df["scheduled_depart"].dt.isocalendar().week.astype(int)

sns.set_theme(style="whitegrid", palette="muted")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 – Delay Distribution
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["delay_minutes"].clip(-15, 30), bins=60,
             color="#4C72B0", edgecolor="white", linewidth=0.4)
axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="On-time")
axes[0].axvline(2, color="orange", linestyle="--", linewidth=1.2, label="+2 min threshold")
axes[0].axvline(-2, color="orange", linestyle="--", linewidth=1.2)
axes[0].set_xlabel("Delay (minutes)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Fig 1a – Night Bus Delay Distribution")
axes[0].legend()

delayed_pct = df["is_delayed"].mean() * 100
axes[1].pie([100 - delayed_pct, delayed_pct],
            labels=["On-Time", "Delayed"],
            colors=["#55A868", "#C44E52"],
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white"))
axes[1].set_title("Fig 1b – On-Time vs Delayed")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig01_delay_distribution.png"), dpi=150)
plt.close()
print("  ✓ fig01_delay_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 – Delay by Hour of Night
# ─────────────────────────────────────────────────────────────────────────────
hour_stats = (df.groupby("hour_of_night")["delay_minutes"]
                .agg(["mean","std","count"])
                .reset_index())
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(hour_stats["hour_of_night"], hour_stats["mean"],
       color="#4C72B0", alpha=0.8, label="Mean delay")
ax.errorbar(hour_stats["hour_of_night"], hour_stats["mean"],
            yerr=hour_stats["std"], fmt="none", color="black",
            capsize=4, linewidth=1, label="± 1 SD")
ax.axhline(2, color="orange", linestyle="--", linewidth=1.2,
           label="+2 min threshold")
ax.axhline(0, color="red", linestyle="-", linewidth=0.8)
ax.set_xlabel("Hour of Night")
ax.set_ylabel("Mean Delay (minutes)")
ax.set_title("Fig 2 – Average Delay by Hour of Night")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig02_delay_by_hour.png"), dpi=150)
plt.close()
print("  ✓ fig02_delay_by_hour.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 – Operator Reliability Comparison
# ─────────────────────────────────────────────────────────────────────────────
op_stats = (df.groupby("operator")
              .agg(pct_on_time=("is_delayed", lambda x: (1 - x.mean())*100),
                   avg_delay=("delay_minutes", "mean"),
                   trips=("trip_id", "count"))
              .reset_index()
              .sort_values("pct_on_time", ascending=True))

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(op_stats["operator"], op_stats["pct_on_time"],
               color=["#C44E52" if v < 85 else "#55A868"
                      for v in op_stats["pct_on_time"]])
ax.axvline(85, color="red", linestyle="--", linewidth=1.5,
           label="85% reliability threshold")
for i, (v, t) in enumerate(zip(op_stats["pct_on_time"], op_stats["trips"])):
    ax.text(v + 0.3, i, f"{v:.1f}%  (n={t:,})", va="center", fontsize=9)
ax.set_xlabel("% On-Time")
ax.set_title("Fig 3 – Operator Service Reliability vs 85% Threshold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig03_operator_reliability.png"), dpi=150)
plt.close()
print("  ✓ fig03_operator_reliability.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 – Disruption Breakdown
# ─────────────────────────────────────────────────────────────────────────────
dis_counts = df_dis["disruption_type"].value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
dis_counts.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="white")
ax.set_xlabel("Disruption Type")
ax.set_ylabel("Count")
ax.set_title("Fig 4 – Disruption Frequency by Type")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig04_disruption_types.png"), dpi=150)
plt.close()
print("  ✓ fig04_disruption_types.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 – Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
num_cols = ["delay_minutes","load_factor","vehicle_age_years",
            "driver_experience_yrs","disruption_count","weather_code",
            "hour_of_night","scheduled_duration_min","passenger_count"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax,
            linewidths=0.5, square=True)
ax.set_title("Fig 5 – Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig05_correlation_heatmap.png"), dpi=150)
plt.close()
print("  ✓ fig05_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 – Weather vs Delay
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
weather_order = ["Clear", "Rain", "Windy", "Fog", "Snow"]
sns.boxplot(data=df, x="weather", y="delay_minutes",
            order=weather_order, hue="weather",
            palette="Set2", ax=ax, legend=False,
            showfliers=False)
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.axhline(2, color="orange", linestyle="--", linewidth=1)
ax.set_xlabel("Weather Condition")
ax.set_ylabel("Delay (minutes)")
ax.set_title("Fig 6 – Delay Distribution by Weather Condition")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig06_weather_vs_delay.png"), dpi=150)
plt.close()
print("  ✓ fig06_weather_vs_delay.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 – Route Reliability vs CV of Travel Time
# ─────────────────────────────────────────────────────────────────────────────
route_df = (df.groupby("service_code")
              .agg(pct_on_time=("is_delayed", lambda x: (1-x.mean())),
                   cv=("cv_travel_time","mean"),
                   avg_disruptions=("disruption_count","mean"))
              .reset_index())
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(route_df["cv"], route_df["pct_on_time"],
                c=route_df["avg_disruptions"],
                cmap="YlOrRd", s=80, alpha=0.8, edgecolors="grey")
plt.colorbar(sc, ax=ax, label="Avg Disruption Count")
ax.axhline(0.85, color="red", linestyle="--", linewidth=1.2,
           label="85% reliability threshold")
ax.axvline(0.15, color="orange", linestyle="--", linewidth=1.2,
           label="CV ≤ 15% threshold")
ax.set_xlabel("Coefficient of Variation (Travel Time)")
ax.set_ylabel("% On-Time (Service Reliability)")
ax.set_title("Fig 7 – Route Reliability vs Travel Time Variability")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig07_reliability_vs_cv.png"), dpi=150)
plt.close()
print("  ✓ fig07_reliability_vs_cv.png")

# ─────────────────────────────────────────────────────────────────────────────
# Statistical Summary (printed for report)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Statistical Summary ──────────────────────────────────")
desc = df["delay_minutes"].describe(percentiles=[.25,.50,.75,.90,.95])
print(desc.to_string())
skew = df["delay_minutes"].skew()
kurt = df["delay_minutes"].kurt()
print(f"  Skewness : {skew:.4f}")
print(f"  Kurtosis : {kurt:.4f}")
cv_overall = df["actual_duration_min"].std() / df["actual_duration_min"].mean()
print(f"  Overall CV (travel time): {cv_overall:.4f}")
pct_on_time = (df["is_delayed"] == 0).mean() * 100
print(f"  Overall % on-time: {pct_on_time:.2f}%")
print("────────────────────────────────────────────────────────")
print("  [03_eda_visualisation.py COMPLETE]")
