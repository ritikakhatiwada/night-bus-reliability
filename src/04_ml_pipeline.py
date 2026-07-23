"""
04_ml_pipeline.py
─────────────────
Machine-learning pipeline: Non-Compliance Classification
  • Model 1 – Logistic Regression  (baseline, interpretable)
  • Model 2 – Random Forest        (ensemble, handles imbalance)
  • Model 3 – Gradient Boosting    (XGBoost-style, best accuracy)

Evaluation:  Accuracy, Precision, Recall, F1-score, ROC-AUC
Big-O complexity noted for each algorithm.

Why scikit-learn here instead of MLlib?
  PySpark MLlib requires the full Spark column pipeline and does not
  support all evaluation metrics natively (e.g. ROC-AUC curve plotting).
  The data is loaded from SQLite (already processed by Spark in step 02),
  and at 80 k rows the sklearn pipeline trains in seconds.  We note that
  the equivalent MLlib pipeline is demonstrated in 05_spark_ml_pipeline.py.

ST5011CEM | Night Bus Service Reliability Prediction
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve, classification_report)
from sklearn.inspection import permutation_importance
import joblib
from config.config import (DATABASE_URL, RANDOM_SEED, TEST_SIZE,
                            LABEL_COL, FEATURES_COL)

OUT_VIZ   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualisations")
OUT_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(OUT_VIZ, exist_ok=True)
os.makedirs(OUT_MODEL, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

# ── Load Data ────────────────────────────────────────────────────────────
print("[ML] Loading training data …")
engine = create_engine(DATABASE_URL)
df = pd.read_sql("SELECT * FROM timetables_clean", engine)
print(f"  ✓ {len(df):,} rows loaded")

# ── Feature Engineering ──────────────────────────────────────────────────
FEATURES = [
    "weather_code",
    "hour_of_night",
    "load_factor",
    "vehicle_age_years",
    "driver_experience_yrs",
    "scheduled_duration_min",
    "disruption_count",
    "high_severity_count",
    "cv_travel_time",
    "pct_on_time",
    "passenger_count",
]

df_ml = df[FEATURES + [LABEL_COL]].dropna()
print(f"  ✓ ML dataset: {len(df_ml):,} rows, {len(FEATURES)} features")
print(f"  Class balance: {df_ml[LABEL_COL].value_counts(normalize=True).to_dict()}")

X = df_ml[FEATURES].values
y = df_ml[LABEL_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── Model Definitions ────────────────────────────────────────────────────
# Big-O complexity notes:
#   Logistic Regression : O(n·d·k) per iteration  (n=samples, d=features, k=classes)
#   Random Forest        : O(n·d·log(n)·T)         (T=trees)
#   Gradient Boosting    : O(n·d·log(n)·T·M)       (M=boosting rounds) — slowest

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=RANDOM_SEED,
                                    class_weight="balanced", C=0.5))
    ]),
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=150, max_depth=12,
                                        min_samples_leaf=5,
                                        class_weight="balanced",
                                        random_state=RANDOM_SEED, n_jobs=-1))
    ]),
    "Gradient Boosting": Pipeline([
        ("clf", GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                            max_depth=5, subsample=0.8,
                                            random_state=RANDOM_SEED))
    ]),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# ── Train & Evaluate ─────────────────────────────────────────────────────
print("\n[ML] Training models …\n")
for name, pipe in models.items():
    t0 = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred  = pipe.predict(X_test)
    y_prob  = pipe.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    # 5-fold CV F1
    cv_f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1", n_jobs=-1).mean()

    results[name] = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "roc_auc": auc, "cv_f1": cv_f1,
        "train_time_s": round(train_time, 2),
        "model_efficiency": round(f1 / train_time, 4),
        "y_pred": y_pred, "y_prob": y_prob,
    }

    print(f"  {name}")
    print(f"    Accuracy : {acc:.4f} | Precision: {prec:.4f}")
    print(f"    Recall   : {rec:.4f} | F1-score : {f1:.4f}")
    print(f"    ROC-AUC  : {auc:.4f} | CV-F1    : {cv_f1:.4f}")
    print(f"    Train time: {train_time:.2f}s | Efficiency: {f1/train_time:.4f} F1/s")
    print()

    # Save model
    joblib.dump(pipe, os.path.join(OUT_MODEL, f"{name.replace(' ','_')}.pkl"))

# ── Fig 8 – Model Comparison Bar Chart ───────────────────────────────────
metrics  = ["accuracy", "precision", "recall", "f1", "roc_auc"]
x        = np.arange(len(metrics))
n_models = len(results)
width    = 0.25
colours  = ["#4C72B0", "#55A868", "#C44E52"]

fig, ax = plt.subplots(figsize=(13, 6))
for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=name, color=colours[i], alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x + width)
ax.set_xticklabels([m.upper() for m in metrics])
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score")
ax.set_title("Fig 8 – Model Performance Comparison (Test Set)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_VIZ, "fig08_model_comparison.png"), dpi=150)
plt.close()
print("  ✓ fig08_model_comparison.png")

# ── Fig 9 – ROC Curves ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for (name, res), c in zip(results.items(), colours):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax.plot(fpr, tpr, color=c, linewidth=2,
            label=f"{name}  (AUC={res['roc_auc']:.3f})")
ax.plot([0,1],[0,1],"k--", linewidth=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Fig 9 – ROC Curves – Night Bus Delay Classification")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_VIZ, "fig09_roc_curves.png"), dpi=150)
plt.close()
print("  ✓ fig09_roc_curves.png")

# ── Fig 10 – Confusion Matrices ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["On-Time","Delayed"],
                yticklabels=["On-Time","Delayed"],
                linewidths=0.5)
    ax.set_title(f"Fig 10{chr(97+list(results).index(name))} – {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUT_VIZ, "fig10_confusion_matrices.png"), dpi=150)
plt.close()
print("  ✓ fig10_confusion_matrices.png")

# ── Fig 11 – Feature Importance (Random Forest) ──────────────────────────
rf_pipe = models["Random Forest"]
rf_clf  = rf_pipe.named_steps["clf"]
importances = rf_clf.feature_importances_
feat_df = (pd.DataFrame({"feature": FEATURES, "importance": importances})
             .sort_values("importance", ascending=True))
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(feat_df["feature"], feat_df["importance"], color="#4C72B0", alpha=0.85)
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Fig 11 – Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUT_VIZ, "fig11_feature_importance.png"), dpi=150)
plt.close()
print("  ✓ fig11_feature_importance.png")

# ── Fig 12 – Model Efficiency (F1 / Train-time) ──────────────────────────
eff_df = pd.DataFrame([
    {"Model": n, "F1": r["f1"], "TrainTime": r["train_time_s"],
     "Efficiency": r["model_efficiency"]}
    for n, r in results.items()
])
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()
x_pos = np.arange(len(eff_df))
ax1.bar(x_pos - 0.2, eff_df["F1"], 0.35, color="#4C72B0", label="F1-score", alpha=0.8)
ax2.bar(x_pos + 0.2, eff_df["TrainTime"], 0.35, color="#C44E52",
        label="Train time (s)", alpha=0.8)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(eff_df["Model"])
ax1.set_ylabel("F1-score", color="#4C72B0")
ax2.set_ylabel("Training Time (s)", color="#C44E52")
ax1.set_title("Fig 12 – Model Efficiency: F1-score vs Training Time")
lines1, lbl1 = ax1.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lbl1 + lbl2, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUT_VIZ, "fig12_model_efficiency.png"), dpi=150)
plt.close()
print("  ✓ fig12_model_efficiency.png")

# ── Print full classification report for best model ───────────────────────
best_name = max(results, key=lambda n: results[n]["f1"])
best_pred  = results[best_name]["y_pred"]
print(f"\n  Best model: {best_name}")
print(classification_report(y_test, best_pred,
                             target_names=["On-Time","Delayed"]))

# ── Save results summary to CSV ───────────────────────────────────────────
summary_cols = ["accuracy","precision","recall","f1","roc_auc",
                "cv_f1","train_time_s","model_efficiency"]
summary = pd.DataFrame({n: {k: r[k] for k in summary_cols}
                         for n, r in results.items()}).T
summary.to_csv(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs", "model_results.csv"))
print("\n  ✓ model_results.csv saved")
print("  [04_ml_pipeline.py COMPLETE]")
