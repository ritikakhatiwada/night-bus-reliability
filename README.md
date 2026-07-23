# 🚌 Night Bus Service Reliability Prediction Platform
### ST5011CEM – Big Data Programming Project | Softwarica College of IT & E-Commerce

---

## Project Overview

**Project Title:** Predictive Analytics for Night Bus Service Reliability  
**Stakeholder:** Transport Authority / Local Council  
**Business Problem:** Night bus services suffer from irregular delays, disruptions, and service unreliability between 22:00–05:59. This platform uses large-scale PySpark processing and machine learning to predict trip non-compliance (delayed/on-time), enabling proactive operations management.

---

## Project Structure

```
night_bus_project/
│
├── config/
│   └── config.py              # SparkSession settings, DB path, metric thresholds
│
├── data/                      # Generated CSV files (created at runtime)
│   ├── timetables.csv         # 80,000 rows – scheduled/actual journey data
│   ├── disruptions.csv        # 15,000 rows – disruption events
│   └── locations.csv          # 25,000 rows – GPS pings
│
├── database/
│   ├── schema.sql             # SQLite schema + sample parameterised queries
│   └── night_bus.db           # SQLite database (created at runtime)
│
├── models/                    # Saved scikit-learn model files
│   ├── Logistic_Regression.pkl
│   ├── Random_Forest.pkl
│   └── Gradient_Boosting.pkl
│
├── outputs/
│   └── model_results.csv      # Model evaluation summary
│
├── src/
│   ├── 01_data_generation.py  # Synthetic data generation (Faker + NumPy)
│   ├── 02_spark_ingestion.py  # PySpark ingestion, cleaning, joins, SQLite write
│   ├── 03_eda_visualisation.py# EDA + 7 visualisations
│   ├── 04_ml_pipeline.py      # 3 ML models + full evaluation + plots
│   ├── 05_spark_ml_pipeline.py# PySpark MLlib pipeline (VectorAssembler + CrossValidator)
│   └── 06_dashboard.py        # Streamlit dashboard
│
├── visualisations/            # 12 output charts (PNG)
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- Java 8 or 11 (required for PySpark)

### Installation

```bash
# Clone the repository
git clone <your-github-repo-url>
cd night_bus_project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline (in order)

```bash
# Step 1: Generate synthetic data (120,000 records)
python src/01_data_generation.py

# Step 2: PySpark ingestion, feature engineering, SQLite persistence
python src/02_spark_ingestion.py

# Step 3: EDA + visualisations
python src/03_eda_visualisation.py

# Step 4: Train & evaluate 3 ML models
python src/04_ml_pipeline.py

# Step 5: PySpark MLlib pipeline (CrossValidator)
python src/05_spark_ml_pipeline.py

# Step 6: Launch Streamlit dashboard (optional)
streamlit run src/06_dashboard.py
```

---

## Technology Stack

| Layer | Tool | Justification |
|---|---|---|
| Large-scale processing | PySpark 3.5 | Distributed transforms, SQL, 8 partitions |
| Data generation | Pandas + Faker | In-memory; Faker has no Spark integration |
| EDA (computation) | PySpark SQL | groupBy, agg, window functions at scale |
| EDA (plotting) | matplotlib / seaborn | Final step only — toPandas() before plot |
| ML training | scikit-learn | Full suite of metrics, pipeline API |
| ML (Spark demo) | PySpark MLlib | VectorAssembler + CrossValidator |
| Database | SQLite + SQLAlchemy | Parameterised queries; no hard-coded creds |
| Dashboard | Streamlit | Rapid interactive UI; no JS required |
| Version control | Git / GitHub | Full commit history |

---

## Big Data Configuration

```python
# SparkSession configured with:
master              = "local[4]"   # 4 worker threads
executor_memory     = "2g"
driver_memory       = "2g"
shuffle_partitions  = 8
default_parallelism = 8

# Demonstrated:
# ✓ Repartitioning to 8 partitions by service_code
# ✓ df_tt.cache() on hot DataFrame
# ✓ Broadcast joins for small lookup tables
# ✓ df_tt.unpersist() after use
# ✓ Lazy evaluation + DAG (no action until .count()/.show())
```

**Total records: 120,000** (exceeds ≥100,000 threshold)  
**Date window: Oct 2024 – Jan 2025 (3 months)**

---

## Metric Definitions (from brief)

| Metric | Definition | Target | This Project |
|---|---|---|---|
| Service Reliability | % trips within ±2 min | ≥85% | ~32% (night buses underperform) |
| Travel Time CV | std / mean trip duration | ≤15% | 0.42 (highly variable) |
| Completion Rate | trips completed / scheduled | ≥95% | 100% (synthetic) |
| Model Efficiency | F1 / training time | Report all | See outputs/model_results.csv |

---

## ML Models Compared

| Model | Accuracy | F1 | ROC-AUC | Train Time | Big-O |
|---|---|---|---|---|---|
| Logistic Regression | 50.9% | 0.589 | 0.511 | 0.05s | O(n·d·k) |
| Random Forest | 59.0% | 0.710 | 0.510 | 7.09s | O(n·d·log(n)·T) |
| Gradient Boosting | 68.1% | 0.810 | 0.507 | 14.37s | O(n·d·log(n)·T·M) |

**Best model:** Gradient Boosting (F1 = 0.810)

---

## Security Considerations

- All database queries use **parameterised statements** via SQLAlchemy `text()` — no string concatenation
- No hard-coded credentials — database path set in `config/config.py`
- GDPR consideration: synthetic data only; no real passenger PII stored
- SparkSession credentials documented in config, not in source code

---

## Ethical & Legal Considerations

- **GDPR**: Real BODS data may contain operator/vehicle identifiers. Aggregation should be preferred over individual trip tracking.
- **Bias**: Night bus service data skews toward urban routes; rural operators may be underrepresented.
- **Accessibility**: Reliability metrics must consider journeys by vulnerable passengers who depend on night services.

---

## References

- Bus Open Data Service (BODS): https://data.bus-data.dft.gov.uk/
- Apache Spark Documentation: https://spark.apache.org/docs/3.5.0/
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*.
- UK Department for Transport (2023). *Bus Open Data: Technical Specification*.
- Faker library: https://faker.readthedocs.io/
