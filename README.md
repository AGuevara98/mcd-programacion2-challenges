# MCD Programación 2 — Challenges

This repository contains the implementation of three challenges for the Master’s in Data Science program. Each challenge follows a structured, reproducible pipeline with clear separation between data, models, and code.



# Repository Structure

```text
.
│   README.md
│   requirements.txt
│   upload_results.py
│
├── cache/                         # local cache files
├── data/                          # datasets and processed outputs
│   ├── raw/
│   ├── processed/
│   └── metrics/
├── docs/                          # project documentation and setup guides
├── mlartifacts/                   # persisted ML artifacts (local store)
├── mlruns/                        # MLflow tracking DB/artifacts
├── models/                        # saved trained models
├── plots/                         # visualizations
├── mcd_challenges/                # local virtualenv (not committed)
└── src/                           # source code
  │   mlops_pipeline.py
  │
  ├── cancer/
  │       pipeline.py
  │       # model training + evaluation
  │
  ├── common/
  │       config.py
  │       evaluation.py
  │       io_utils.py
  │       mlflow_utils.py
  │       preprocessing.py
  │       validation.py
  │
  ├── nlp_glassdoor/
  │       scraper_manual_login.py
  │       preprocessing.py
  │       model.py
  │       pipeline.py
  │
  └── thesis/                     # Thesis geospatial PoC pipeline
      __init__.py
      pipeline.py             # Phase 1-3 orchestration
      network_synthesis.py    # Steiner tree network optimization
      modeling.py             # LightGBM + RF training
      critic.py               # CRITIC weighting
      data_access.py          # PostGIS/DuckDB connectors
      credentials_template.py # Credential template (copy to credentials.py)
      credentials.py          # Database credentials (not in git)
```



# Challenges Overview

## 1. Cancer Detection (ML)

* Dataset: Breast cancer dataset
* Task: Binary classification (malignant vs benign)
* Output:

  * trained model
  * evaluation metrics
  * MLflow tracking



## 2. NLP — Glassdoor Reviews

* Dataset: Glassdoor reviews collected with a live browser session
* Task:

  * sentiment analysis
  * text preprocessing
  * feature extraction (n-grams, etc.)



## 3. Thesis — Transit Route Suitability PoC

* Dataset: PostGIS geospatial features (AGEB accessibility, employment, vitality)
* Task: Multi-phase pipeline (CRITIC weighting → predictive modeling → network optimization)
* Output:

  * suitability scores per AGEB
  * optimized transit corridors
  * MLflow tracking with SHAP explainability



## 4. General MLOps Pipeline

* Unified execution through:

```bash
python -m src.mlops_pipeline --challenge <name>
```



# ⚠️ Important: Glassdoor Data Collection

The NLP workflow uses live scraping with a browser session.

Requirements:

* a valid Glassdoor account session
* `data/raw/glassdoor_targets.csv` with `company` and `url` columns
* Playwright browser binaries installed (`playwright install chromium`)



# NLP Pipeline Workflow

Run order:

1. Scrape latest reviews into `data/raw/glassdoor_reviews.csv`
2. Run the NLP pipeline

The scraper is a separate manual/browser-assisted step and is not invoked by `src.mlops_pipeline.py`. The NLP pipeline expects the CSV to already exist and only processes the prepared data.

If `data/raw/glassdoor_reviews.csv` is already up to date, you can skip Step 1.

## Step 1 — Scrape Glassdoor reviews into CSV

Run:

```bash
python -m src.nlp_glassdoor.scraper_manual_login \
        --targets-file data/raw/glassdoor_targets.csv \
        --output-file data/raw/glassdoor_reviews.csv
```

Output:

```text
data/raw/glassdoor_reviews.csv
```

This file contains:

* company
* review_title
* pros
* cons
* source_url
* page_number



## Step 2 — Run NLP pipeline

```bash
python -m src.mlops_pipeline --challenge nlp
```

Outputs:

* processed data → `data/processed/`
* metrics → `data/metrics/`
* model → `models/nlp/`
* plots → `plots/nlp/`



# Cancer Pipeline Workflow

Run:

```bash
python -m src.mlops_pipeline --challenge cancer --data_path data/raw/cancer.csv
```

Outputs:

* model → `models/cancer/`
* metrics → `data/metrics/`
* plots → `plots/cancer/`


# Thesis Pipeline Workflow — Transit Route Suitability

This is a **3-phase geospatial PoC** that predicts transit route suitability in Guadalajara (ZMG) using multi-criteria weighting, LightGBM modeling with SHAP explainability, and Steiner-tree network optimization.

## Prerequisites

### 1. Database Setup (WSL/Linux with PostgreSQL + PostGIS)

The thesis pipeline requires a **PostgreSQL database with PostGIS extension** running in WSL (Windows Subsystem for Linux) or natively on Linux.

**To set up the database locally:**

1. Clone the database setup repository:
```bash
git clone https://github.com/AGuevara98/predictive-transit-zmg.git
cd predictive-transit-zmg
```

2. Follow the setup instructions in [`SETUP.md`](https://github.com/AGuevara98/predictive-transit-zmg/blob/main/SETUP.md) to:
   - Install PostgreSQL and PostGIS in WSL
   - Load base geospatial data (AGEB boundaries, route supply, employment, accessibility)
   - Initialize the `gdl_metro` database with `raw`, `base`, and `features` schemas

3. Verify the database is running:
```bash
# In WSL
pg_isready -h localhost -p 5432
```

### 2. Configure Credentials

The thesis pipeline reads PostgreSQL credentials from `src/thesis/credentials.py`, which is **NOT committed to git** for security.

**To set up credentials:**

1. Copy the template:
```bash
cp src/thesis/credentials_template.py src/thesis/credentials.py
```

2. Edit `src/thesis/credentials.py` with your database connection details:
```python
PG_USER = "your_username"
PG_PASS = "your_password"
PG_HOST = "localhost"        # or your WSL IP
PG_PORT = "5432"
PG_DB = "gdl_metro"
```

3. The file is automatically excluded from git (see `.gitignore`)

## Running the Pipeline

### Quick Run (Default: PostGIS Data)

```bash
python -m src.mlops_pipeline --challenge thesis
```

This loads suitability data directly from the `features.thesis_ageb_scored` table in PostGIS.


## Pipeline Output

The thesis pipeline produces three phases of outputs:

### Phase 1: CRITIC Weighting & Labeling
- Computes objective multi-criteria weights for NP-RV (Node-Place-Real Estate-Vitality) indicators
- Auto-labels AGEBs as Balanced/Unbalanced if target column missing

### Phase 2: Predictive Modeling with MLflow
- Trains LightGBM and Random Forest models with 5-fold cross-validation
- Logs metrics, SHAP summary plots, ROC curves, and feature importance to MLflow
- Persists scored AGEBs to `features.thesis_ageb_scored` in PostGIS

**MLflow Outputs:**
```
Experiment: thesis_run
├── lightgbm_model (nested run)
│   ├── Metrics: accuracy, precision, recall, F1, ROC-AUC
│   ├── Artifacts: shap_summary.png, roc_curve.png, feature_importance.csv
│   └── Model: lightgbm_model with input examples
└── random_forest_model (nested run)
    ├── Metrics: (same as LightGBM)
    ├── Artifacts: (same as LightGBM)
    └── Model: random_forest_model with input examples
```

Access results at: **http://localhost:5000**

### Phase 3: Steiner Tree Network Synthesis
- Builds suitability-weighted street network from OSMnx
- Selects top 12 high-suitability AGEBs as terminals
- Runs Steiner tree approximation to find optimal corridors connecting terminals
- Persists optimized corridors to `features.thesis_steiner_corridors` in PostGIS (1,275+ edges)

**PostGIS Output Tables:**
```sql
-- Scored AGEBs with suitability predictions
SELECT * FROM features.thesis_ageb_scored LIMIT 10;

-- Optimized transit corridors
SELECT * FROM features.thesis_steiner_corridors LIMIT 10;
```

## Monitoring Progress

The pipeline includes detailed progress logging, especially during long-running Steiner synthesis:

```
[08:43:12] [steiner] Initializing Steiner run from PostGIS
[08:43:20] [steiner] Loading boundary geometry from base.ageb
[08:44:14] [steiner] Graph ready: 81958 nodes, 197399 edges
[08:44:26] [steiner] Mapping 2068 centroids to nearest graph nodes
[08:50:17] [steiner] Suitability edge weighting complete
[08:50:20] [steiner] Building undirected graph for Steiner approximation with 12 terminals
[08:50:22] [steiner] Steiner result: 759 nodes, 758 edges
[08:50:26] [steiner] Steiner corridor GeoDataFrame contains 1275 edges
```

## Troubleshooting

### "Connection refused" or "Host unreachable"

**Issue**: PostgreSQL in WSL unreachable from Windows.

**Solution**: 
- Verify WSL PostgreSQL is running: `wsl -d Ubuntu -e pg_isready -h localhost`
- Pipeline automatically falls back to WSL IP if localhost fails
- Or manually set `PG_HOST` in `credentials.py` to your WSL's IP address (e.g., `192.168.1.100`)

### "Table not found" error

**Issue**: Required feature tables not in PostGIS.

**Solution**: 
- Ensure database setup completed (see [predictive-transit-zmg setup](https://github.com/AGuevara98/predictive-transit-zmg))
- Tables required: `base.ageb`, `features.ageb_accessibility`, `features.ageb_employment`, `features.ageb_route_supply`, `features.ageb_economic_activity`
- Pipeline will synthesize missing tables from available ones if possible

### "credentials.py not found"

**Issue**: Credentials file missing.

**Solution**:
```bash
cp src/thesis/credentials_template.py src/thesis/credentials.py
# Edit with your database details
```


---



## 📚 Documentation

**Complete documentation is available in the `docs/` directory:**

- **[docs/README.md](docs/README.md)** - Documentation index and quick start guide
- **[docs/SETUP_INSTRUCTIONS.md](docs/SETUP_INSTRUCTIONS.md)** - Detailed installation and deployment guide
- **[docs/DATASET_DOCUMENTATION.md](docs/DATASET_DOCUMENTATION.md)** - All three datasets explained
- **[docs/MODEL_CONSTRUCTION.md](docs/MODEL_CONSTRUCTION.md)** - Model details, training, hyperparameters
- **[docs/MLOPS_DOCUMENTATION.md](docs/MLOPS_DOCUMENTATION.md)** - MLflow tracking and experiment management
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture and data flow diagrams
- **[docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md)** - Code organization and extension guide

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- Python 3.8+
- Git

### Installation

```bash
# 1. Clone repository
git clone https://github.com/AGuevara98/mcd-programacion2-challenges.git
cd mcd-programacion2-challenges

# 2. Create virtual environment
python -m venv mcd_challenges
source mcd_challenges/bin/activate  # Linux/macOS
# or
.\mcd_challenges\Scripts\Activate.ps1  # Windows

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm

# 4. Start MLflow (in separate terminal)
mlflow ui --host 127.0.0.1 --port 5000

# 5. Run cancer pipeline
python src/mlops_pipeline.py --challenge cancer --data_path data/raw/cancer.csv

# 6. View results at http://127.0.0.1:5000
```

**For detailed instructions**: See [docs/SETUP_INSTRUCTIONS.md](docs/SETUP_INSTRUCTIONS.md)

---

## 📊 Challenge Overview

### 1. Cancer Detection 🏥

**Binary classification of breast cancer tumors**

- **Dataset**: Breast Cancer Wisconsin (Diagnostic)
- **Samples**: 569 with 30 features
- **Task**: Classify as Malignant (M) or Benign (B)
- **Models**: Logistic Regression, Random Forest
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

```bash
python src/mlops_pipeline.py --challenge cancer --data_path data/raw/cancer.csv
```

**Documentation**: [docs/MODEL_CONSTRUCTION.md#cancer](docs/MODEL_CONSTRUCTION.md#1-cancer-detection-models)

---

### 2. NLP Glassdoor Analysis 💬

**Text classification and sentiment analysis of company reviews**

- **Dataset**: Glassdoor company reviews (bilingual: English & Spanish)
- **Task**: Classify segments as pros/cons + sentiment analysis
- **Models**: TF-IDF + Logistic Regression (classification), VADER, Pysentimiento (sentiment)
- **Features**: Language detection, lemmatization, n-gram analysis
- **Metrics**: Accuracy, Precision, Recall, F1-Score (macro)

```bash
python src/mlops_pipeline.py --challenge nlp --data_path data/raw/glassdoor_reviews.csv
```

**Documentation**: [docs/MODEL_CONSTRUCTION.md#nlp](docs/MODEL_CONSTRUCTION.md#2-nlp-glassdoor-models)

---

### 3. Thesis Challenge 🎓

**Transit route suitability geospatial PoC (ZMG)**

- **Framework**: 3-phase pipeline (CRITIC weighting -> predictive modeling -> network synthesis)
- **Data**: PostGIS features for AGEB accessibility, employment, route supply, and vitality
- **Models**: LightGBM + Random Forest with SHAP explainability and MLflow tracking
- **Output**: AGEB suitability scores + optimized Steiner corridor network

```bash
# Default: load directly from PostGIS
python src/mlops_pipeline.py --challenge thesis

# Optional: run with custom input
python src/mlops_pipeline.py --challenge thesis --data_path data/my_dataset.csv --target_column label
```

---

## 🔧 Key Features

✅ **Three complete ML challenges** with reproducible pipelines  
✅ **Full MLOps integration** with MLflow experiment tracking  
✅ **Bilingual NLP support** (English & Spanish)  
✅ **Web scraping** capability for Glassdoor reviews  
✅ **Comprehensive documentation** (~20,000 words)  
✅ **Modular, extensible architecture** for custom projects  
✅ **Best practices** in data science and software engineering  
✅ **Cross-platform support** (Windows, macOS, Linux)  

---

## 📁 Project Structure

```
mcd-programacion2-challenges/
├── src/                          # Source code
│   ├── mlops_pipeline.py        # Entry point for all challenges
│   ├── cancer/                  # Cancer detection challenge
│   ├── nlp_glassdoor/           # NLP Glassdoor challenge
│   ├── thesis/                  # Thesis geospatial pipeline
│   └── common/                  # Shared utilities
├── data/                        # Data storage
│   ├── raw/                     # Raw data (cancer.csv, glassdoor_reviews.csv)
│   ├── processed/               # Processed data
│   └── metrics/                 # Evaluation metrics (JSON)
├── models/                      # Saved trained models
├── plots/                       # Visualizations
├── docs/                        # Complete documentation (6 files)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── mlflow.db                    # MLflow tracking database
```


## 📋 Dependencies

**Core Libraries:**
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Machine learning
- `mlflow` - Experiment tracking
- `spacy` - NLP preprocessing
- `selenium`, `beautifulsoup4` - Web scraping
- `matplotlib` - Visualization
- `vaderSentiment`, `pysentimiento` - Sentiment analysis

**See**:`requirements.txt` for full list with versions

---

## 🚀 Running Pipelines

### Option 1: Command Line Interface

```bash
# Cancer challenge
python src/mlops_pipeline.py --challenge cancer --data_path data/raw/cancer.csv

# NLP challenge
python src/mlops_pipeline.py --challenge nlp --data_path data/raw/glassdoor_reviews.csv

# Thesis challenge (default PostGIS)
python src/mlops_pipeline.py --challenge thesis

# Thesis challenge (custom data)
python src/mlops_pipeline.py --challenge thesis --data_path data/my_data.csv --target_column target
```

### Option 2: Python API

```python
from src.cancer.pipeline import run_cancer_pipeline
from src.nlp_glassdoor.pipeline import run_nlp_pipeline
from src.thesis.pipeline import run_thesis_pipeline

# Run any pipeline
results = run_cancer_pipeline("data/raw/cancer.csv")
print(results)
```


---

## 📊 MLflow Experiment Tracking

All pipelines automatically track metrics and artifacts to MLflow.

```bash
# Start MLflow UI
mlflow ui --host 127.0.0.1 --port 5000

# Web interface: http://127.0.0.1:5000
```

**Features:**
- Real-time metrics dashboard
- Hyperparameter comparison
- Model versioning
- Artifact storage
- Run history

**Documentation**: [docs/MLOPS_DOCUMENTATION.md](docs/MLOPS_DOCUMENTATION.md)

---

## 🧪 Model Performance

### Cancer Detection Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.956 | 0.950 | 0.940 | 0.945 | 0.985 |
| Random Forest | 0.951 | 0.930 | 0.920 | 0.925 | 0.975 |

### NLP Classification Results

| Metric | Value |
|--------|-------|
| Accuracy (Micro) | 0.84 |
| Precision (Macro) | 0.83 |
| Recall (Macro) | 0.82 |
| F1-Score (Macro) | 0.825 |

---

## 🔍 Data Sources

### Cancer Dataset
- **UCI Machine Learning Repository** - Breast Cancer Wisconsin (Diagnostic)
- **569 samples**, 30 features, binary classification
- **Source**: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

### Glassdoor Reviews
- Web-scraped from Glassdoor company pages
- **Bilingual** (English & Spanish)
- **variable size** depending on scraping run
- **Terms**: See Glassdoor Terms of Service

### Thesis Data (Transit Suitability)
- Primary source: PostGIS database (gdl_metro) with base and features schemas
- Optional source: CSV or SQL/table references via --data_path
- Includes geospatial indicators for NP-RV-style suitability modeling

**Full details**: [docs/DATASET_DOCUMENTATION.md](docs/DATASET_DOCUMENTATION.md)


# Notes on Scraping

* Direct automated scraping of Glassdoor is unreliable due to:

  * Cloudflare protection
  * dynamic rendering
* This project uses:

```text
manual login → live scrape → NLP pipeline
```

This approach is:

* reproducible
* stable
* sufficient for the academic requirements



# Deliverables Summary

Each challenge produces:

## Cancer

* trained model
* evaluation metrics
* MLflow logs

## NLP

* cleaned dataset
* sentiment model
* evaluation metrics
* n-gram analysis
* predictions

## General

* reproducible pipeline
* modular structure
* consistent outputs across challenges
        ↓
mlops_pipeline.py (nlp)
        ↓
models + metrics + plots
```

---

## 📤 Uploading Results to GitHub

Once your pipelines complete and generate results, upload them automatically:

```bash
python upload_results.py
```

This command:
1. Stages all modified files (`git add .`)
2. Creates a commit with timestamp
3. Pushes changes to GitHub (`main` branch)

**No manual git commands needed!**
