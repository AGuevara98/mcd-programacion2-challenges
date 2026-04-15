# MCD Programación 2 — Challenges

This repository contains the implementation of three challenges for the Master’s in Data Science program. Each challenge follows a structured, reproducible pipeline with clear separation between data, models, and code.



# Repository Structure

```text
.
│   README.md
│   requirements.txt
│
├── data
│   ├── raw              # Input data (CSV, manually collected HTML-derived data)
│   ├── processed        # Cleaned / intermediate datasets
│   └── metrics          # Model evaluation outputs (JSON, CSV)
│
├── models
│   ├── cancer           # Saved cancer models
│   └── nlp              # Saved NLP models
│
├── plots
│   ├── cancer           # Visualizations for cancer challenge
│   └── nlp              # Visualizations for NLP challenge
│
└── src
    │   mlops_pipeline.py
    │
    ├── cancer
    │       pipeline.py
    │
    ├── common
    │       config.py
    │       evaluation.py
    │       io_utils.py
    │       mlflow_utils.py
    │       preprocessing.py
    │       validation.py
    │
    └── nlp_glassdoor
            scraper_manual_login.py # Live Glassdoor scraper with manual login
            preprocessing.py
            model.py
            pipeline.py
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



## 3. General MLOps Pipeline

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

**Generic ML pipeline template for custom projects**

- **Framework**: Flexible classification/regression pipeline
- **Features**: Automatic feature encoding, configurable preprocessing
- **Customizable**: Data, models, metrics, features
- **Use Cases**: Any structured data classification/regression task

```bash
python src/mlops_pipeline.py --challenge thesis --data_path data/my_dataset.csv --target_column label
```

**Documentation**: [docs/CODE_STRUCTURE.md#thesis](docs/CODE_STRUCTURE.md#thesis-challenge-thesis-pipelinepy)

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
│   ├── thesis/                  # Thesis template
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

### Option 3: Docker

```bash
# Build image
docker build -t mcd-challenges:latest .

# Run container
docker run -p 5000:5000 -v ./mlruns:/app/mlruns mcd-challenges:latest
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

### Custom Thesis Data
- User-provided dataset in CSV format
- Flexible schema (any features + target column)
- Automatically preprocessed and analyzed

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
