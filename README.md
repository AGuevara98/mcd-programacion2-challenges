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

The NLP workflow now uses live scraping with a browser session.

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



# Installation

## 1. Create environment

```bash
python -m venv mcd_challenges
```

## 2. Activate

### Windows (PowerShell)

```bash
mcd_challenges\Scripts\Activate
```

### Linux / WSL

```bash
source mcd_challenges/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Install Playwright browser

```bash
playwright install chromium
```



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



# Final Workflow Summary

```text
Glassdoor (manual login)
        ↓
scraper_manual_login.py
        ↓
data/raw/glassdoor_reviews.csv
        ↓
mlops_pipeline.py (nlp)
        ↓
models + metrics + plots
```
