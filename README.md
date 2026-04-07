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
            local_html_parser.py   # Converts downloaded HTML → CSV
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

* Dataset: Glassdoor reviews (manually collected HTML → parsed to CSV)
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

The repository **does NOT include raw HTML files** due to:

* size constraints
* scraping restrictions
* reproducibility limitations

You must collect them manually.



# How to Collect Glassdoor Data

1. Go to a company’s Glassdoor reviews page
   Example:

   ```text
   https://www.glassdoor.com/Reviews/Softtek-Reviews-E5766.htm
   ```

2. Scroll to load multiple reviews

3. Right-click → **Save Page As** → “Webpage, Complete”

4. Save files into:

```text
data/raw/fullsites/
```

You should end up with multiple `.html` or `.htm` files.



# NLP Pipeline Workflow

## Step 1 — Parse HTML files into CSV

Run:

```bash
python -m src.nlp_glassdoor.local_html_parser \
  --input_folder data/raw/fullsites \
  --output_csv data/raw/glassdoor_reviews.csv
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
* rating
* review_date
* language



## Step 2 — Run NLP pipeline

```bash
python -m src.mlops_pipeline \
  --challenge nlp \
  --data_path data/raw/glassdoor_reviews.csv
```

Outputs:

* processed data → `data/processed/`
* metrics → `data/metrics/`
* model → `models/nlp/`
* plots → `plots/nlp/`



# Cancer Pipeline Workflow

Run:

```bash
python -m src.mlops_pipeline --challenge cancer
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



# Notes on Scraping

* Direct automated scraping of Glassdoor is unreliable due to:

  * Cloudflare protection
  * dynamic rendering
* This project uses:

```text
manual download → local parsing → NLP pipeline
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
Glassdoor (manual download)
        ↓
data/raw/fullsites/*.html
        ↓
local_html_parser.py
        ↓
data/raw/glassdoor_reviews.csv
        ↓
mlops_pipeline.py (nlp)
        ↓
models + metrics + plots
```
