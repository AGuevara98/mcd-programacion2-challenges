# Setup & Deployment Instructions

## Quick Start (5 minutes)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/AGuevara98/mcd-programacion2-challenges.git
cd mcd-programacion2-challenges

# 2. Create virtual environment
python -m venv mcd_challenges

# 3. Activate virtual environment
# On Windows (PowerShell):
.\mcd_challenges\Scripts\Activate.ps1

# On Windows (Command Prompt):
mcd_challenges\Scripts\activate.bat

# On macOS/Linux:
source mcd_challenges/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download spaCy models (required for NLP)
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm

# 6. Pre-download pysentimiento weights (recommended for NLP)
python -c "from pysentimiento import create_analyzer; create_analyzer(task='sentiment', lang='en'); create_analyzer(task='sentiment', lang='es'); print('pysentimiento models cached')"

# 7. Verify installation
python -c "import mlflow; print('MLflow installed:', mlflow.__version__)"
```

---

## Detailed Installation Guide

### Step 1: Clone Repository from GitHub

```bash
# HTTPS method (no SSH key required)
git clone https://github.com/AGuevara98/mcd-programacion2-challenges.git

# SSH method (if you have SSH key set up)
git clone git@github.com:AGuevara98/mcd-programacion2-challenges.git

# Navigate to project
cd mcd-programacion2-challenges
```

### Step 2: Create Python Virtual Environment

**Why virtual environment?**
- Isolates project dependencies
- Prevents conflicts with system Python
- Makes reproducibility easier

**Creation:**

**Windows (PowerShell):**
```powershell
python -m venv mcd_challenges
```

**Windows (Command Prompt):**
```cmd
python -m venv mcd_challenges
```

**macOS/Linux:**
```bash
python3 -m venv mcd_challenges
```

### Step 3: Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\mcd_challenges\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
mcd_challenges\Scripts\activate.bat
```

### Step 4: Install Python Dependencies

```bash
# Upgrade pip first (recommended)
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

```bash
# Install Playwright browsers (required for scraping)
python -m playwright install chromium
```

**What gets installed:**
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning models
- `mlflow`: Experiment tracking
- `spacy`: NLP preprocessing
- `playwright`: Web scraping (used by the project scraper)
- `vaderSentiment`: Sentiment analysis
- `pysentimiento`: Spanish sentiment analysis
- Plus 8 more dependencies (see requirements.txt)

### Step 5: Download spaCy Language Models

spaCy models are required for NLP preprocessing:

```bash
# English model (for Glassdoor reviews in English)
python -m spacy download en_core_web_sm

# Spanish model (for Glassdoor reviews in Spanish)
python -m spacy download es_core_news_sm
```

**Model sizes:**
- en_core_web_sm: ~40MB
- es_core_news_sm: ~35MB
- Total download: ~75MB

### Step 6: Pre-download pysentimiento Weights

This is optional but recommended before the first NLP pipeline run so the model download does not happen during execution:

```bash
python -c "from pysentimiento import create_analyzer; create_analyzer(task='sentiment', lang='en'); create_analyzer(task='sentiment', lang='es'); print('pysentimiento models cached')"
```

This warms the local cache for the English and Spanish sentiment models used by the NLP pipeline.


## Running the Pipelines

### Starting MLflow Tracking Server

Open a **new terminal** (keep the original one for running code):

```bash
# Navigate to project directory
cd /path/to/mcd-programacion2-challenges

# Start MLflow server
mlflow ui --host 127.0.0.1 --port 5000
```

**Expected output:**
```
[2026-04-14 10:30:00 +0000] [12345] [INFO] Started server process [12345]
[2026-04-14 10:30:00 +0000] [12345] [INFO] Waiting for application startup
[2026-04-14 10:30:00 +0000] [12345] [INFO] Application startup complete
[2026-04-14 10:30:00 +0000] [12345] [INFO] Uvicorn running on http://127.0.0.1:5000
```

**Access UI:** Open browser to http://127.0.0.1:5000

### Cancer Detection Pipeline

**In original terminal** (where virtual environment is activated):

```bash
# Run cancer detection pipeline
python -m src.mlops_pipeline --challenge cancer --data_path data/raw/cancer.csv
```

**Expected output:**
```
Validated: cancer
MLflow setup complete
Pipeline completed successfully.
{'logistic': {'accuracy': 0.956, 'precision': 0.950, ...}, 'rf': {...}}
```

**Output files created:**
- `data/processed/cancer_segments.csv` (if applicable)
- `data/metrics/cancer_metrics.json`
- `plots/cancer/*.png` (visualizations)
- MLflow artifacts logged to `mlruns/`

### NLP Glassdoor Pipeline

```bash
# Run NLP pipeline
python -m src.mlops_pipeline --challenge nlp --data_path data/raw/glassdoor_reviews.csv
```

**Expected output:**
```
Validated: nlp
MLflow setup complete
Processing Glassdoor reviews...
Expanding segments...
Preprocessing text...
Training classifier...
Running sentiment analysis...
Pipeline completed successfully.
```

**Output files created:**
- `data/processed/nlp_segments.csv`
- `data/processed/nlp_processed_segments.csv`
- `data/processed/nlp_classifier_eval.csv`
- `data/metrics/nlp_metrics.json`
- `plots/nlp/*.png` (visualizations)

---

## Making Predictions

### Load a Trained Model

```python
import mlflow
import pandas as pd

# Set tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load model from run
model = mlflow.sklearn.load_model("runs:/<run_id>/logistic_model")

# Make predictions
X_new = pd.DataFrame([[...]])  # Your data
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
```
