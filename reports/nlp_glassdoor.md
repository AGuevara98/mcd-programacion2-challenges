# NLP Glassdoor Sentiment Analysis Report

## 1. Objective
The objective of this project is to analyze employee reviews from Glassdoor to build a bilingual sentiment classification pipeline (English and Spanish). The project includes data collection, preprocessing, exploratory analysis, supervised modeling, comparison with pretrained sentiment models, and experiment tracking using MLflow.

---

## 2. Data Collection

Reviews were scraped manually from Glassdoor for multiple companies, extracting:
- Pros
- Cons

These were labeled as:
- **Pros → Positive**
- **Cons → Negative**

### Raw Data Files
- `data/raw/glassdoor_targets.csv`
- `data/raw/glassdoor_reviews.csv`

---

## 3. Data Processing

### Validation
- Removed null or empty entries
- Standardized text fields

### Dataset Construction
- Each review split into individual rows
- Assigned labels:
  - `pros → positive`
  - `cons → negative`

### Language Detection
- Dataset split into:
  - English (`glassdoor_en.csv`)
  - Spanish (`glassdoor_es.csv`)

---

## 4. Preprocessing

Applied separately for each language.

### Steps
- Lowercasing
- Stopword removal
- Tokenization
- Lemmatization

### Outputs
- `glassdoor_en_clean.csv`
- `glassdoor_es_clean.csv`

---

## 5. Exploratory Analysis

### N-grams
Top unigrams and bigrams were extracted for both languages.

### Key Observations
- Positive reviews focus on:
  - Work environment
  - Benefits
  - Flexibility
- Negative reviews focus on:
  - Management
  - Salary
  - Workload

---

## 6. Modeling

### Approach
- TF-IDF vectorization
- Logistic Regression classifier

### Results

#### English
- Accuracy: **0.875**
- F1 (macro): **0.875**

#### Spanish
- Accuracy: **0.884**
- F1 (macro): **0.883**

### Interpretation
- Strong performance despite small dataset
- Balanced predictions across classes
- TF-IDF + Logistic Regression is effective for this task

---

## 7. Sentiment Model Comparison

### Models Used
- English: VADER
- Spanish: pysentimiento

### Results

#### English (VADER)
- Accuracy: **0.709**
- Weak negative recall (~0.52)

#### Spanish (pysentimiento)
- Accuracy: **0.648**
- Strong negative detection
- Weak positive recall (~0.47)

### Key Findings
- Pretrained models underperform compared to supervised models
- Domain mismatch affects performance
- Neutral class mismatch impacts evaluation

---

## 8. MLOps (MLflow)

Tracked experiments:
- `baseline_en`
- `baseline_es`
- `vader_en`
- `pysentimiento_es`

### Logged
- Metrics (accuracy, precision, recall, F1)
- Model comparisons
- Reproducible runs

---

## 9. Conclusions

- Supervised models outperform rule-based and pretrained approaches
- Language-specific preprocessing improves results
- Domain-specific data is critical for performance
- MLflow enables reproducibility and comparison

---

## 10. Limitations

- Small dataset size (~200 samples per language)
- Binary labels only (no neutral ground truth)
- Limited company diversity

---

## 11. Future Work

- Increase dataset size
- Add neutral class
- Try advanced models (e.g., BERT)
- Improve class balance
- Deploy as an API or dashboard

---

## 12. Repository Structure (Summary)

- `data/`: datasets and metrics
- `plots/`: visualizations
- `src/`: pipeline and modules
- `models/`: placeholder for trained models
- `docs/`: documentation

---

## 13. Reproducibility

Main commands:

```bash
python -m src.nlp_glassdoor.data.validate_data
python -m src.nlp_glassdoor.data.prepare_dataset
python -m src.nlp_glassdoor.data.split_languages
python -m src.nlp_glassdoor.models.train
python -m src.nlp_glassdoor.sentiment.compare_sentiment
python -m src.nlp_glassdoor.pipeline.mlflow_pipeline