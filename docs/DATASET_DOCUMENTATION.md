# Dataset Documentation

## Overview

This document provides comprehensive information about the datasets used in the three ML challenges of the MCD Programación 2 project.

---

## 1. Cancer Detection Challenge

### Dataset: Breast Cancer Wisconsin (Diagnostic)

#### Data Source
- **Source**: UCI Machine Learning Repository - Breast Cancer Wisconsin Dataset
- **Link**: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
- **Source File**: `data/raw/cancer.csv`

#### Dataset Information

| Attribute | Value |
|-----------|-------|
| **Samples** | 569 observations |
| **Features** | 30 numeric features (originally 32, 2 are IDs) |
| **Target Variable** | `diagnosis` (Binary: M=Malignant, B=Benign) |
| **Missing Values** | None |
| **Data Type** | Float/Integer |

#### Feature Description

The dataset contains computed measurements of mean, standard error, and worst values for:

1. **Measurements** (3 statistics × 10 characteristics):
   - Radius (mean, se, worst)
   - Texture (mean, se, worst)
   - Perimeter (mean, se, worst)
   - Area (mean, se, worst)
   - Smoothness (mean, se, worst)
   - Compactness (mean, se, worst)
   - Concavity (mean, se, worst)
   - Concave Points (mean, se, worst)
   - Symmetry (mean, se, worst)
   - Fractal Dimension (mean, se, worst)

Total: 30 features derived from digital image analysis of fine needle aspirate (FNA) of breast mass.

#### Class Distribution

| Class | Count | Percentage |
|-------|-------|-----------|
| Benign (0) | 357 | 62.7% |
| Malignant (1) | 212 | 37.3% |

**Note**: The classes are imbalanced with benign cases slightly more prevalent.

#### Data Quality Issues & Solutions

| Issue | Detection | Solution |
|-------|-----------|----------|
| **Missing Values** | None detected | N/A |
| **Duplicate Rows** | Checked during preprocessing | Removed via `basic_cleaning()` |
| **Outliers** | Present in extreme measurements | Handled by StandardScaler normalization |
| **Data Type Issues** | Diagnosis column stored as string "M"/"B" | Mapped to 1/0 in `clean_cancer_dataset()` |
| **Feature Scaling** | Different units and ranges | StandardScaler applied before modeling |

#### Data Processing Pipeline

```
Raw Data (cancer.csv)
    ↓
[validate_file_exists]
    ↓
Load CSV → DataFrame
    ↓
[clean_cancer_dataset] - Map M→1, B→0
    ↓
[split_features_target] - Separate X, y
    ↓
[train_test_split_data] - 80/20 split, random_state=42
    ↓
[scale_numeric_features] - StandardScaler normalization
    ↓
Ready for Model Training
```

#### Usage Example

```python
from src.common.io_utils import load_csv
from src.cancer.pipeline import clean_cancer_dataset
from src.common.preprocessing import split_features_target

# Load and clean
df = load_csv("data/raw/cancer.csv")
df = clean_cancer_dataset(df)

# Prepare features and target
X, y = split_features_target(df, "diagnosis")

# Check data info
print(f"Features shape: {X.shape}")  # (569, 30)
print(f"Target distribution:\n{y.value_counts()}")
```

---

## 2. NLP Glassdoor Challenge

### Dataset: Glassdoor Companies Feedback Analysis

#### Data Sources

##### Primary Data
- **Source**: Glassdoor company reviews (Web-scraped)
- **Source Files**: 
  - `data/raw/glassdoor_reviews.csv` (processed reviews)
  - `data/raw/glassdoor_targets.csv` (company metadata)
  - `data/raw/fullsites/` (HTML archives for manual verification)

#### Dataset Information

| Attribute | Value |
|-----------|-------|
| **Samples** | Variable (depends on scraping run) |
| **Languages** | English & Spanish |
| **Key Fields** | company, review_title, pros, cons, rating, review_date |
| **Text Type** | Employee feedback (short to medium length) |

#### Data Structure

**glassdoor_reviews.csv columns:**

| Column | Type | Description |
|--------|------|-------------|
| `company` | String | Company name |
| `review_title` | String | Review title/headline |
| `pros` | String | Pros/advantages of working at company (can be null) |
| `cons` | String | Cons/disadvantages of working at company (can be null) |
| `rating` | Float | Overall rating (typically 1-5 scale) |
| `review_date` | String | Date of review |
| `source_url` | String | URL where review was extracted |
| `page_number` | Integer | Page number in Glassdoor pagination |

**glassdoor_targets.csv columns:**

| Column | Type | Description |
|--------|------|-------------|
| `company` | String | Company name (join key) |
| `industry` | String | Industry classification |
| `size` | String | Company size (Small, Medium, Large, etc.) |
| `founded_year` | Integer | Year company was founded |

#### Data Characteristics

**Text Structure:**
- **Pros**: 2-10 sentences, highlighting positive aspects
- **Cons**: 2-10 sentences, highlighting negative aspects
- **Languages**: Approximately 70% English, 30% Spanish (varies by dataset)
- **Length**: 50-500 characters per segment

**Example Pro Text:**
> "Good team collaboration and learning opportunities. Flexible work arrangements and supportive management."

**Example Con Text:**
> "Limited growth opportunities in certain departments. Compensation could be more competitive in the current market."

#### Language Distribution

The dataset contains bilingual content:

| Language | Typical % | Processing |
|----------|-----------|-----------|
| English | ~70% | Uses `en_core_web_sm` spaCy model |
| Spanish | ~30% | Uses `es_core_news_sm` spaCy model |
| Unknown | <1% | Marked as "unknown", minimal impact |

#### Data Quality Issues & Solutions

| Issue | Detection | Solution |
|-------|-----------|----------|
| **Missing Pros/Cons** | Some reviews have null values | Handled by optional expansion in `expand_reviews_to_segments()` |
| **HTML Entities** | Review text may contain HTML entities | Cleaned by BeautifulSoup during scraping |
| **Non-ASCII Characters** | Accented characters in Spanish text | Preserved in `clean_text()` with character set: `[a-záéíóúñü\s]` |
| **URLs/Emails** | May appear in cons/pros text | Regex pattern removes: `http\S+\|www\.\S+` |
| **Special Characters** | Punctuation, symbols | Removed during `clean_text()` |
| **Language Detection Errors** | Inconsistent/short text | Handled with try-except in `detect_language()` |
| **Stop Words** | High frequency, low semantic value | Removed via spaCy stop words during lemmatization |
| **Duplicates** | Identical reviews possible | Not explicitly removed (dataset quality dependent) |

#### Data Processing Pipeline

```
Raw Glassdoor CSV
    ↓
[validate_file_exists & schema check]
    ↓
Load CSV → DataFrame
    ↓
[expand_reviews_to_segments]
  - Separate pros and cons
  - Create segment_label (pros/cons)
  - Save: nlp_segments.csv
    ↓
[preprocess_segment_dataset]
  - [detect_language] → language column
  - [clean_text] → clean_text column
  - [lemmatize_text] → lemma_text column
  - Calculate text_length
  - Save: nlp_processed_segments.csv
    ↓
[train_segment_classifier]
  - TF-IDF vectorization
  - Logistic Regression training
  - Evaluation metrics
  - Save: nlp_classifier_eval.csv
    ↓
[Sentiment Analysis]
  - [run_vader_sentiment]
  - [run_pysentimiento]
    ↓
[N-gram Analysis & Visualization]
  - Build bigram/trigram summaries
  - Save metrics: nlp_metrics.json
  - Generate plots
```

#### Usage Example

```python
from src.nlp_glassdoor.pipeline import run_nlp_pipeline

# Run complete NLP pipeline
results = run_nlp_pipeline("data/raw/glassdoor_reviews.csv")

# Access processed data
import pandas as pd
processed_df = pd.read_csv("data/processed/nlp_processed_segments.csv")
print(f"Total segments: {len(processed_df)}")
print(f"Language distribution:\n{processed_df['language'].value_counts()}")
```

---

## 3. Thesis Challenge

### Dataset: Thesis-Specific Data

#### Planned Data Sources (Customizable)

The Thesis Challenge is designed as a flexible framework for any classification/regression task.

**Recommended domains:**
- Medical/Healthcare data (similar to cancer detection)
- Financial/Banking data (fraud detection, credit scoring)
- Environmental data (climate, pollution monitoring)
- Social media sentiment analysis
- Customer behavior/churn prediction

#### Generic Dataset Requirements

| Aspect | Requirement |
|--------|------------|
| **Format** | CSV with columns: features + target |
| **Minimum Samples** | 100-500 (depending on feature count) |
| **Feature Types** | Numeric features preferred; categorical can be encoded |
| **Target Variable** | Classification (binary/multi-class) or Regression |
| **Missing Values** | Minimal; can be handled with imputation |
| **Imbalance** | Use class weights or sampling strategies if severe |

#### Customization Points

The Thesis pipeline is modular and supports:

1. **Custom preprocessing**: Modify `preprocess_thesis_data()`
2. **Model selection**: Add any scikit-learn compatible model
3. **Feature engineering**: Create domain-specific features
4. **Evaluation metrics**: Choose appropriate metrics for domain
5. **MLOps logging**: Track custom parameters and metrics

---

## Data Organization

### Directory Structure

```
data/
├── raw/
│   ├── cancer.csv                    # Cancer detection raw data
│   ├── glassdoor_reviews.csv        # Scraped Glassdoor reviews
│   ├── glassdoor_targets.csv        # Company metadata
│   └── fullsites/                   # HTML archives (backup)
│
├── processed/
│   ├── nlp_segments.csv             # Expanded pros/cons segments
│   ├── nlp_processed_segments.csv   # Preprocessed NLP data
│   ├── nlp_classifier_eval.csv      # Classifier evaluation results
│   └── [thesis_processed.csv]       # Thesis processed data (if used)
│
└── metrics/
    ├── cancer_metrics.json           # Cancer model evaluation metrics
    ├── nlp_metrics.json              # NLP model evaluation metrics
    └── [thesis_metrics.json]         # Thesis metrics (if used)
```

### File Size Reference

| File | Approx. Size | Records |
|------|-------------|---------|
| cancer.csv | 100 KB | 569 |
| glassdoor_reviews.csv | 5-50 MB | 1,000-10,000+ |
| nlp_processed_segments.csv | 10-100 MB | 2,000-20,000+ |
| cancer_metrics.json | 50 KB | 2 models |
| nlp_metrics.json | 100-500 KB | Per run |
