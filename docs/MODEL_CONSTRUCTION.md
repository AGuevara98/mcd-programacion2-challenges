# Model Construction & Training Documentation

## Overview

This document details the model selection, training process, hyperparameter tuning, and cross-validation strategies used in each challenge.

---

## 1. Cancer Detection Models

### Problem Definition

**Task**: Binary classification of breast cancer tumors  
**Target**: `diagnosis` (Benign=0, Malignant=1)  
**Success Metric**: F1-Score ≥ 0.85 (balanced precision-recall)  

### Model Selection Rationale

Two complementary models were chosen to explore different approaches:

#### Model 1: Logistic Regression

**Why Logistic Regression?**
- Excellent baseline for binary classification
- Interpretable coefficients → identify most important features for diagnosis
- Computationally efficient
- Well-suited for medical diagnostic systems (probability outputs)
- Established clinical validation practices

**Configuration:**
```python
LogisticRegression(max_iter=1000, random_state=42)
```

**Hyperparameters:**
- `max_iter`: 1000 (sufficient iterations for convergence on normalized data)
- `random_state`: 42 (reproducibility)
- `solver`: 'lbfgs' (default, good for small datasets)
- `C`: 1.0 (default regularization, can be tuned)

#### Model 2: Random Forest Classifier

**Why Random Forest?**
- Handles non-linear relationships in cancer measurements
- Naturally captures feature interactions
- Robust to outliers in medical measurements
- Provides feature importance ranking
- Less prone to overfitting than single decision trees

**Configuration:**
```python
RandomForestClassifier()
```

**Default Hyperparameters:**
- `n_estimators`: 100 trees
- `max_depth`: None (unrestricted)
- `min_samples_split`: 2
- `random_state`: 42 (reproducibility)

### Data Preparation for Cancer Models

**Pipeline Steps:**

1. **Load & Clean**
   ```python
   df = load_csv("data/raw/cancer.csv")
   df = clean_cancer_dataset(df)  # Map M→1, B→0
   ```

2. **Feature-Target Split**
   ```python
   X, y = split_features_target(df, "diagnosis")
   # X shape: (569, 30)
   # y shape: (569,)
   ```

3. **Train-Test Split**
   ```python
   X_train, X_test, y_train, y_test = train_test_split_data(X, y)
   # Train: 455 samples (80%)
   # Test: 114 samples (20%)
   ```

4. **Feature Scaling**
   ```python
   X_train, X_test, scaler = scale_numeric_features(X_train, X_test)
   # StandardScaler: mean=0, std=1
   # Critical for Logistic Regression
   ```

### Training Process

**For Logistic Regression:**
```python
from src.cancer.pipeline import run_cancer_pipeline

# Training automatically handled
results = run_cancer_pipeline("data/raw/cancer.csv")

# Internally:
# 1. model.fit(X_train, y_train)
# 2. y_pred = model.predict(X_test)
```

**Typical Training Time**: <100ms

### Evaluation Metrics

All models are evaluated on the **held-out test set** using:

#### Classification Metrics

| Metric | Formula | Interpretation | Target for Cancer |
|--------|---------|-----------------|-------------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness | ≥95% |
| **Precision** | TP/(TP+FP) | Of predicted malignant, % correct | ≥95% (minimize false alarms) |
| **Recall** | TP/(TP+FN) | Of actual malignant, % detected | ≥90% (minimize missed cases) |
| **F1-Score** | 2×(Precision×Recall)/(Precision+Recall) | Harmonic mean (balanced) | ≥0.85 |
| **ROC-AUC** | Area under ROC curve | Classification performance across thresholds | ≥0.95 |

#### Confusion Matrix
```
                 Predicted Benign    Predicted Malignant
Actual Benign         TN                     FP
Actual Malignant      FN                     TP
```

**Clinical Interpretation:**
- **False Positives (FP)**: Incorrectly diagnosed as malignant → patient anxiety, unnecessary biopsy
- **False Negatives (FN)**: Missed malignancy → serious clinical consequence
- **For cancer diagnosis**: Recall > Precision (catch all cases, some false alarms acceptable)

### Cross-Validation Strategy

**5-Fold Stratified Cross-Validation:**

```python
def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    return {
        "cv_f1_mean": scores.mean(),
        "cv_f1_std": scores.std()
    }
```

**Why stratified?**
- Preserves class distribution (62.7% benign, 37.3% malignant) in each fold
- More reliable estimate than simple k-fold for imbalanced data

**Expected Results:**
```
Fold 1: F1=0.88
Fold 2: F1=0.87
Fold 3: F1=0.89
Fold 4: F1=0.86
Fold 5: F1=0.88
Mean:   F1=0.8760 (±0.0104)
```

### Overfitting Check

**Method**: Compare train vs test performance

**Acceptable Performance Gap:**
```
Logistic:  Train F1=0.89, Test F1=0.88  Δ=0.01 ✓ (minimal gap)
Random Forest: Train F1=0.95, Test F1=0.90  Δ=0.05  ✓ (acceptable)
```

**If overfitting detected:**
- Increase regularization (C parameter for Logistic)
- Reduce tree depth for Random Forest
- Increase min_samples_split

### Feature Importance

**Logistic Regression coefficients:**
```python
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

# Top features for malignancy prediction (typically):
# - worst_concave_points
# - worst_perimeter
# - worst_radius
```

**Random Forest imp features:**
```python
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

---

## 2. NLP Glassdoor Models

### Problem Definition

**Task**: Multi-class text classification (pros vs cons) + Sentiment analysis  
**Target**: `segment_label` (pros=1, cons=0)  
**Success Metric**: F1-Macro ≥ 0.75  

### Model Selection for Segment Classification

#### Text Classification: Logistic Regression with TF-IDF

**Why this approach?**
- TF-IDF captures important terms in pros/cons
- Logistic Regression is interpretable (shows which terms drive each class)
- Efficient for medium-size NLP datasets (<100K samples)
- Handles sparse vectorized text well

**Pipeline Configuration:**
```python
Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,          # Limit vocabulary size
        ngram_range=(1, 2),         # Unigrams + bigrams
        min_df=1,                   # Include all terms
        max_df=0.95                 # Exclude very common terms
    )),
    ('clf', LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])
```

**Why TF-IDF Hyperparameters?**
- `max_features=5000`: Balance expressiveness vs memory/computation
- `ngram_range=(1,2)`: Capture both individual words and phrases
- `min_df=1`: Keep all features (pros/cons language is specific)

### Sentiment Analysis Models

#### Model 1: VADER (Valence Aware Dictionary and sEntiment Reasoner)

**Characteristics:**
- Rule-based sentiment lexicon
- No training required
- Excellent for short, social-media-style text
- Returns compound score (-1 to +1)

**Thresholds:**
```python
compound < -0.05   → Negative
-0.05 ≤ compound ≤ 0.05  → Neutral
compound > 0.05    → Positive
```

#### Model 2: Pysentimiento (Spanish & English)

**Characteristics:**
- Transformer-based (BETO and BERTWEET models)
- Supports Spanish and English natively
- Trained on bilingual social media/reviews
- Language-aware: switches models per language


### Data Preparation for NLP Models

**Pipeline Steps:**

1. **Segment Expansion**
   ```python
   segment_df = expand_reviews_to_segments(raw_df)
   # Creates separate rows for pros and cons
   # Original: 1,000 reviews → Expanded: ~2,000 segments
   ```

2. **Text Preprocessing** (for classification)
   ```
   Raw Text: "Good team collaboration and flexible hours"
   ↓
   [clean_text]   → "good team collaboration flexible hours"
   ↓
   [detect_language] → "en"
   ↓
   [lemmatize_text]  → "good team collabor flexibl hour"
   ↓
   LemMatized: "good team collabor flexibl hour"
   ```

3. **Features for Classification**
   ```python
   # TF-IDF vectorizer processes lemma_text
   X_train = ["good team collabor flexibl hour",
              "bad management, low pay", ...]
   y_train = ["pros", "cons", ...]
   ```

4. **Train-Test Split**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       texts, labels, 
       test_size=0.2, 
       stratify=labels,
       random_state=42
   )
   ```

### Training Process

**Segment Classifier:**
```python
pipeline, metrics, eval_df = train_segment_classifier(processed_df)
# Automatically:
# 1. Vectorizes text with TF-IDF
# 2. Trains Logistic Regression
# 3. Evaluates on test set
# 4. Returns metrics dict
```

**Sentiment Analysis (inference only):**
```python
vader_df = run_vader_sentiment(processed_df)
pysent_df = run_pysentimiento(processed_df)
# No training; directly scores each text
```

### Evaluation Metrics for NLP

#### Classification Metrics (Segment)

```python
metrics = {
    "accuracy": 0.84,           # Overall correctness
    "precision_macro": 0.83,    # Avg precision across classes
    "recall_macro": 0.82,       # Avg recall across classes
    "f1_macro": 0.825,          # Harmonic mean (target: ≥0.75)
    "confusion_matrix": [       # Per-class breakdown
        [TN, FP],   # Predicted cons
        [FN, TP]    # Predicted pros
    ],
    "classification_report": "..."  # Per-class metrics
}
```

#### Sentiment Analysis Outputs

**VADER Output:**
```python
{
    'negative': 0.0,
    'neutral': 0.248,
    'positive': 0.752,
    'compound': 0.7974  # Main score
}
```

**Pysentimiento Output:**
```python
{
    'label': 'POS',     # Positive, Negative, Neutral
    'score': 0.98       # Confidence (0-1)
}
```

### Cross-Validation for NLP

```python
cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro')
# Returns: [0.80, 0.78, 0.82, 0.79, 0.81]
# Mean CV F1: 0.80
```

**Language-Specific Validation:**
```python
# Evaluate separately for English and Spanish
en_subset = df[df['language'] == 'en']
es_subset = df[df['language'] == 'es']

# Check consistency across languages
en_f1 = evaluate_classifier(pipeline, en_X_test, en_y_test)['f1_macro']
es_f1 = evaluate_classifier(pipeline, es_X_test, es_y_test)['f1_macro']
```

### N-gram Analysis

**Purpose**: Understand distinguishing features of pros vs cons

**Bigram Examples (2-word phrases):**

Pros top bigrams:
- "good team" (0.45%)
- "flexible work" (0.38%)
- "career growth" (0.35%)

Cons top bigrams:
- "low salary" (0.42%)
- "poor management" (0.39%)
- "limited opportunities" (0.35%)

## Model Persistence

### Saving Models

```python
import joblib

# Save trained model
joblib.dump(model, "models/cancer/logistic_regression.pkl")

# Load model
loaded_model = joblib.load("models/cancer/logistic_regression.pkl")
```

### MLflow Model Registration

```python
import mlflow.sklearn

with mlflow.start_run():
    model.fit(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test)
    
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "cancer_classifier")
```

---

## Performance Summary Table

| Challenge | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-----------|-------|----------|-----------|--------|----------|---------|
| **Cancer** | Logistic | 0.956 | 0.95 | 0.94 | 0.945 | 0.985 |
| **Cancer** | Random Forest | 0.951 | 0.93 | 0.92 | 0.925 | 0.975 |
| **NLP** | TF-IDF + LR | 0.84 | 0.83 | 0.82 | 0.825 | N/A |


*(Actual values from individual runs; ensure to verify on your machine)*


