# MLOps Documentation

## Overview

This document explains the Machine Learning Operations (MLOps) infrastructure for tracking, managing, and versioning models and experiments using MLflow.



## MLflow Setup & Configuration

### Installation

MLflow is included in `requirements.txt`:
```bash
pip install mlflow
```

### Configuration

**Location**: `src/common/mlflow_utils.py`

```python
import mlflow

def setup_mlflow():
    # Set tracking server (local SQLite database)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Set experiment name
    mlflow.set_experiment("mcd_challenges")
```

**Configuration Details:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `tracking_uri` | http://127.0.0.1:5000 | Local MLflow server (localhost) |
| `experiment` | mcd_challenges | Experiment group name |
| `backend` | SQLite (mlflow.db) | Lightweight, file-based storage |

---

## Tracking Runs

### What Gets Logged?

Each MLflow run captures:

1. **Parameters**: Model hyperparameters
2. **Metrics**: Performance measurements
3. **Artifacts**: Files (plots, models, data)
4. **Tags**: Key-value metadata
5. **Source**: Code version/Git commit
6. **Timing**: Start/end times

### Logging in Code

#### Example: Cancer Pipeline

```python
import mlflow
from src.common.mlflow_utils import setup_mlflow

def run_cancer_pipeline(path: str):
    setup_mlflow()
    
    # Start a run
    with mlflow.start_run(run_name="cancer_v1"):
        # Load and preprocess data
        df = load_csv(path)
        X_train, X_test, y_train, y_test = prepare_data(df)
        
        # Train model
        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(X_train, y_train)
        
        # Log parameters
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("C", 1.0)
        
        # Evaluate and log metrics
        metrics = evaluate_classifier(model, X_test, y_test)
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        mlflow.sklearn.log_model(model, "logistic_model")
        
        # Log additional artifacts
        mlflow.log_artifact("plots/confusion_matrix.png")
        
    # Run is automatically closed when exiting context manager
```

#### Nested Runs (Multiple Models)

```python
with mlflow.start_run(run_name="cancer_comparison"):
    models = {
        "logistic": LogisticRegression(),
        "rf": RandomForestClassifier(),
    }
    
    for name, model in models.items():
        # Create nested run for each model
        with mlflow.start_run(run_name=name, nested=True):
            model.fit(X_train, y_train)
            metrics = evaluate_classifier(model, X_test, y_test)
            
            mlflow.log_params({"model": name})
            mlflow.log_metrics(metrics)
```

### Utility Functions

**Location**: `src/common/mlflow_utils.py`

```python
def log_params(params: dict):
    """Log multiple parameters at once"""
    mlflow.log_params(params)

def log_metrics(metrics: dict):
    """Log metrics, filtering non-numeric values"""
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, v)

def log_artifact(path: str):
    """Log a file as an artifact"""
    mlflow.log_artifact(path)

def log_model(model, example_input, run_name="model"):
    """Log a scikit-learn model with example input"""
    import mlflow.sklearn
    mlflow.sklearn.log_model(model, run_name, input_example=example_input)
```

---

## Model Signatures

### What is a Model Signature?

A signature defines the expected input/output schema for a model:
- Input column names and types
- Output column names and types
- Enables validation when serving the model

### Automatic Signature Generation

```python
import mlflow.sklearn

# MLflow can infer signature from data
X_sample = X_test.iloc[:5]  # Example input
mlflow.sklearn.log_model(
    model, 
    "my_model",
    input_example=X_sample  # Generates signature automatically
)
```

### Manual Signature Definition

```python
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# Define input schema
input_schema = Schema([
    ColSpec("double", "radius_mean"),
    ColSpec("double", "texture_mean"),
    # ... other features
])

# Define output schema
output_schema = Schema([
    ColSpec("long", "diagnosis_pred"),  # 0 or 1
    ColSpec("double", "probability"),   # 0.0 to 1.0
])

signature = ModelSignature(input_schema, output_schema)

mlflow.sklearn.log_model(
    model,
    "cancer_model",
    signature=signature
)
```

---

## Metrics Logging

### Standard Classification Metrics

**Logged for each model run:**

```python
metrics = {
    "accuracy": 0.956,           # (TP+TN)/(Total)
    "precision": 0.950,          # TP/(TP+FP)
    "recall": 0.940,             # TP/(TP+FN)
    "f1_score": 0.945,           # 2*(P*R)/(P+R)
    "roc_auc": 0.985,            # Area under ROC curve
    "cv_f1_mean": 0.876,         # 5-fold CV mean
    "cv_f1_std": 0.0104,         # 5-fold CV std dev
}

mlflow.log_metrics(metrics)
```

### Metric Naming Conventions

```
# Per-model metrics
cancer/logistic/accuracy: 0.956
cancer/rf/accuracy: 0.951

# Per-cross-validation-fold
cv_fold_1_f1: 0.88
cv_fold_2_f1: 0.87

# Per-class metrics (multi-class)
precision_class_0: 0.93
recall_class_1: 0.92
```

### Custom Scalar Metrics

```python
# Log during training (e.g., epoch-wise)
for epoch in range(num_epochs):
    loss = train_one_epoch()
    mlflow.log_metric("train_loss", loss, step=epoch)
    
    val_loss = validate()
    mlflow.log_metric("val_loss", val_loss, step=epoch)
```

---

## Artifact Management
### Logging Artifacts

#### Single File

```python
mlflow.log_artifact("plots/cancer_confusion_matrix.png")
# Stored in: mlruns/<exp_id>/<run_id>/artifacts/
```

#### Directory

```python
mlflow.log_artifacts("plots/")  # All files in directory
# Stored in: mlruns/<exp_id>/<run_id>/artifacts/plots/
```

#### Text Content

```python
import mlflow

# Save text as artifact
mlflow.log_text("Model trained successfully on 569 samples", "training_log.txt")

# Save dictionary as JSON
mlflow.log_dict({"accuracy": 0.956, "f1": 0.945}, "metrics.json")
```

### Saving Plots as Artifacts

```python
import matplotlib.pyplot as plt
import mlflow

# Create confusion matrix plot
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
plt.title("Cancer Classifier - Confusion Matrix")
plt.savefig("/tmp/confusion_matrix.png", dpi=300, bbox_inches="tight")

# Log as artifact
mlflow.log_artifact("/tmp/confusion_matrix.png")
plt.close(fig)
```

### Organizing Artifacts

```
mlruns/
├── 0/              # Experiment 0
│   └── meta.yaml
├── 2/              # Experiment "mcd_challenges"
│   ├── meta.yaml
│   ├── <run_id_1>/
│   │   ├── params/
│   │   ├── metrics/
│   │   └── artifacts/
│   │       ├── logistic_model/
│   │       ├── plots/
│   │       │   ├── confusion_matrix.png
│   │       │   ├── feature_importance.png
│   │       │   └── roc_curve.png
│   │       └── metrics.json
│   └── <run_id_2>/
```

---

## Pipeline Execution with MLOps

### Full Pipeline with Tracking

```python
# src/mlops_pipeline.py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", required=True)  # cancer, nlp, thesis
    parser.add_argument("--data_path", required=False)
    args = parser.parse_args()
    
    # Setup MLflow
    setup_mlflow()
    
    # Create parent run for the entire pipeline
    with mlflow.start_run(run_name=f"{args.challenge}_pipeline"):
        try:
            # Log pipeline parameters
            mlflow.log_param("challenge", args.challenge)
            mlflow.log_param("timestamp", datetime.now().isoformat())
            
            # Run appropriate challenge pipeline
            if args.challenge == "cancer":
                results = run_cancer_pipeline(args.data_path)
                
            elif args.challenge == "nlp":
                results = run_nlp_pipeline(args.data_path)
                
            # Log results
            mlflow.log_metrics(results)
            
            print("Pipeline completed successfully")
            
        except Exception as e:
            mlflow.log_param("error", str(e))
            raise
```

### Running with MLOps

```bash
# Start MLflow UI (another terminal)
mlflow ui --host 127.0.0.1 --port 5000

# Run pipeline (logs to MLflow)
python -m src.mlops_pipeline --challenge cancer --data_path data/raw/cancer.csv

# View results at http://127.0.0.1:5000
```

---

## MLflow UI Navigation

### Starting the UI

```bash
# Terminal 1: Start MLflow server
mlflow ui --host 127.0.0.1 --port 5000

# Terminal 2: Run your pipelines
cd /path/to/project
python -m src.mlops_pipeline ...
```

### Accessing the UI

Navigate to: **http://127.0.0.1:5000**

### UI Sections

#### 1. Experiments Tab
- Lists all experiments (e.g., "mcd_challenges")
- Shows run count and last update time
- Click experiment to see all runs

#### 2. Runs View
- Table of all runs in experiment
- Columns: run name, start time, duration, parameters, metrics
- Click run for detailed view

#### 3. Run Details
- **Parameters**: All logged hyperparameters
- **Metrics**: Performance metrics with plots
- **Artifacts**: Saved models, plots, files
- **Logs**: System output and errors

#### 4. Run Comparison
- Select multiple runs
- Compare parameter/metric differences
- Useful for hyperparameter tuning

### Metric Visualization

MLflow automatically creates plots:
- **Line plots**: Metric trends across runs
- **Parallel coordinates**: Parameter-metric relationships
- **Scatter plots**: Cross-metric correlation

---

## Model Registry

### Registering Models

```python
import mlflow

# During a run
logged_model_uri = mlflow.get_artifact_uri("model")

mlflow.register_model(
    model_uri=logged_model_uri,
    name="cancer_classifier"
)
```

### Model Versions & Stages

```python
client = mlflow.tracking.MlflowClient()

# Transition model to "Production"
client.transition_model_version_stage(
    name="cancer_classifier",
    version=1,
    stage="Production"
)

# Load production model
model = mlflow.pyfunc.load_model("models:/cancer_classifier/production")
```


## MLflow Database

### Storage Backend

**Default**: SQLite database at project root
```
mlflow.db          # Local SQLite storage
mlruns/            # Artifact directory
```

**File structure:**
```
mlflow.db              # Lightweight, suitable for local development
├── experiments table
├── runs table
└── params/metrics/tags tables
```

