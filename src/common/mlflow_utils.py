import mlflow

def setup_mlflow():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("mcd_challenges")

def log_params(params: dict):
    mlflow.log_params(params)

def log_metrics(metrics: dict):
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, v)

def log_artifact(path: str):
    mlflow.log_artifact(path)

def log_model(model, example_input, run_name="model"):
    import mlflow.sklearn
    mlflow.sklearn.log_model(model, run_name, input_example=example_input)