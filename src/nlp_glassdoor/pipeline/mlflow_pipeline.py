import mlflow
import json

def log_metrics_from_json(json_path: str, run_name: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with mlflow.start_run(run_name=run_name):
        if "metrics" in data:
            metrics = data["metrics"]
        else:
            metrics = data.get("weighted avg", {})

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)


def main():
    mlflow.set_experiment("nlp_glassdoor")

    log_metrics_from_json(
        "reports/nlp_glassdoor/metrics/baseline_metrics_en.json",
        "baseline_en",
    )

    log_metrics_from_json(
        "reports/nlp_glassdoor/metrics/baseline_metrics_es.json",
        "baseline_es",
    )

    log_metrics_from_json(
        "reports/nlp_glassdoor/metrics/vader_metrics_en.json",
        "vader_en",
    )

    log_metrics_from_json(
        "reports/nlp_glassdoor/metrics/pysentimiento_metrics_es.json",
        "pysentimiento_es",
    )


if __name__ == "__main__":
    main()