import argparse
import mlflow

from src.common.config import RAW_DATA_DIR
from src.common.mlflow_utils import setup_mlflow
from src.common.validation import validate_challenge_name
from src.cancer.pipeline import run_cancer_pipeline
from src.nlp_glassdoor.pipeline import run_nlp_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--data_path", required=False)
    args = parser.parse_args()

    validate_challenge_name(args.challenge)
    setup_mlflow()

    with mlflow.start_run(run_name=f"{args.challenge}_run"):
        if args.challenge == "cancer":
            if not args.data_path:
                raise ValueError("--data_path is required for the cancer challenge.")
            res = run_cancer_pipeline(args.data_path)
        elif args.challenge == "nlp":
            data_path = args.data_path or str(RAW_DATA_DIR / "glassdoor_reviews.csv")
            res = run_nlp_pipeline(data_path)
        else:
            raise NotImplementedError("Thesis pipeline not yet implemented.")

    print("Pipeline completed successfully.")
    print(res)


if __name__ == "__main__":
    main()