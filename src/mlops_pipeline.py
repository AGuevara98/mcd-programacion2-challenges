import argparse
import mlflow

from src.common.mlflow_utils import setup_mlflow
from src.common.validation import validate_challenge_name
from src.cancer.pipeline import run_cancer_pipeline
from src.nlp_glassdoor.pipeline import run_nlp_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--data_path", required=True)
    args = parser.parse_args()

    validate_challenge_name(args.challenge)
    setup_mlflow()

    with mlflow.start_run(run_name=f"{args.challenge}_run"):
        if args.challenge == "cancer":
            res = run_cancer_pipeline(args.data_path)
        elif args.challenge == "nlp":
            res = run_nlp_pipeline(args.data_path)
        else:
            raise NotImplementedError("Thesis pipeline not yet implemented.")

    print("Pipeline completed successfully.")
    print(res)


if __name__ == "__main__":
    main()