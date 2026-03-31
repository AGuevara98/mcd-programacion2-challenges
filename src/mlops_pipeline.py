import argparse
import mlflow

from src.common.mlflow_utils import setup_mlflow
from src.common.validation import validate_challenge_name
from src.cancer.pipeline import run_cancer_pipeline
from nlp_glassdoor.pipeline_2 import run_nlp_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run MCD UDG challenge pipeline")
    parser.add_argument("--challenge", required=True, help="cancer | nlp | thesis")
    parser.add_argument("--data_path", required=False, help="Path to input dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_challenge_name(args.challenge)
    setup_mlflow()

    with mlflow.start_run(run_name=f"{args.challenge}_run"):
        if args.challenge == "cancer":
            if not args.data_path:
                raise ValueError("Cancer challenge requires --data_path")
            results = run_cancer_pipeline(args.data_path)

        elif args.challenge == "nlp":
            if not args.data_path:
                raise ValueError("NLP challenge requires --data_path")
            results = run_nlp_pipeline(args.data_path)

        elif args.challenge == "thesis":
            raise NotImplementedError("Thesis pipeline not yet implemented.")

        else:
            raise ValueError(f"Unexpected challenge: {args.challenge}")

    print("Pipeline completed successfully.")
    print(results)


if __name__ == "__main__":
    main()