import argparse
import mlflow

from src.common.config import RAW_DATA_DIR
from src.common.mlflow_utils import setup_mlflow
from src.common.validation import validate_challenge_name
from src.cancer.pipeline import run_cancer_pipeline
from src.nlp_glassdoor.pipeline import run_nlp_pipeline
from src.thesis.pipeline import run_thesis_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run ML pipelines for MCD Programación 2 Challenges"
    )
    parser.add_argument("--challenge", required=True, choices=['cancer', 'nlp', 'thesis'],
                        help="Challenge to run: cancer, nlp, or thesis")
    parser.add_argument("--data_path", required=False,
                        help="Path to input data file (CSV)")
    parser.add_argument("--target_column", required=False, default="target",
                        help="Target column name (for thesis challenge)")
    args = parser.parse_args()

    validate_challenge_name(args.challenge)
    setup_mlflow()

    with mlflow.start_run(run_name=f"{args.challenge}_run"):
        try:
            if args.challenge == "cancer":
                if not args.data_path:
                    raise ValueError("--data_path is required for the cancer challenge.")
                res = run_cancer_pipeline(args.data_path)
                
            elif args.challenge == "nlp":
                data_path = args.data_path or str(RAW_DATA_DIR / "glassdoor_reviews.csv")
                res = run_nlp_pipeline(data_path)
                
            elif args.challenge == "thesis":
                if not args.data_path:
                    raise ValueError("--data_path is required for the thesis challenge.")
                res = run_thesis_pipeline(args.data_path, target_column=args.target_column)
            
            print("\n" + "="*60)
            print("Pipeline completed successfully!")
            print("="*60)
            print(res)
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            mlflow.log_param("error", str(e))
            raise


if __name__ == "__main__":
    main()