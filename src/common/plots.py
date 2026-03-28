from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def plot_confusion_matrix(model, X_test, y_test, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(model, X_test, y_test, output_path: str) -> None:
    if not hasattr(model, "predict_proba"):
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)