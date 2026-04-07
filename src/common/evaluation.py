from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score

def evaluate_classifier(model, X, y):
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "roc_auc": roc_auc_score(y, model.predict_proba(X)[:,1]),
    }

def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    return {"cv_f1_mean": scores.mean(), "cv_f1_std": scores.std()}