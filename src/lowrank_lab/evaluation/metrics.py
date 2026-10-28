from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def classification_metrics(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "class_support": [labels.count(0), labels.count(1)],
    }
