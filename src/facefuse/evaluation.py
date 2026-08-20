"""Classification metrics and throughput measurement."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class SystemEvaluator:
    def evaluate(self, y_true, y_pred):
        """Macro-averaged metrics plus a confusion matrix and its axis labels.

        ``labels`` is the sorted union of true and predicted classes, so it
        always matches the confusion matrix. It is wider than the set of true
        identities whenever the recogniser emits "Unknown" - without it the
        matrix axes get mislabelled.
        """
        labels = sorted(set(y_true) | set(y_pred))
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels),
            'labels': labels,
        }

    def compute_fps(self, processing_times):
        if not processing_times:
            return 0.0
        return len(processing_times) / sum(processing_times)