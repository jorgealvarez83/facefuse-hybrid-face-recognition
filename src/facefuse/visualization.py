"""Publication-style figures for the benchmark results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


def _finish(save_path=None, show=True):
    """Persist and/or display the current figure, then release it."""
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_accuracy_comparison(pure_metrics, hybrid_metrics, save_path=None, show=True):
    accs = [pure_metrics['accuracy'], hybrid_metrics['accuracy']]
    labels = ['Deep only\n(InsightFace)', 'FaceFuse\n(hybrid)']
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, accs, color=['#4c72b0', '#3f9b5b'])
    plt.title('Top-1 accuracy on the probe set')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.3f}", ha='center')
    _finish(save_path, show)


def plot_confusion_matrix(cm, class_names, save_path=None, show=True):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix - FaceFuse hybrid')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    _finish(save_path, show)


def plot_roc_curve(y_true, y_score, class_names, save_path=None, show=True):
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc(fpr, tpr):.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve - FaceFuse hybrid (one-vs-rest)')
    plt.legend(loc="lower right")
    _finish(save_path, show)


def plot_quality_weighting(quality_scale=500.0, save_path=None, show=True):
    """Show how the Laplacian-variance quality score sets the fusion weights."""
    variance = np.linspace(0, quality_scale * 2.0, 400)
    w_deep = np.clip(variance / quality_scale, 0.0, 1.0)

    plt.figure(figsize=(8, 4.5))
    plt.plot(variance, w_deep, label='deep embedding weight', color='#4c72b0', lw=2)
    plt.plot(variance, 1.0 - w_deep, label='LBP + HOG weight', color='#c44e52', lw=2)
    plt.axvline(quality_scale, ls='--', c='grey', lw=1)
    plt.text(quality_scale, 0.5, '  quality_scale', color='grey', va='center')
    plt.xlabel('Laplacian variance of the enhanced crop  (blurred  ->  sharp)')
    plt.ylabel('Fusion weight')
    plt.title('Quality-adaptive fusion weights')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    _finish(save_path, show)
