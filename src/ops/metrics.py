import os
import logging

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report as sk_classification_report

from typing import List


logger = logging.getLogger(__name__)


def classification_report(y_true: List, y_pred: List) -> None:
    """
    Calculates and logs the classification report
    """
    report = sk_classification_report(y_true, y_pred, labels=[1, 0], digits=4)
    logger.info("Classification Report")
    logger.info(report)


def write_confusion_matrix(y_true: List, y_pred: List, output_path: str) -> None:
    """
    Creates and writes confusion matrix plot to `output_path`
    """
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")
    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['reporting-buren', 'no-burden'])
    ax.yaxis.set_ticklabels(['reporting-buren', 'no-burden'])

    png_path = os.path.join(output_path, 'confusion-matrix.png')
    logger.info(f"Saving Confusion Matrix Plot to '{png_path}'")
    plt.savefig(png_path)
