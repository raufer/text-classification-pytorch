import torch

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from typing import List
from typing import Tuple

from transformers import PreTrainedTokenizer


def evaluate(model, test_iter: DataLoader) -> Tuple[List, List]:
    """
    Given a model and a test iterator
    returns the model prediction and the true labels

    y_pred :: (test_iter.shape[0])
    y_true :: (test_iter.shape[0])
    """
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():

        for batch in test_iter:

            target = batch.pop('target')

            output = model(**batch)

            y_pred.extend(torch.argmax(output, axis=-1).tolist())
            y_true.extend(target.tolist())

    return y_true, y_pred


