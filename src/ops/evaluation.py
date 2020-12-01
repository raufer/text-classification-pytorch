import torch

from sklearn.metrics import f1_score
from torchtext.data import Iterator

from src.text.tokenizer import PAD_INDEX

from typing import List
from typing import Tuple


def evaluate(model, test_iter: Iterator) -> Tuple[List, List]:
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
        for (source, target), _ in test_iter:
            mask = (source != PAD_INDEX).type(torch.uint8)

            output = model(source, attention_mask=mask)

            y_pred.extend(torch.argmax(output, axis=-1).tolist())
            y_true.extend(target.tolist())

    return y_true, y_pred


