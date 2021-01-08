import torch

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from typing import List
from typing import Tuple

from src import device


def predict(model, iterator: DataLoader) -> Tuple[List, List]:
    """
    Given a model and a data iterator
    returns the model prediction

    y_pred :: (iter.shape[0])
    """
    y_pred = []
    y_probs = []

    model.eval()

    sm = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for batch in iterator:
            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(**batch)

            y_pred.extend(torch.argmax(output, axis=-1).tolist())
            y_probs.extend(sm(output).tolist())

    return y_pred, y_probs


