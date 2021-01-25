import logging
import torch

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from typing import List
from typing import Tuple

from src import device
from src.constants import BATCH_SIZE
from src.data.simple_text_dataset import SimpleTextDataset

logger = logging.getLogger(__name__)


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


def predict_texts(model, tokenizer, texts: List[str]) -> Tuple[List[int], List[Tuple[float, float]]]:
    """
    Returns a model inference for each text entry in `texts`

    * List with the index of the predicted class
    * List with the normalized of all classes (normalized)
    """
    acc_y_preds = []
    acc_y_probs = []

    inference_batch_size = 20000

    n = (len(texts) // inference_batch_size) + 1

    for i in range(n):
        a = i * inference_batch_size
        b = (i + 1) * inference_batch_size

        logger.info(f"Inference batch: '{a}' to '{b}'. Batch '{i}/{n}'")

        working_texts = texts[a:b]

        dataset = SimpleTextDataset(working_texts, tokenizer)
        iterator = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        y_pred, y_probs = predict(model, iterator)

        acc_y_preds.extend(y_pred)
        acc_y_probs.extend(y_probs)

    return acc_y_preds, acc_y_probs


