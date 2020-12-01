import logging

from src import device
from src.constants import BATCH_SIZE

from torchtext.data import BucketIterator
from torchtext.data import TabularDataset
from torchtext.data import Iterator

from typing import List
from typing import Tuple


logger = logging.getLogger(__name__)


def create_train_val_test_iterator(dataset: TabularDataset, split_ratio: List[int]) -> Tuple[Iterator, Iterator, Iterator]:
    """
    Given a dataset splits it into a train, validation and test sets
    according to `split_ratio`

    The split is stratified according to the label column
    Assumes `dataset` has the following schema:
    text (string) | label (int)
    """
    logger.info(f"Splitting the dataset into training, validation and test")

    train_data, valid_data, test_data = dataset.split(
        split_ratio=split_ratio,
        stratified=True,
        strata_field='label'
    )

    train_iter, valid_iter = BucketIterator.splits(
        (train_data, valid_data),
        batch_size=BATCH_SIZE,
        device=device,
        shuffle=True,
        sort_key=lambda x: len(x.text),
        sort=True,
        sort_within_batch=False
    )

    test_iter = Iterator(test_data, batch_size=BATCH_SIZE, device=device, train=False, shuffle=False, sort=False)

    logger.info(f"Training Set    '{len(train_data)}' examples")
    logger.info(f"Validation Set  '{len(valid_data)}' examples")
    logger.info(f"Test Set        '{len(test_data)}' examples")

    return train_iter, valid_iter, test_iter
