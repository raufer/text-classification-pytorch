import logging

import pandas as pd

from sklearn.model_selection import train_test_split

from transformers import PreTrainedTokenizer
from src.data.text_dataset import TextDataset

from typing import Tuple
from typing import List

from src.ops.sample import stratified_split

logger = logging.getLogger(__name__)


def create_datasets(tokenizer: PreTrainedTokenizer, filepath: str, split_ratios: List[float]) -> Tuple[TextDataset, TextDataset, TextDataset]:
    """
    Creates a dataset from the file `filepath`
    Each record contains two fields: 'data', 'label'
    split_ratio :: [train, val, test]
    """
    logger.info(f"Creating a dataset from file in '{filepath}'; format '{format}'")
    df = pd.read_csv(filepath)

    train, val, test = stratified_split(df, 'label', split_ratios)

    dataset_train = TextDataset(train, tokenizer)
    dataset_val = TextDataset(val, tokenizer)
    dataset_test = TextDataset(test, tokenizer)

    logger.info(f"Training Set    '{len(dataset_train)}' examples")
    logger.info(f"Validation Set  '{len(dataset_val)}' examples")
    logger.info(f"Test Set        '{len(dataset_test)}' examples")

    return dataset_train, dataset_val, dataset_test

