import unittest

import pandas as pd

from src.main import training_job
from src.ops.loss import cross_entropy_loss
from src.ops.weights import calculate_multiclass_weights
from src.text.dataset import create_dataset
from src.text.iterator import create_train_val_test_iterator
from src.text.tokenizer import tokenizer

from src.config import config


class TestFunctionalClassification(unittest.TestCase):

    def test_classification(self):

        datapath = 'data/fake_news_sample.csv'
        output_dir = 'output'

        df = pd.read_csv(datapath)

        dataset = create_dataset(tokenizer=tokenizer, filepath=datapath)

        split_ratio = [0.7, 0.2, 0.1]
        train_iter, valid_iter, test_iter = create_train_val_test_iterator(dataset, split_ratio)

        weights = calculate_multiclass_weights(df['label'])

        config['num-epochs-pretrain'] = 3
        config['num-epochs-train'] = 3

        training_job(
            config=config,
            train_iter=train_iter,
            valid_iter=valid_iter,
            test_iter=test_iter,
            output_dir=output_dir,
            weights=weights
        )


