import unittest

import pandas as pd

from src.models.electra_classifier import ElectraClassifier
from src.models.legalbert_classifier import LegalBertClassifier
from src.models.roberta_classifier import ROBERTAClassifier

from src.tokenizer.electra import make_electra_tokenizer
from src.tokenizer.legalbert import make_legalbert_tokenizer
from src.tokenizer.roberta import make_roberta_tokenizer

from src.ops.weights import calculate_multiclass_weights
from src.data.dataset import create_datasets
from src.data.iterator import make_iterators

from src.main import training_job
from src.config import config


class TestFunctionalClassification(unittest.TestCase):

    def test_classification(self):

        datapath = '/Users/raulferreira/pytorch-roberta-classifier/data/fake_news_sample.csv'
        output_dir = '/Users/raulferreira/pytorch-roberta-classifier/output'

        df = pd.read_csv(datapath)

        tokenizer = make_roberta_tokenizer()
        model = ROBERTAClassifier(dropout_rate=config['dropout-ratio'], n_outputs=2)

        split_ratios = [0.7, 0.2, 0.1]
        train, val, test = create_datasets(tokenizer=tokenizer, filepath=datapath, split_ratios=split_ratios)
        train_iter, valid_iter, test_iter = make_iterators(train, val, test)

        weights = calculate_multiclass_weights(df['label'])

        config['num-epochs-pretrain'] = 3
        config['num-epochs-train'] = 3

        training_job(
            config=config,
            model=model,
            train_iter=train_iter,
            valid_iter=valid_iter,
            test_iter=test_iter,
            output_path=output_dir,
            weights=weights
        )

    def test_classification_electra(self):

        datapath = '/Users/raulferreira/pytorch-roberta-classifier/data/fake_news_sample.csv'
        output_dir = '/Users/raulferreira/pytorch-roberta-classifier/output'

        df = pd.read_csv(datapath)

        tokenizer = make_electra_tokenizer()

        split_ratios = [0.7, 0.2, 0.1]

        train, val, test = create_datasets(tokenizer=tokenizer, filepath=datapath, split_ratios=split_ratios)
        train_iter, valid_iter, test_iter = make_iterators(train, val, test)

        weights = calculate_multiclass_weights(df['label'])

        config['num-epochs-pretrain'] = 3
        config['num-epochs-train'] = 3

        model = ElectraClassifier(dropout_rate=config['dropout-ratio'])

        training_job(
            config=config,
            model=model,
            train_iter=train_iter,
            valid_iter=valid_iter,
            test_iter=test_iter,
            output_path=output_dir,
            weights=weights
        )

    def test_classification_legal_bert(self):

        datapath = '/Users/raulferreira/pytorch-roberta-classifier/data/fake_news_sample.csv'
        output_dir = '/Users/raulferreira/pytorch-roberta-classifier/output'

        df = pd.read_csv(datapath)
        n_outputs = df['label'].nunique()

        tokenizer = make_legalbert_tokenizer()

        split_ratios = [0.7, 0.2, 0.1]
        train, val, test = create_datasets(tokenizer=tokenizer, filepath=datapath, split_ratios=split_ratios)
        train_iter, valid_iter, test_iter = make_iterators(train, val, test)

        weights = calculate_multiclass_weights(df['label'])

        config['num-epochs-pretrain'] = 3
        config['num-epochs-train'] = 3

        model = LegalBertClassifier(dropout_rate=config['dropout-ratio'], n_outputs=n_outputs)

        training_job(
            config=config,
            model=model,
            train_iter=train_iter,
            valid_iter=valid_iter,
            test_iter=test_iter,
            output_path=output_dir,
            weights=weights
        )

