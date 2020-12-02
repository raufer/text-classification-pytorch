import unittest

import pandas as pd

from src.ops.sample import stratified_split


class TestOpsSample(unittest.TestCase):

    def test_stratified_split(self):

        data = [
            ('A', 0),
            ('A', 0),
            ('A', 0),
            ('A', 0),
            ('A', 0),
            ('A', 0),
            ('A', 1),
            ('A', 1),
            ('A', 1),
            ('A', 1),

            ('A', 0),
            ('A', 0),
            ('A', 0),
            ('A', 0),
            ('A', 0),
            ('A', 0),
            ('A', 1),
            ('A', 1),
            ('A', 1),
            ('A', 1),
        ]
        df = pd.DataFrame(data, columns=['text', 'label'])

        train, val, test = stratified_split(df, 'label', split_ratios=[0.6, 0.3, 0.1])

        self.assertTrue(train.shape[0] == 12)
        self.assertTrue(val.shape[0] == 6)
        self.assertTrue(test.shape[0] == 2)


