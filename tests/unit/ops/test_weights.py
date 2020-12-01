import unittest

import pandas as pd

from src.ops.weights import calculate_multiclass_weights


class TestOpsLabels(unittest.TestCase):

    def test_calculate_labels_weights(self):

        data = [
            ('A', 0), ('A', 0), ('A', 0), ('A', 0), ('A', 0),
            ('A', 1), ('A', 1), ('A', 1), ('A', 1), ('A', 1),
            ('A', 2), ('A', 2), ('A', 2), ('A', 2), ('A', 2),
            ('A', 3), ('A', 3), ('A', 3), ('A', 3), ('A', 3)
        ]
        df = pd.DataFrame(data, columns=['value', 'label'])
        result = calculate_multiclass_weights(df['label'])
        expected = [1.0, 1.0, 1.0, 1.0]
        self.assertListEqual(result, expected)

        labels = [0]*2741 + [1]*37919 + [2]*22858 + [3]*31235 + [4]*5499
        result = calculate_multiclass_weights(labels)
        expected = [13.83, 1.0, 1.66, 1.21, 6.9]
        for a, b in zip(result, expected):
            self.assertAlmostEqual(a, b, places=2)

        labels = [0]*40 + [1]*60
        result = calculate_multiclass_weights(labels)
        expected = [1.5, 1.0]
        for a, b in zip(result, expected):
            self.assertAlmostEqual(a, b, places=2)
