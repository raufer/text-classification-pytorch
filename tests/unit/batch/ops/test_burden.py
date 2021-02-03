import dictdiffer
import unittest

from src.batch.ops.burden import document_burden_view


class TestBatchOpsBurden(unittest.TestCase):

    def test_burden_document_view(self):

        labels = {
            0: 'No Burden',
            1: 'Detail',
            2: 'Reporting',
            3: 'Standards'
        }

        data = [
            ('A0', 'B0', [''], 2, [0.1, 0.2, 0.5, 0.2]),
            ('A0', 'B1', [''], 0, [0.7, 0.5, 0.2, 0.2]),
            ('A1', 'B2', [''], 3, [0.0, 0.0, 0.2, 0.8]),
            ('A1', 'B3', [''], 3, [0.1, 0.1, 0.1, 0.7]),
            ('A2', 'B4', [''], 2, [0.0, 0.0, 0.8, 0.2]),
            ('A2', 'B5', [''], 2, [0.1, 0.1, 0.7, 0.1]),
            ('A2', 'B6', [''], 2, [0.1, 0.1, 0.7, 0.1]),
            ('A3', 'B7', [''], 3, [0.0, 0.0, 0.2, 0.8]),
            ('A3', 'B8', [''], 1, [0.0, 0.7, 0.1, 0.1]),
            ('A3', 'B9', [''], 3, [0.1, 0.1, 0.1, 0.7]),
            ('A4', 'B10', [''], 3, [0.0, 0.0, 0.2, 0.8]),
            ('A4', 'B11', [''], 2, [0.1, 0.1, 0.7, 0.1]),
            ('A4', 'B12', [''], 3, [0.1, 0.1, 0.1, 0.7]),
            ('A5', 'B13', [''], 1, [0.1, 0.7, 0.1, 0.1]),
            ('A6', 'B14', [''], 0, [0.7, 0.1, 0.1, 0.1]),
            ('A7', 'B15', [''], 1, [0.0, 0.0, 0.2, 0.8]),
            ('A7', 'B16', [''], 3, [0.1, 0.1, 0.1, 0.7]),
            ('A7', 'B17', [''], 2, [0.1, 0.1, 0.7, 0.1]),
            ('A7', 'B18', [''], 2, [0.1, 0.1, 0.7, 0.1]),
            ('A7', 'B19', [''], 1, [0.1, 0.7, 0.1, 0.1])
        ]

        result = document_burden_view(data, labels, bridges=['Detail'], ignore=['No Burden', 'Detail'])

        expected = [
            {
                'node-ids': ['B0'],
                'class': 'Reporting',
                'probabilities': [
                    {'No Burden': 0.1, 'Detail': 0.2, 'Reporting': 0.5, 'Standards': 0.2}
                ]
            },
            {
                'node-ids': ['B2', 'B3'],
                'class': 'Standards',
                'probabilities': [
                    {'No Burden': 0.0, 'Detail': 0.0, 'Reporting': 0.2, 'Standards': 0.8},
                    {'No Burden': 0.1, 'Detail': 0.1, 'Reporting': 0.1, 'Standards': 0.7}
                ]
            },
            {
                'node-ids': ['B4', 'B5', 'B6'],
                'class': 'Reporting',
                'probabilities': [
                    {'No Burden': 0.0, 'Detail': 0.0, 'Reporting': 0.8, 'Standards': 0.2},
                    {'No Burden': 0.1, 'Detail': 0.1, 'Reporting': 0.7, 'Standards': 0.1},
                    {'No Burden': 0.1, 'Detail': 0.1, 'Reporting': 0.7, 'Standards': 0.1},
                ]
            },
            {
                'node-ids': ['B7', 'B8'],
                'class': 'Standards',
                'probabilities': [
                    {'No Burden': 0.0, 'Detail': 0.0, 'Reporting': 0.2, 'Standards': 0.8},
                    {'No Burden': 0.0, 'Detail': 0.7, 'Reporting': 0.1, 'Standards': 0.1},
                ]
            },
            {
                'node-ids': ['B9'],
                'class': 'Standards',
                'probabilities': [
                    {'No Burden': 0.1, 'Detail': 0.1, 'Reporting': 0.1, 'Standards': 0.7},
                ]
            },
            {
                'node-ids': ['B10'],
                'class': 'Standards',
                'probabilities': [
                    {'No Burden': 0.0, 'Detail': 0.0, 'Reporting': 0.2, 'Standards': 0.8},
                ]
            },
            {
                'node-ids': ['B11'],
                'class': 'Reporting',
                'probabilities': [
                    {'No Burden': 0.1, 'Detail': 0.1, 'Reporting': 0.7, 'Standards': 0.1},
                ]
            },
            {
                'node-ids': ['B12'],
                'class': 'Standards',
                'probabilities': [
                    {'No Burden': 0.1, 'Detail': 0.1, 'Reporting': 0.1, 'Standards': 0.7},
                ]
            },
            {
                'node-ids': ['B16'],
                'class': 'Standards',
                'probabilities': [
                    {'No Burden': 0.1, 'Detail': 0.1, 'Reporting': 0.1, 'Standards': 0.7},
                ]
            },
            {
                'node-ids': ['B17', 'B18', 'B19'],
                'class': 'Reporting',
                'probabilities': [
                    {'No Burden': 0.1, 'Detail': 0.1, 'Reporting': 0.7, 'Standards': 0.1},
                    {'No Burden': 0.1, 'Detail': 0.1, 'Reporting': 0.7, 'Standards': 0.1},
                    {'No Burden': 0.1, 'Detail': 0.7, 'Reporting': 0.1, 'Standards': 0.1},
                ]
            }
        ]

        for r, e in zip(result, expected):
            for diff in list(dictdiffer.diff(r, e)):
                print(diff)
            self.assertDictEqual(r, e)

    def test_burden_document_view_2(self):

        labels = {
            0: 'No Burden',
            1: 'Detail',
            2: 'Reporting',
            3: 'Standards'
        }

        data = [
            ('A0', 'B0', [''], 3, [0.1, 0.1, 0.1, 0.7]),
            ('A0', 'B1', [''], 3, [0.1, 0.1, 0.1, 0.7]),
            ('A0', 'B2', [''], 1, [0.1, 0.7, 0.1, 0.1]),
            ('A0', 'B3', [''], 1, [0.1, 0.7, 0.1, 0.1]),
            ('A0', 'B4', [''], 3, [0.1, 0.1, 0.1, 0.7]),
            ('A0', 'B5', [''], 2, [0.1, 0.1, 0.7, 0.1])
        ]

        result = document_burden_view(data, labels, bridges=['Detail'], ignore=['No Burden', 'Detail'])

        expected = [
            {
                'node-ids': ['B0', 'B1', 'B2', 'B3'],
                'class': 'Standards',
                'probabilities': [
                    {'No Burden': 0.1, 'Detail': 0.1, 'Reporting': 0.1, 'Standards': 0.7},
                    {'No Burden': 0.1, 'Detail': 0.1, 'Reporting': 0.1, 'Standards': 0.7},
                    {'No Burden': 0.1, 'Detail': 0.7, 'Reporting': 0.1, 'Standards': 0.1},
                    {'No Burden': 0.1, 'Detail': 0.7, 'Reporting': 0.1, 'Standards': 0.1}
                ]
            },
            {
                'node-ids': ['B4'],
                'class': 'Standards',
                'probabilities': [
                    {'No Burden': 0.1, 'Detail': 0.1, 'Reporting': 0.1, 'Standards': 0.7},
                ]
            },
            {
                'node-ids': ['B5'],
                'class': 'Reporting',
                'probabilities': [
                    {'No Burden': 0.1, 'Detail': 0.1, 'Reporting': 0.7, 'Standards': 0.1},
                ]
            }
        ]

        for r, e in zip(result, expected):
            for diff in list(dictdiffer.diff(r, e)):
                print(diff)
            self.assertDictEqual(r, e)

    def test_burden_document_view_3(self):

        labels = {
            0: 'No Burden',
            1: 'Detail',
            2: 'Reporting',
            3: 'Standards'
        }
        data = [
            ('A0', 'B0', [''], 0, [0.1, 0.1, 0.1, 0.7]),
            ('A0', 'B1', [''], 0, [0.1, 0.1, 0.1, 0.7]),
            ('A0', 'B2', [''], 1, [0.1, 0.7, 0.1, 0.1]),
            ('A0', 'B3', [''], 1, [0.1, 0.7, 0.1, 0.1]),
            ('A0', 'B4', [''], 0, [0.1, 0.1, 0.1, 0.7]),
            ('A0', 'B5', [''], 1, [0.1, 0.1, 0.7, 0.1])
        ]
        result = document_burden_view(data, labels, bridges=['Detail'], ignore=['No Burden', 'Detail'])
        self.assertListEqual(result, [])

        labels = {
            0: 'No Burden',
            1: 'Detail',
            2: 'Reporting',
            3: 'Standards'
        }
        data = [
            ('A0', 'B0', [''], 0, [0.1, 0.1, 0.1, 0.7]),
            ('A0', 'B1', [''], 0, [0.1, 0.1, 0.1, 0.7]),
            ('A0', 'B2', [''], 0, [0.1, 0.7, 0.1, 0.1]),
            ('A0', 'B3', [''], 0, [0.1, 0.7, 0.1, 0.1]),
            ('A0', 'B4', [''], 0, [0.1, 0.1, 0.1, 0.7]),
            ('A0', 'B5', [''], 1, [0.1, 0.1, 0.7, 0.1])
        ]
        result = document_burden_view(data, labels, bridges=['Detail'], ignore=['No Burden'])
        expected = [
            {
                'node-ids': ['B5'],
                'class': 'Detail',
                'probabilities': [
                    {'No Burden': 0.1, 'Detail': 0.1, 'Reporting': 0.7, 'Standards': 0.1},
                ]
            }
        ]
        for r, e in zip(result, expected):
            for diff in list(dictdiffer.diff(r, e)):
                print(diff)
            self.assertDictEqual(r, e)
