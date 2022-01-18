import unittest
import norm
dataset_in = [[50, 30], [20, 90]]
dataset_out = [[20, 50], [30, 90]]

class Testnorm(unittest.TestCase):
    def test_dataset_MinMax(self):
        self.assertEqual(norm.dataset_MinMax(dataset_in), (dataset_out))

    def test_normalization(self):
        self.assertEqual(norm.normalization(dataset_in, dataset_out), ([[1, 0], [0, 1]]))



if __name__ == '__main__':
    unittest.main()
