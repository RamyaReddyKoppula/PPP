import unittest
import PCA
import pytest
import numpy as np

dataset_in = [[50, 30], [20, 90]]
dataset_out = [[20, 50], [30, 90]]
data=[[1, 0,1, 1, 0, 1], [0, 1, 0, 0, 2, 1]]
class Testnorm(unittest.TestCase):
    def test_dataset_MinMax(self):
        self.assertEqual(PCA.dataset_MinMax(dataset_in), (dataset_out))

    # def test_normalization(self):
    #      self.assertTrue(PCA.normalization(dataset_in, dataset_out), ([[1, 0], [0, 1]])) 
if __name__ == '__main__':
    unittest.main()
    #test_pca()
