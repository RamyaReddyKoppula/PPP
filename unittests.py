import unittest
import norm
import numpy as np
import random
import math

dataset_in = [[50, 30], [20, 90]]
dataset_out = [[20, 50], [30, 90]]

class Testnorm(unittest.TestCase):
    def test_dataset_MinMax(self):
        self.assertEqual(norm.dataset_MinMax(dataset_in), (dataset_out))

    def test_normalization(self):
         self.assertTrue(norm.normalization(dataset_in, dataset_out), ([[1, 0], [0, 1]]))
    def test_to_categorical(self):
    #"""Simple test for One hot."""
        
        X =[1, 0, 2]

        y = [[0, 1, 0], 
              [1, 0, 0],  
              [0, 0, 1]]

        self.assertTrue(norm.to_categorical(X), ([[0, 1, 0],[1, 0, 0],[0, 0, 1]]))
    def test_accuracy_score(self):
        y_preds = [0, 2, 1, 3]
        y_true = [0, 1, 2, 3]
        self.assertRaises(norm.accuracy_score(y_true, y_preds), (0.0))
if __name__ == '__main__':
    unittest.main()