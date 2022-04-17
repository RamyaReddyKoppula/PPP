import activate
import numpy as np
import random
import math
import unittest
import pytest
import modulefinder

def test_sigmoid():
    assert activate.sigmoid(0) ==  0.5
def test_sigmoid_gra():
    assert activate.gradient_sigmoid(0) ==  0.25


def test_Softmax():
    assert activate.Softmax(3) == 1
def test_gradient_Softmax():
    assert activate.gradient_Softmax(4) == 0


def test_TanH():
    assert activate.TanH(10) == pytest.approx(1)
def test_gradient_TanH():
    assert activate.gradient_TanH(0) == 1


def test_ReLU():
    arr = np.array([[-1,2,3],[1,2,3]])
    hrr = activate.ReLU(arr)
    expected = np.array([[0,2,3],[1,2,3]])
    assert (hrr == expected).all()
def test_gradient_ReLU():
    z = np.array([[ 1.5, -1.5,  2],
       [ -2.5,  0,  0.5 ],
       [-0.5,  0.1,  0.1]])
    zz= activate.gradient_ReLU(z)
    rr=np.array([[ 1,  0,  1],[ 0,  1,  1],[ 0,  1,  1]])
    assert (zz == rr).all()

def test_LeakyReLU():
    arr = np.array([[-1,2,3],[1,2,3]])
    hrr = activate.LeakyReLU(arr)
    expected = np.array([[-0.2,2,3],[1,2,3]])
    assert (hrr == expected).all()
def test_gradient_LeakyReLU():
    z = np.array([[-1,2,3],[1,2,3]])
    zz= activate.gradient_LeakyReLU(z)
    rr=np.array([[0.2,1,1],[1,1,1]])
    assert (zz == rr).all()

def test_ELU():
    lr = np.array([[0,-1.5,3],[-1,2,3]])
    lru = activate.ELU(lr)
    expected = np.array([[ 0,pytest.approx(-0.07768698),3],
                        [pytest.approx(-0.06321206),2,3]])
    assert (lru == expected).all()
def test_gradient_ELU():
    ll = np.array([[0,-1.5,3],[-1,2,3]])
    rr= activate.gradient_ELU(ll)
    ex=np.array([[1,pytest.approx(0.02231302),1],
                [pytest.approx(0.03678794), 1,1]])
    assert (rr == ex).all()

def test_SoftPlus():
    assert activate.SoftPlus(3) == pytest.approx(3.04858735)
def test_gradient_SoftPlus():
    z = np.array([[0,-1.5,3],[-1,2,3]])
    zz= activate.gradient_SoftPlus(z)
    rr=np.array([[pytest.approx(0.5), pytest.approx(0.18242552), pytest.approx(0.95257413)],
                 [pytest.approx(0.26894142), pytest.approx(0.88079708), pytest.approx(0.95257413)]])
    assert (zz == rr).all()

test_sigmoid()
test_sigmoid_gra()
test_Softmax()
test_gradient_Softmax()
test_TanH()
test_gradient_TanH()
test_ReLU()
test_gradient_ReLU()
test_LeakyReLU()
test_gradient_LeakyReLU()
test_ELU()
test_SoftPlus()
test_gradient_ELU()
test_gradient_SoftPlus()

#Loss functions