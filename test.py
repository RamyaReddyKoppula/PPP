import activate
import numpy as np
import pytest
import PCA

#Activation Functions
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

#PCA

def test_pca():
    data=[[1, 0,1, 1, 0, 1], [0, 1, 0, 0, 2, 1]]
    #data is reduced to two dimensional
    pro=PCA.PCA(data, 2)
    rey= np.array([[pytest.approx(1.41421356e+00),pytest.approx(1.11022302e-16)], 
                                [pytest.approx(-1.41421356e+00), pytest.approx(-1.11022302e-16)]])
    assert (pro == rey).all() 
def test_dataset_MinMax():
    dataset_in = np.array([[50, 30], [20, 90]])
    dataset_out =np.array ([[20, 50], [30, 90]])
    pro=PCA.dataset_MinMax(dataset_in)
    assert (pro == dataset_out).all()

test_pca()
test_dataset_MinMax()

#confusion Matrix
def test_CM():
    dataset_test = np.array([1, 3, 3, 2, 5, 5, 3, 2, 1, 4, 3, 2, 1, 1, 2])
    dataset_predict =np.array([1, 2, 3, 4, 2, 3, 3, 2, 1, 2, 3, 1, 5, 1, 1])
    dataset_out =np.array( [[3., 0., 0., 0., 1.],
                            [2., 1., 0., 1., 0.],
                            [0., 1., 3., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 1., 1., 0., 0.]])
    cm=activate.comp_confmat(dataset_test,dataset_predict)
    assert (cm == dataset_out).all()
test_CM()

Matrix=np.array([[6, 2,0], [1, 6,0], [1, 1,8]])
def test_pma():
    obt=activate.precision_macro_average(Matrix)
    out=pytest.approx(0.805)
    assert (obt == out).all()
test_pma()