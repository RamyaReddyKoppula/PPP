import numpy as np
import math

def sigmoid(a):
    return 1 / (1 + np.exp(-a))
    #derivative of sigmoid
def gradient_sigmoid(a):
        return sigmoid(a) * (1 - sigmoid(a))

def Softmax(a):
        e_x = np.exp(a - np.max(a, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
def gradient_Softmax(a):
        p = Softmax(a)
        return p * (1 - p)

def TanH(a):
        return 2 / (1 + np.exp(-2*a)) - 1
def gradient_TanH(a):
    return 1 - np.power(TanH(a), 2)

def ReLU(a):
        return np.where(a >= 0, a, 0)
def gradient_ReLU(a):
    return np.where(a >= 0, 1, 0)       



def LeakyReLU(x, alpha=0.2):
    return np.where(x >= 0, x, alpha * x)
def gradient_LeakyReLU(x, alpha=0.2):
    return np.where(x >= 0, 1, alpha)


def ELU(x, alpha=0.1):
    return np.where(x >= 0.0, x, alpha * (np.exp(x) - 1))

def gradient_ELU(x, alpha=0.1):
    return np.where(x >= 0.0, 1, ELU(x) + alpha)

def SoftPlus(x):
    return np.log(1 + np.exp(x))

def gradient_SoftPlus(x):
    return 1 / (1 + np.exp(-x))

#Confusion Matrix

def comp_confmat(actual, predicted):

    # extract the different classes
    classes = np.unique(actual)

    # initialize the confusion matrix
    confmat = np.zeros((len(classes), len(classes)))

    # loop across the different combinations of actual / predicted classes
    for i in range(len(classes)):
        for j in range(len(classes)):

           # count the number of instances in each combination of actual / predicted classes
           confmat[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))  
    return confmat 
def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()

def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows

def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 